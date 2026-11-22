"""Efficient publishable evaluation plan for the revised higher-ed case study.

This driver deliberately separates four questions that should not be mixed:

1. PRIMARY STABILITY: 5x5 repeated, paired CV with fixed, prespecified model
   settings.  Repeating CV is affordable because tuning is not repeated inside
   every outer fold.
2. TUNING FAIRNESS: a separate nested 5-fold audit for the conventional
   baselines, plus a 1-vs-4-estimator TabPFN sensitivity on the same five folds.
3. ROBUSTNESS: proxy-cohort GroupKFold and registration-field ablations using
   only logistic regression and TabPFN-1.
4. OPERATIONS: separately fitted binary Dropout-vs-rest models generate queue,
   break-even, and subgroup results.  Multiclass F1 remains a diagnostic; it is
   not silently reused as an outreach policy.

The split objects are shared within each experiment so all comparisons remain
paired.  This is an exploratory case study, not a preregistered benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import higher_ed_tfm_revised as core


PRIMARY_MODELS = ["logistic", "catboost", "histgb", "tabpfn_1"]
ROBUSTNESS_MODELS = ["logistic", "tabpfn_1"]
BINARY_MODELS = ["logistic_binary", "tabpfn_1_binary"]


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/uci697_student_dropout_snapshot.csv")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("publishable_outputs"))
    parser.add_argument("--profile", choices=["smoke", "full"], default="full")
    parser.add_argument("--skip-tuning-audit", action="store_true")
    return parser.parse_args()


def config(profile: str) -> core.RunConfig:
    if profile == "smoke":
        return core.RunConfig(2, 1, 2, 100, (1,))
    return core.RunConfig(5, 5, 3, 1000, (1, 4))


def fixed_preprocessor(features: list[str]) -> ColumnTransformer:
    """Training-fold-only encoding shared by logistic and histogram boosting."""

    cats = core.categorical_for(features)
    nums = [c for c in features if c not in cats]
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
            ("num", StandardScaler(), nums),
        ],
        sparse_threshold=0,
    )


def fixed_multiclass_predict(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, float]:
    """Fit one prespecified multiclass model without looking at outer-test data."""

    started = perf_counter()
    if model_name == "logistic":
        model = Pipeline(
            [
                ("prepare", fixed_preprocessor(features)),
                (
                    "model",
                    LogisticRegression(
                        C=1.0, solver="lbfgs", max_iter=4000, random_state=core.SEED
                    ),
                ),
            ]
        ).fit(X_train[features], y_train)
        raw = model.predict_proba(X_test[features])
        classes = model.named_steps["model"].classes_
    elif model_name == "histgb":
        model = Pipeline(
            [
                ("prepare", fixed_preprocessor(features)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.08,
                        max_leaf_nodes=31,
                        l2_regularization=1.0,
                        early_stopping=True,
                        random_state=core.SEED,
                    ),
                ),
            ]
        ).fit(X_train[features], y_train)
        raw = model.predict_proba(X_test[features])
        classes = model.named_steps["model"].classes_
    elif model_name == "catboost":
        from catboost import CatBoostClassifier

        cats = core.categorical_for(features)
        train, test = X_train[features].copy(), X_test[features].copy()
        for column in cats:
            train[column] = train[column].astype(str)
            test[column] = test[column].astype(str)
        model = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=5,
            loss_function="MultiClass",
            random_seed=core.SEED,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
        ).fit(train, y_train, cat_features=cats)
        raw, classes = model.predict_proba(test), model.classes_
    elif model_name == "tabpfn_1":
        model = core.fit_tabpfn(X_train, y_train, features, estimators=1)
        raw, classes = model.predict_proba(X_test[features]), model.classes_
    else:
        raise ValueError(model_name)
    return core.reorder_proba(raw, classes), perf_counter() - started


def binary_predict(
    model_name: str,
    X_train: pd.DataFrame,
    y_train_binary: np.ndarray,
    X_test: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, float]:
    """Fit a genuine binary Dropout-vs-rest model for outreach ranking."""

    started = perf_counter()
    if model_name == "logistic_binary":
        model = Pipeline(
            [
                ("prepare", fixed_preprocessor(features)),
                (
                    "model",
                    LogisticRegression(
                        C=1.0, solver="lbfgs", max_iter=4000, random_state=core.SEED
                    ),
                ),
            ]
        ).fit(X_train[features], y_train_binary)
        raw = model.predict_proba(X_test[features])
        classes = model.named_steps["model"].classes_
    elif model_name == "tabpfn_1_binary":
        model = core.fit_tabpfn(
            X_train, y_train_binary, features, estimators=1
        )
        raw, classes = model.predict_proba(X_test[features]), model.classes_
    else:
        raise ValueError(model_name)

    # Explicit class lookup prevents an estimator-specific probability-column
    # order from silently reversing risk.
    positive_column = int(np.flatnonzero(np.asarray(classes) == 1)[0])
    return np.asarray(raw)[:, positive_column], perf_counter() - started


def multiclass_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    predicted = p.argmax(axis=1)
    dropout = (y == 0).astype(int)
    return {
        "macro_f1": f1_score(y, predicted, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y, predicted),
        "multiclass_log_loss": log_loss(y, p, labels=[0, 1, 2]),
        "dropout_ap_from_multiclass": average_precision_score(dropout, p[:, 0]),
    }


def primary_paired_effects(
    primary: pd.DataFrame, cfg: core.RunConfig
) -> pd.DataFrame:
    """Estimate paired model and checkpoint effects for the primary task.

    Only multiclass outcomes belong here.  Queue metrics come from the
    separately fitted binary policy models and are summarized in their own
    files; mixing the two tasks would make the comparison hard to interpret.
    """

    rows: list[dict] = []
    metrics = ["macro_f1", "multiclass_log_loss", "dropout_ap_from_multiclass"]
    for metric in metrics:
        for moment in core.MOMENTS:
            wide = primary.loc[primary.moment.eq(moment)].pivot(
                index="split_id", columns="model", values=metric
            )
            for model in [name for name in wide.columns if name != "logistic"]:
                difference = wide[model] - wide["logistic"]
                if metric == "multiclass_log_loss":
                    difference = -difference
                rows.append(
                    {
                        "comparison_type": "model_effect_vs_logistic",
                        "moment": moment,
                        "metric": metric,
                        "first": model,
                        "second": "logistic",
                        **core.corrected_repeated_cv_interval(
                            difference, cfg.outer_splits
                        ),
                    }
                )

        for model in primary.model.unique():
            wide = primary.loc[primary.model.eq(model)].pivot(
                index="split_id", columns="moment", values=metric
            )
            difference = wide["after_first_semester"] - wide["registration"]
            if metric == "multiclass_log_loss":
                difference = -difference
            rows.append(
                {
                    "comparison_type": "checkpoint_effect",
                    "moment": "after_first_semester_vs_registration",
                    "metric": metric,
                    "first": "after_first_semester",
                    "second": "registration",
                    "model": model,
                    **core.corrected_repeated_cv_interval(
                        difference, cfg.outer_splits
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_multiclass_split(
    *,
    analysis: str,
    split_id: str,
    repeat: int,
    fold: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    X: pd.DataFrame,
    y: np.ndarray,
    moments: dict[str, list[str]],
    models: list[str],
) -> list[dict]:
    rows = []
    for moment, features in moments.items():
        for model_name in models:
            probability, runtime = fixed_multiclass_predict(
                model_name, X.iloc[train_idx], y[train_idx],
                X.iloc[test_idx], features
            )
            rows.append(
                {
                    "analysis": analysis,
                    "split_id": split_id,
                    "repeat": repeat,
                    "fold": fold,
                    "moment": moment,
                    "model": model_name,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "n_features": len(features),
                    "runtime_seconds": runtime,
                    **multiclass_metrics(y[test_idx], probability),
                }
            )
    return rows


def run_binary_split(
    *,
    split_id: str,
    repeat: int,
    fold: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    X: pd.DataFrame,
    y: np.ndarray,
    models: list[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Fit binary models and return row-level OOF risks plus queue curves."""

    y_binary = (y == 0).astype(int)
    prediction_rows, queue_rows = [], []
    selected_models = BINARY_MODELS if models is None else models
    for moment, features in core.MOMENTS.items():
        for model_name in selected_models:
            risk, runtime = binary_predict(
                model_name,
                X.iloc[train_idx], y_binary[train_idx],
                X.iloc[test_idx], features,
            )
            truth = y_binary[test_idx]
            common = {
                "analysis": "repeated_stratified_binary_cv",
                "split_id": split_id,
                "repeat": repeat,
                "fold": fold,
                "moment": moment,
                "model": model_name,
                "runtime_seconds": runtime,
            }
            for position, row_id in enumerate(test_idx):
                prediction_rows.append(
                    {
                        **common,
                        "row_id": int(row_id),
                        "target": int(y[row_id]),
                        "dropout": int(truth[position]),
                        "p_dropout": float(risk[position]),
                    }
                )
            for capacity in np.arange(0.05, 0.51, 0.05):
                selected = core.top_fraction_mask(risk, float(capacity))
                reached = int(np.sum(selected & (truth == 1)))
                queue_rows.append(
                    {
                        **common,
                        "capacity": float(capacity),
                        "selected_n": int(selected.sum()),
                        "dropout_n": int(truth.sum()),
                        "true_dropout_labels_reached": reached,
                        "queue_recall": recall_score(truth, selected, zero_division=0),
                        # This name is intentionally literal.  The public table
                        # does not reveal why each student left, so this is not
                        # equivalent to "students who needed help."
                        "queue_dropout_label_share": precision_score(
                            truth, selected, zero_division=0
                        ),
                        "dropout_ap": average_precision_score(truth, risk),
                        "dropout_roc_auc": roc_auc_score(truth, risk),
                        "dropout_brier": brier_score_loss(truth, risk),
                    }
                )
    return prediction_rows, queue_rows


def tuning_audit(
    splits: list[tuple[np.ndarray, np.ndarray]],
    X: pd.DataFrame,
    y: np.ndarray,
    cfg: core.RunConfig,
) -> pd.DataFrame:
    """Nested audit on the first five outer folds, separate from primary CV."""

    rows = []
    audit_models = ["logistic", "catboost", "histgb", "tabpfn_1", "tabpfn_4"]
    for fold, (train_idx, test_idx) in enumerate(splits[: cfg.outer_splits]):
        for moment, features in core.MOMENTS.items():
            for model_name in audit_models:
                p, chosen, runtime = core.predict_model(
                    model_name,
                    X.iloc[train_idx], y[train_idx],
                    X.iloc[test_idx], features,
                    cfg.inner_splits,
                )
                rows.append(
                    {
                        "analysis": "five_fold_nested_tuning_audit",
                        "fold": fold,
                        "moment": moment,
                        "model": model_name,
                        "chosen_parameters": json.dumps(chosen, sort_keys=True),
                        "runtime_seconds": runtime,
                        **multiclass_metrics(y[test_idx], p),
                    }
                )
    return pd.DataFrame(rows)


def binary_break_even(queue: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the timing threshold and transparent net-value scenarios."""

    mean_curve = (
        queue.groupby(["model", "moment", "capacity"], as_index=False)
        .agg(
            queue_recall=("queue_recall", "mean"),
            dropout_label_share=("queue_dropout_label_share", "mean"),
            selected_n=("selected_n", "mean"),
            dropout_n=("dropout_n", "mean"),
        )
    )
    wide = mean_curve.pivot(
        index=["model", "capacity"], columns="moment", values="queue_recall"
    ).reset_index()
    wide["early_effectiveness_break_even_ratio"] = (
        wide["after_first_semester"] / wide["registration"]
    )
    wide = wide.rename(
        columns={
            "registration": "registration_queue_recall",
            "after_first_semester": "late_queue_recall",
        }
    )

    scenarios = []
    for row in wide.itertuples(index=False):
        for relative_effect in np.arange(0.50, 2.01, 0.05):
            for early_cost, late_cost in [(0, 0), (0.05, 0.05), (0.10, 0.05)]:
                early = (
                    row.registration_queue_recall * relative_effect
                    - row.capacity * early_cost
                )
                late = row.late_queue_recall - row.capacity * late_cost
                scenarios.append(
                    {
                        "model": row.model,
                        "capacity": row.capacity,
                        "relative_early_effectiveness": relative_effect,
                        "early_outreach_cost_index": early_cost,
                        "late_outreach_cost_index": late_cost,
                        "registration_net_value_index": early,
                        "late_net_value_index": late,
                        "preferred_checkpoint": (
                            "registration" if early > late else "after_first_semester"
                        ),
                    }
                )
    return wide, pd.DataFrame(scenarios)


def binary_subgroups(
    predictions: pd.DataFrame,
    X: pd.DataFrame,
    cfg: core.RunConfig,
) -> pd.DataFrame:
    """Subgroup diagnostics with row bootstrap intervals.

    Each student has one held-out risk from the five-fold binary audit. We
    aggregate defensively by row id before bootstrapping.
    """

    averaged = (
        predictions.loc[predictions.moment.eq("after_first_semester")]
        .groupby(["model", "row_id", "target", "dropout"], as_index=False)
        .p_dropout.mean()
    )
    attrs = pd.DataFrame(
        {
            "row_id": np.arange(len(X)),
            "Gender": X.Gender.map({0: "Female", 1: "Male"}),
            "Scholarship": X["Scholarship holder"].map({0: "No", 1: "Yes"}),
            "Debtor": X.Debtor.map({0: "No", 1: "Yes"}),
            "Age band": pd.cut(
                X["Age at enrollment"], [-np.inf, 19, 24, np.inf],
                labels=["19 or younger", "20-24", "25 or older"]
            ).astype(str),
        }
    )
    averaged = averaged.merge(attrs, on="row_id", validate="many_to_one")
    rows = []
    rng = np.random.default_rng(core.SEED)
    for model_name, model_rows in averaged.groupby("model"):
        model_rows = model_rows.copy()
        model_rows["selected"] = core.top_fraction_mask(
            model_rows.p_dropout.to_numpy(), 0.20
        )
        for attribute in ["Gender", "Scholarship", "Debtor", "Age band"]:
            for level, group in model_rows.groupby(attribute):
                if len(group) < 50 or group.dropout.sum() < 10:
                    continue
                functions = {
                    "dropout_prevalence": lambda d: d.dropout.mean(),
                    "mean_predicted_dropout": lambda d: d.p_dropout.mean(),
                    "calibration_gap": lambda d: d.p_dropout.mean() - d.dropout.mean(),
                    "selection_rate": lambda d: d.selected.mean(),
                    "queue_recall": lambda d: d.loc[d.dropout.eq(1), "selected"].mean(),
                    "false_negative_rate": lambda d: 1 - d.loc[d.dropout.eq(1), "selected"].mean(),
                    "dropout_brier": lambda d: np.mean((d.p_dropout - d.dropout) ** 2),
                }
                for metric, function in functions.items():
                    estimate = float(function(group))
                    boot = []
                    for _ in range(cfg.subgroup_bootstraps):
                        sample = group.iloc[rng.integers(0, len(group), len(group))]
                        boot.append(float(function(sample)))
                    low, high = np.quantile(boot, [0.025, 0.975])
                    rows.append(
                        {
                            "model": model_name,
                            "attribute": attribute,
                            "level": level,
                            "n": len(group),
                            "dropout_n": int(group.dropout.sum()),
                            "metric": metric,
                            "estimate": estimate,
                            "ci_low": low,
                            "ci_high": high,
                        }
                    )
    return pd.DataFrame(rows)


def charts(
    primary: pd.DataFrame,
    break_even: pd.DataFrame,
    robustness: pd.DataFrame,
    out: Path,
) -> None:
    summary = (
        primary.groupby(["model", "moment"], as_index=False)
        .agg(mean_macro_f1=("macro_f1", "mean"), sd_macro_f1=("macro_f1", "std"))
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    for model, rows in summary.groupby("model"):
        rows = rows.set_index("moment").loc[list(core.MOMENTS)].reset_index()
        ax.errorbar(
            [0, 1], rows.mean_macro_f1, yerr=rows.sd_macro_f1,
            marker="o", capsize=4, label=model
        )
    ax.set_xticks([0, 1], ["Registration", "After first semester"])
    ax.set_ylabel("Macro F1 across paired outer folds")
    ax.set_title("The information checkpoint matters more than the model")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "primary_repeated_cv_macro_f1.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for model, rows in break_even.groupby("model"):
        ax.plot(
            100 * rows.capacity,
            rows.early_effectiveness_break_even_ratio,
            marker="o", label=model
        )
    ax.axhline(1.0, linestyle="--", color="#555555", linewidth=1)
    ax.set(
        xlabel="Students selected for review (%)",
        ylabel="Early/late support effectiveness needed for break-even",
        title="Prediction timing becomes a decision assumption",
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "binary_timing_break_even.png", dpi=180)
    plt.close(fig)

    if not robustness.empty:
        table = (
            robustness.groupby(["analysis", "moment", "model"], as_index=False)
            .macro_f1.mean()
        )
        table.to_csv(out / "robustness_table_compact.csv", index=False)


def main() -> None:
    parsed = args()
    cfg = config(parsed.profile)
    out = parsed.output_dir
    out.mkdir(parents=True, exist_ok=True)
    X, y, target_text = core.load_snapshot(parsed.input)

    # Smoke mode validates orchestration, output schemas, summaries, and charts
    # with fast sklearn models. TabPFN/CatBoost have direct unit checks and
    # would turn a two-fold smoke test into a long benchmark run.
    primary_models = (
        PRIMARY_MODELS if parsed.profile == "full" else ["logistic", "histgb"]
    )
    binary_models = (
        BINARY_MODELS if parsed.profile == "full" else ["logistic_binary"]
    )
    robustness_models = (
        ROBUSTNESS_MODELS if parsed.profile == "full" else ["logistic"]
    )

    splitter = RepeatedStratifiedKFold(
        n_splits=cfg.outer_splits,
        n_repeats=cfg.outer_repeats,
        random_state=core.SEED,
    )
    splits = list(splitter.split(X, y))

    primary_rows, binary_prediction_rows, binary_queue_rows = [], [], []
    for split_number, (train_idx, test_idx) in enumerate(splits):
        repeat, fold = divmod(split_number, cfg.outer_splits)
        split_id = f"r{repeat:02d}_f{fold:02d}"
        primary_rows.extend(
            run_multiclass_split(
                analysis="primary_repeated_stratified_cv",
                split_id=split_id,
                repeat=repeat,
                fold=fold,
                train_idx=train_idx,
                test_idx=test_idx,
                X=X, y=y,
                moments=core.MOMENTS,
                models=primary_models,
            )
        )
        # Binary queue analysis is a separate operational audit.  One complete
        # five-fold pass gives every student exactly one held-out binary risk
        # and avoids multiplying the most expensive CPU model by five repeats.
        # The 5x5 multiclass experiment above remains the primary stability
        # analysis; queue uncertainty is reported across these five folds.
        if repeat == 0:
            p_rows, q_rows = run_binary_split(
                split_id=split_id,
                repeat=repeat,
                fold=fold,
                train_idx=train_idx,
                test_idx=test_idx,
                X=X, y=y,
                models=binary_models,
            )
            binary_prediction_rows.extend(p_rows)
            binary_queue_rows.extend(q_rows)
        print(
            f"Completed primary split {split_number + 1}/{len(splits)} "
            f"({split_id})",
            flush=True,
        )

    primary = pd.DataFrame(primary_rows)
    binary_predictions = pd.DataFrame(binary_prediction_rows)
    binary_queues = pd.DataFrame(binary_queue_rows)

    # Grouped stress test: five folds, macro fields excluded from features.
    groups = core.proxy_cohort_ids(X)
    grouped_moments = {
        name: [c for c in columns if c not in core.MACRO_FIELDS]
        for name, columns in core.MOMENTS.items()
    }
    robustness_rows = []
    grouped_splits = 5 if parsed.profile == "full" else 2
    for fold, (train_idx, test_idx) in enumerate(
        GroupKFold(n_splits=grouped_splits).split(X, y, groups)
    ):
        robustness_rows.extend(
            run_multiclass_split(
                analysis="proxy_cohort_group_cv",
                split_id=f"group_f{fold:02d}",
                repeat=0, fold=fold,
                train_idx=train_idx, test_idx=test_idx,
                X=X, y=y,
                moments=grouped_moments,
                models=robustness_models,
            )
        )
        print(f"Completed proxy-group fold {fold + 1}/{grouped_splits}", flush=True)

    # Registration contract ablations use the first repeat's five paired folds.
    ablations = {
        "registration_full": core.ADMISSION,
        "registration_without_payment_scholarship": [
            c for c in core.ADMISSION if c not in core.TIMING_SENSITIVE_FIELDS
        ],
        "registration_without_macro_or_payment": [
            c for c in core.ADMISSION
            if c not in core.TIMING_SENSITIVE_FIELDS + core.MACRO_FIELDS
        ],
    }
    for fold, (train_idx, test_idx) in enumerate(splits[: cfg.outer_splits]):
        for label, features in ablations.items():
            robustness_rows.extend(
                run_multiclass_split(
                    analysis="registration_contract_ablation",
                    split_id=f"{label}_f{fold:02d}",
                    repeat=0, fold=fold,
                    train_idx=train_idx, test_idx=test_idx,
                    X=X, y=y,
                    moments={label: features},
                    models=robustness_models,
                )
            )
        print(
            f"Completed registration-ablation fold {fold + 1}/{cfg.outer_splits}",
            flush=True,
        )
    robustness = pd.DataFrame(robustness_rows)

    pairs = primary_paired_effects(primary, cfg)
    break_even, scenarios = binary_break_even(binary_queues)
    subgroups = binary_subgroups(binary_predictions, X, cfg)

    audit = pd.DataFrame()
    if not parsed.skip_tuning_audit:
        audit = tuning_audit(splits, X, y, cfg)

    primary.to_csv(out / "primary_multiclass_fold_metrics.csv", index=False)
    pairs.to_csv(out / "primary_paired_effects_corrected_intervals.csv", index=False)
    binary_predictions.to_csv(out / "binary_out_of_fold_predictions.csv", index=False)
    binary_queues.to_csv(out / "binary_queue_capacity_by_fold.csv", index=False)
    break_even.to_csv(out / "binary_timing_break_even.csv", index=False)
    scenarios.to_csv(out / "binary_net_value_scenarios.csv", index=False)
    subgroups.to_csv(out / "binary_subgroup_diagnostics_with_intervals.csv", index=False)
    robustness.to_csv(out / "robustness_fold_metrics.csv", index=False)
    audit.to_csv(out / "nested_tuning_and_tabpfn_ensemble_audit.csv", index=False)
    pd.DataFrame(
        {"row_id": np.arange(len(X)), "target": target_text, "proxy_cohort": groups}
    ).to_csv(out / "proxy_cohort_manifest.csv", index=False)
    charts(primary, break_even, robustness, out)

    (out / "run_manifest.json").write_text(
        json.dumps(
            {
                "profile": parsed.profile,
                "outer_splits": cfg.outer_splits,
                "outer_repeats": cfg.outer_repeats,
                "primary_models": primary_models,
                "binary_models": binary_models,
                "binary_queue_outer_folds": cfg.outer_splits,
                "binary_queue_outer_repeats": 1,
                "robustness_models": robustness_models,
                "interpretation_limits": [
                    "Proxy cohorts are inferred, not verified enrollment cohorts.",
                    "Break-even outputs vary assumptions; they do not estimate treatment effects.",
                    "Dropout is a source-defined administrative label; the public table does not reveal why each student left.",
                    "Repeated-CV corrected intervals are approximate, not external-validity intervals.",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        primary.groupby(["moment", "model"])
        .agg(macro_f1=("macro_f1", "mean"), log_loss=("multiclass_log_loss", "mean"))
    )
    print(f"\nOutputs: {out.resolve()}")


if __name__ == "__main__":
    main()
