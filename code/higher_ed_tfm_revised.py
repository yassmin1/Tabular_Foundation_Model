"""Revised, leakage-conscious evaluation for the UCI 697 student dataset.

WHY THIS SCRIPT EXISTS
----------------------
The original case study used one train/test split.  That was useful for a
reproducible demonstration, but a single split cannot tell us whether a small
model ranking is stable.  This revision asks a more defensible question:

    Is the gain from waiting for first-semester information larger and more
    stable than the gain from choosing one strong model over another?

The script therefore keeps every comparison *paired*: all models and both
prediction moments see the same outer folds.  It also includes three stress
tests that are important for the educational interpretation:

1. proxy-cohort GroupKFold based on the three macroeconomic values;
2. a registration-time ablation of payment/scholarship fields whose timestamp
   cannot be verified from the released matrix; and
3. a decision analysis that separates predictive recall from an assumed
   benefit of intervening earlier.

The code is intentionally verbose.  Comments explain not only what each block
does, but why it is needed and what it does *not* prove.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.base import clone
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
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------------
# Reproducibility and the prediction contract
# ---------------------------------------------------------------------------

SEED = 42
CLASS_NAMES = ["Dropout", "Enrolled", "Graduate"]
CLASS_TO_INT = {name: i for i, name in enumerate(CLASS_NAMES)}
MACRO_FIELDS = ["Unemployment rate", "Inflation rate", "GDP"]
TIMING_SENSITIVE_FIELDS = [
    "Debtor",
    "Tuition fees up to date",
    "Scholarship holder",
]

CATEGORICAL = [
    "Marital Status",
    "Application mode",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]

ADMISSION_NUMERIC = [
    "Application order",
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
] + MACRO_FIELDS

FIRST_SEMESTER = [
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
]

SECOND_SEMESTER = [
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
]

ADMISSION = CATEGORICAL + ADMISSION_NUMERIC
MOMENTS = {
    "registration": ADMISSION,
    "after_first_semester": ADMISSION + FIRST_SEMESTER,
}


@dataclass(frozen=True)
class RunConfig:
    """Settings that change runtime but not the meaning of the experiment."""

    outer_splits: int
    outer_repeats: int
    inner_splits: int
    subgroup_bootstraps: int
    tabpfn_estimators: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeated paired evaluation of higher-education risk models."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/uci697_student_dropout_snapshot.csv"),
        help="Local UCI-697 snapshot produced by the original project.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--profile",
        choices=["smoke", "full"],
        default="full",
        help=(
            "smoke runs 2 folds once; full runs the planned 5x5 outer CV. "
            "The smoke profile validates code paths but is not publishable."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logistic", "catboost", "histgb", "tabpfn_1", "tabpfn_4"],
        help="Models to evaluate. Omit unavailable optional packages for a smoke test.",
    )
    parser.add_argument(
        "--skip-grouped",
        action="store_true",
        help="Skip the proxy-cohort GroupKFold stress test.",
    )
    parser.add_argument(
        "--skip-ablation",
        action="store_true",
        help="Skip the registration timing-field ablation.",
    )
    return parser.parse_args()


def config_for(profile: str) -> RunConfig:
    if profile == "smoke":
        return RunConfig(2, 1, 2, 100, (1,))
    return RunConfig(5, 5, 3, 1000, (1, 4))


def load_snapshot(path: Path) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """Load and validate the fixed public-data snapshot.

    Returning the text target as well as the encoded target keeps exported
    files readable while allowing estimators to work with stable integer class
    positions.
    """

    frame = pd.read_csv(path)
    required = set(ADMISSION + FIRST_SEMESTER + SECOND_SEMESTER + ["Target"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Snapshot is missing required columns: {missing}")
    if frame[list(required)].isna().any().any():
        raise ValueError("Unexpected missing values in the released snapshot.")

    target_text = frame.pop("Target").astype(str)
    y = target_text.map(CLASS_TO_INT)
    if y.isna().any():
        raise ValueError(f"Unexpected labels: {sorted(target_text.unique())}")

    # Hard leakage guard: second-semester evidence must never enter either
    # prediction moment.  This fails loudly if a future edit violates it.
    for moment, features in MOMENTS.items():
        leaked = sorted(set(features) & set(SECOND_SEMESTER))
        if leaked:
            raise AssertionError(f"{moment} includes future fields: {leaked}")
    return frame, y.to_numpy(dtype=int), target_text


def proxy_cohort_ids(X: pd.DataFrame) -> pd.Series:
    """Create an *imperfect* historical-context group identifier.

    The dataset has no cohort field.  Its three macroeconomic values form ten
    repeated tuples, each plausibly associated with an enrollment period.  The
    tuple is useful as a stress-test group, but must not be described as a
    verified academic-year identifier.
    """

    tuples = pd.MultiIndex.from_frame(X[MACRO_FIELDS])
    codes, _ = pd.factorize(tuples, sort=True)
    return pd.Series(codes, index=X.index, name="proxy_cohort")


def categorical_for(features: list[str]) -> list[str]:
    return [column for column in CATEGORICAL if column in features]


def preprocessing(features: list[str]) -> ColumnTransformer:
    """Fit all learned transformations inside the training fold.

    Integer category codes must not be treated as ordered measurements.  A
    course code of 9500 is not "larger" than a course code of 171, so categorical
    columns are one-hot encoded.  Numeric columns are standardized for the
    linear model; scaling is harmless for the histogram model.
    """

    cats = categorical_for(features)
    nums = [column for column in features if column not in cats]
    return ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cats,
            ),
            ("numeric", StandardScaler(), nums),
        ],
        sparse_threshold=0,
    )


def tune_sklearn_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    features: list[str],
    inner_splits: int,
) -> tuple[object, dict]:
    """Tune a baseline using training-only inner cross-validation.

    We optimize multiclass log loss because the operational analysis uses
    probabilities, not only hard labels.  The outer fold remains untouched.
    """

    prep = preprocessing(features)
    if model_name == "logistic":
        estimator = LogisticRegression(
            solver="lbfgs", max_iter=4000, random_state=SEED
        )
        grid = {"model__C": [0.1, 1.0, 10.0]}
    elif model_name == "histgb":
        estimator = HistGradientBoostingClassifier(
            loss="log_loss", random_state=SEED, early_stopping=True
        )
        grid = {
            "model__learning_rate": [0.03, 0.08],
            "model__max_leaf_nodes": [15, 31],
            "model__l2_regularization": [0.0, 1.0],
        }
    else:
        raise ValueError(model_name)

    pipe = Pipeline([("prepare", prep), ("model", estimator)])
    inner = StratifiedKFold(
        n_splits=inner_splits, shuffle=True, random_state=SEED
    )
    search = GridSearchCV(
        pipe,
        grid,
        scoring="neg_log_loss",
        cv=inner,
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train[features], y_train)
    return search.best_estimator_, search.best_params_


def tune_catboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    features: list[str],
    inner_splits: int,
) -> tuple[object, dict]:
    """Small, explicit CatBoost search evaluated only inside the outer train set."""

    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise RuntimeError("Install catboost or omit 'catboost' from --models") from exc

    cats = categorical_for(features)
    data = X_train[features].copy()
    for column in cats:
        data[column] = data[column].astype(str)

    candidates = [
        {"depth": 5, "learning_rate": 0.05, "l2_leaf_reg": 3},
        {"depth": 7, "learning_rate": 0.05, "l2_leaf_reg": 5},
        {"depth": 6, "learning_rate": 0.10, "l2_leaf_reg": 5},
    ]
    inner = StratifiedKFold(
        n_splits=inner_splits, shuffle=True, random_state=SEED
    )
    scored: list[tuple[float, dict]] = []
    for params in candidates:
        losses = []
        for inner_train, inner_valid in inner.split(data, y_train):
            model = CatBoostClassifier(
                iterations=500,
                loss_function="MultiClass",
                random_seed=SEED,
                thread_count=1,
                verbose=False,
                allow_writing_files=False,
                **params,
            )
            model.fit(
                data.iloc[inner_train],
                y_train[inner_train],
                cat_features=cats,
            )
            p = reorder_proba(
                model.predict_proba(data.iloc[inner_valid]), model.classes_
            )
            losses.append(log_loss(y_train[inner_valid], p, labels=[0, 1, 2]))
        scored.append((float(np.mean(losses)), params))
    best_loss, best = min(scored, key=lambda item: item[0])
    final = CatBoostClassifier(
        iterations=500,
        loss_function="MultiClass",
        random_seed=SEED,
        thread_count=1,
        verbose=False,
        allow_writing_files=False,
        **best,
    ).fit(data, y_train, cat_features=cats)
    return final, {**best, "inner_log_loss": best_loss}


def fit_tabpfn(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    features: list[str],
    estimators: int,
) -> object:
    """Fit TabPFN with a declared ensemble size; no outer-test tuning occurs."""

    try:
        from tabpfn import TabPFNClassifier
    except ImportError as exc:
        raise RuntimeError("Install tabpfn or omit TabPFN models from --models") from exc
    cats = categorical_for(features)
    cat_indices = [features.index(column) for column in cats]
    model = TabPFNClassifier(
        device="cpu",
        n_estimators=estimators,
        categorical_features_indices=cat_indices,
        random_state=SEED,
    )
    return model.fit(X_train[features], y_train)


def reorder_proba(proba: np.ndarray, classes: Iterable[int]) -> np.ndarray:
    """Force every estimator's probability columns into Dropout/Enrolled/Graduate order."""

    ordered = np.zeros((len(proba), 3), dtype=float)
    for source, target in enumerate(np.asarray(list(classes), dtype=int)):
        ordered[:, target] = proba[:, source]
    return ordered / ordered.sum(axis=1, keepdims=True)


def predict_model(
    name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    features: list[str],
    inner_splits: int,
) -> tuple[np.ndarray, dict, float]:
    """Fit one model and return ordered probabilities, tuning metadata, runtime."""

    started = perf_counter()
    if name in {"logistic", "histgb"}:
        model, chosen = tune_sklearn_model(
            name, X_train, y_train, features, inner_splits
        )
        raw = model.predict_proba(X_test[features])
        classes = model.named_steps["model"].classes_
    elif name == "catboost":
        model, chosen = tune_catboost(
            X_train, y_train, features, inner_splits
        )
        test = X_test[features].copy()
        for column in categorical_for(features):
            test[column] = test[column].astype(str)
        raw, classes = model.predict_proba(test), model.classes_
    elif name.startswith("tabpfn_"):
        estimators = int(name.rsplit("_", 1)[1])
        model = fit_tabpfn(X_train, y_train, features, estimators)
        raw, classes = model.predict_proba(X_test[features]), model.classes_
        chosen = {"n_estimators": estimators}
    else:
        raise ValueError(f"Unknown model: {name}")
    return reorder_proba(raw, classes), chosen, perf_counter() - started


def top_fraction_mask(scores: np.ndarray, fraction: float) -> np.ndarray:
    """Select exactly ceil(fraction*n) rows using stable tie-breaking."""

    n_selected = max(1, math.ceil(fraction * len(scores)))
    selected = np.zeros(len(scores), dtype=bool)
    selected[np.argsort(scores, kind="mergesort")[-n_selected:]] = True
    return selected


def score_predictions(y: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """Report multiclass quality plus binary Dropout-vs-rest operating metrics."""

    predicted = proba.argmax(axis=1)
    dropout = (y == CLASS_TO_INT["Dropout"]).astype(int)
    risk = proba[:, CLASS_TO_INT["Dropout"]]
    selected = top_fraction_mask(risk, 0.20)
    return {
        "macro_f1": f1_score(y, predicted, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y, predicted),
        "multiclass_log_loss": log_loss(y, proba, labels=[0, 1, 2]),
        "dropout_ap": average_precision_score(dropout, risk),
        "dropout_roc_auc": roc_auc_score(dropout, risk),
        "dropout_brier": brier_score_loss(dropout, risk),
        "queue_recall_20": recall_score(dropout, selected, zero_division=0),
        "queue_dropout_label_share_20": precision_score(
            dropout, selected, zero_division=0
        ),
    }


def evaluate_split(
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
    config: RunConfig,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Evaluate all models/moments on one shared outer split.

    The split is supplied once and reused, which is the key to paired model and
    checkpoint comparisons.
    """

    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    queue_rows: list[dict] = []
    for moment, features in moments.items():
        for model_name in models:
            proba, chosen, runtime = predict_model(
                model_name,
                X.iloc[train_idx],
                y[train_idx],
                X.iloc[test_idx],
                features,
                config.inner_splits,
            )
            common = {
                "analysis": analysis,
                "split_id": split_id,
                "repeat": repeat,
                "fold": fold,
                "moment": moment,
                "model": model_name,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_features": len(features),
            }
            metric_rows.append(
                {
                    **common,
                    "runtime_seconds": runtime,
                    "chosen_parameters": json.dumps(chosen, sort_keys=True),
                    **score_predictions(y[test_idx], proba),
                }
            )

            # Save out-of-fold predictions so every aggregate, subgroup table,
            # and chart can be traced back to individual held-out rows.
            for position, row_id in enumerate(test_idx):
                prediction_rows.append(
                    {
                        **common,
                        "row_id": int(row_id),
                        "target": int(y[row_id]),
                        "p_dropout": proba[position, 0],
                        "p_enrolled": proba[position, 1],
                        "p_graduate": proba[position, 2],
                    }
                )

            dropout = (y[test_idx] == 0).astype(int)
            risk = proba[:, 0]
            for capacity in np.arange(0.05, 0.51, 0.05):
                selected = top_fraction_mask(risk, float(capacity))
                queue_rows.append(
                    {
                        **common,
                        "capacity": float(capacity),
                        "selected_n": int(selected.sum()),
                        "dropout_n": int(dropout.sum()),
                        "true_dropout_labels_reached": int(
                            np.sum(selected & (dropout == 1))
                        ),
                        "queue_recall": recall_score(
                            dropout, selected, zero_division=0
                        ),
                        "queue_dropout_label_share": precision_score(
                            dropout, selected, zero_division=0
                        ),
                    }
                )
    return metric_rows, prediction_rows, queue_rows


def corrected_repeated_cv_interval(
    differences: pd.Series, n_splits: int, confidence: float = 0.95
) -> dict[str, float]:
    """Nadeau-Bengio-style corrected interval for repeated k-fold differences.

    Repeated folds are correlated because their training data overlap.  A naive
    standard error treats them as independent and is too small.  This correction
    inflates the variance by (1/n + test/train).  It remains an approximation,
    so the article should also show the fold-level distribution.
    """

    values = differences.dropna().to_numpy(dtype=float)
    n = len(values)
    mean = float(values.mean())
    if n < 2:
        return {"mean_difference": mean, "ci_low": np.nan, "ci_high": np.nan}
    variance = float(values.var(ddof=1))
    test_train_ratio = 1.0 / (n_splits - 1)
    corrected_se = math.sqrt((1.0 / n + test_train_ratio) * variance)
    critical = float(student_t.ppf((1.0 + confidence) / 2.0, df=n - 1))
    return {
        "mean_difference": mean,
        "ci_low": mean - critical * corrected_se,
        "ci_high": mean + critical * corrected_se,
        "folds_favoring_first": float(np.mean(values > 0)),
        "n_paired_folds": n,
    }


def paired_summaries(metrics: pd.DataFrame, config: RunConfig) -> pd.DataFrame:
    """Summarize model effects and checkpoint effects without mixing folds."""

    primary = metrics.loc[metrics.analysis.eq("repeated_stratified_cv")]
    rows = []
    for metric in ["macro_f1", "multiclass_log_loss", "dropout_ap", "queue_recall_20"]:
        # Model effect: each model minus logistic within a checkpoint and fold.
        for moment in MOMENTS:
            wide = primary.loc[primary.moment.eq(moment)].pivot(
                index="split_id", columns="model", values=metric
            )
            for model in [c for c in wide if c != "logistic"]:
                diff = wide[model] - wide["logistic"]
                if metric == "multiclass_log_loss":
                    diff = -diff  # positive always means the first option is better
                rows.append(
                    {
                        "comparison_type": "model_effect_vs_logistic",
                        "moment": moment,
                        "metric": metric,
                        "first": model,
                        "second": "logistic",
                        **corrected_repeated_cv_interval(
                            diff, config.outer_splits
                        ),
                    }
                )

        # Checkpoint effect: semester-one minus registration within model/fold.
        for model in primary.model.unique():
            wide = primary.loc[primary.model.eq(model)].pivot(
                index="split_id", columns="moment", values=metric
            )
            diff = wide["after_first_semester"] - wide["registration"]
            if metric == "multiclass_log_loss":
                diff = -diff
            rows.append(
                {
                    "comparison_type": "checkpoint_effect",
                    "moment": "after_first_semester_vs_registration",
                    "metric": metric,
                    "first": "after_first_semester",
                    "second": "registration",
                    "model": model,
                    **corrected_repeated_cv_interval(diff, config.outer_splits),
                }
            )
    return pd.DataFrame(rows)


def break_even_tables(queue: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Translate queue recall into an explicit, assumption-driven timing choice.

    The ratio is *not* an estimated treatment effect.  It is the relative
    effectiveness early support would need in order to offset lower predictive
    recall at registration.
    """

    base = queue.loc[queue.analysis.eq("repeated_stratified_cv")]
    mean_curve = (
        base.groupby(["model", "moment", "capacity"], as_index=False)
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

    # Decision-grid values are normalized per eventual dropout.  Queue cost is
    # expressed per selected student relative to the value of successfully
    # supporting one late-identified dropout.  These are transparent scenarios,
    # not claims about observed intervention effectiveness or cost.
    decision_rows = []
    for row in wide.itertuples(index=False):
        for relative_early_effect in np.arange(0.5, 2.01, 0.05):
            for outreach_cost in [0.00, 0.05, 0.10, 0.20]:
                early_value = (
                    row.registration_queue_recall * relative_early_effect
                    - row.capacity * outreach_cost
                )
                late_value = row.late_queue_recall - row.capacity * outreach_cost
                decision_rows.append(
                    {
                        "model": row.model,
                        "capacity": row.capacity,
                        "relative_early_effectiveness": relative_early_effect,
                        "outreach_cost_assumption": outreach_cost,
                        "registration_net_value_index": early_value,
                        "late_net_value_index": late_value,
                        "preferred_checkpoint": (
                            "registration" if early_value > late_value else
                            "after_first_semester"
                        ),
                    }
                )
    return wide, pd.DataFrame(decision_rows)


def bootstrap_group_metric(
    values: pd.DataFrame,
    metric_fn,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    """Row-bootstrap one descriptive subgroup metric and return estimate/CI."""

    estimate = float(metric_fn(values))
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_bootstrap):
        sample = values.iloc[rng.integers(0, len(values), len(values))]
        estimates.append(float(metric_fn(sample)))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return estimate, float(low), float(high)


def subgroup_table(
    predictions: pd.DataFrame,
    X: pd.DataFrame,
    config: RunConfig,
) -> pd.DataFrame:
    """Create interpretable subgroup diagnostics from averaged OOF predictions.

    Each student has one held-out prediction per repeat.  Averaging those risks
    prevents a student from being counted 5 times in the subgroup bootstrap.
    The intervals quantify sampling variation in this dataset; they do not
    establish fairness or transportability.
    """

    base = predictions.loc[
        predictions.analysis.eq("repeated_stratified_cv")
        & predictions.moment.eq("after_first_semester")
    ]
    averaged = (
        base.groupby(["model", "row_id", "target"], as_index=False)
        .agg(p_dropout=("p_dropout", "mean"))
    )
    attributes = pd.DataFrame(
        {
            "row_id": np.arange(len(X)),
            "Gender": X["Gender"].map({0: "Female", 1: "Male"}),
            "Scholarship": X["Scholarship holder"].map({0: "No", 1: "Yes"}),
            "Debtor": X["Debtor"].map({0: "No", 1: "Yes"}),
            "Age band": pd.cut(
                X["Age at enrollment"],
                [-np.inf, 19, 24, np.inf],
                labels=["19 or younger", "20-24", "25 or older"],
            ).astype(str),
        }
    )
    averaged = averaged.merge(attributes, on="row_id", validate="many_to_one")
    averaged["dropout"] = (averaged.target == 0).astype(int)

    # Create one global 20% queue per model, then inspect its composition and
    # error rates within groups.  Group-specific thresholds would answer a
    # different policy question.
    pieces = []
    for model, model_rows in averaged.groupby("model"):
        model_rows = model_rows.copy()
        model_rows["selected"] = top_fraction_mask(
            model_rows.p_dropout.to_numpy(), 0.20
        )
        for attribute in ["Gender", "Scholarship", "Debtor", "Age band"]:
            for level, group in model_rows.groupby(attribute, dropna=False):
                if len(group) < 50 or group.dropout.sum() < 10:
                    continue
                functions = {
                    "dropout_prevalence": lambda d: d.dropout.mean(),
                    "mean_predicted_dropout": lambda d: d.p_dropout.mean(),
                    "calibration_gap": lambda d: d.p_dropout.mean() - d.dropout.mean(),
                    "selection_rate": lambda d: d.selected.mean(),
                    "queue_recall": lambda d: (
                        d.loc[d.dropout.eq(1), "selected"].mean()
                    ),
                    "false_negative_rate": lambda d: (
                        1.0 - d.loc[d.dropout.eq(1), "selected"].mean()
                    ),
                    "dropout_brier": lambda d: np.mean(
                        (d.p_dropout - d.dropout) ** 2
                    ),
                }
                for metric_i, (metric, function) in enumerate(functions.items()):
                    estimate, low, high = bootstrap_group_metric(
                        group,
                        function,
                        config.subgroup_bootstraps,
                        SEED + metric_i,
                    )
                    pieces.append(
                        {
                            "model": model,
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
    return pd.DataFrame(pieces)


def make_figures(
    metrics: pd.DataFrame,
    break_even: pd.DataFrame,
    out: Path,
) -> None:
    """Create exact, traceable charts from exported numeric tables."""

    base = metrics.loc[metrics.analysis.eq("repeated_stratified_cv")]
    summary = (
        base.groupby(["model", "moment"], as_index=False)
        .macro_f1.agg(["mean", "std"])
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    for model, rows in summary.groupby("model"):
        rows = rows.set_index("moment").loc[list(MOMENTS)].reset_index()
        ax.errorbar(
            [0, 1], rows["mean"], yerr=rows["std"], marker="o", capsize=4,
            label=model,
        )
    ax.set_xticks([0, 1], ["Registration", "After first semester"])
    ax.set_ylabel("Macro F1 across outer folds")
    ax.set_title("Information timing changes performance more than model choice")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "repeated_cv_macro_f1.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for model, rows in break_even.groupby("model"):
        ax.plot(
            100 * rows.capacity,
            rows.early_effectiveness_break_even_ratio,
            marker="o",
            label=model,
        )
    ax.axhline(1.0, linestyle="--", color="#555555", linewidth=1)
    ax.set(
        xlabel="Students selected for review (%)",
        ylabel="Early/late effectiveness ratio needed for break-even",
        title="How much more effective must earlier support be?",
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "timing_break_even_curve.png", dpi=180)
    plt.close(fig)


def run() -> None:
    args = parse_args()
    config = config_for(args.profile)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    X, y, target_text = load_snapshot(args.input)
    groups = proxy_cohort_ids(X)

    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    queue_rows: list[dict] = []

    repeated = RepeatedStratifiedKFold(
        n_splits=config.outer_splits,
        n_repeats=config.outer_repeats,
        random_state=SEED,
    )
    for split_number, (train_idx, test_idx) in enumerate(repeated.split(X, y)):
        repeat, fold = divmod(split_number, config.outer_splits)
        result = evaluate_split(
            analysis="repeated_stratified_cv",
            split_id=f"r{repeat:02d}_f{fold:02d}",
            repeat=repeat,
            fold=fold,
            train_idx=train_idx,
            test_idx=test_idx,
            X=X,
            y=y,
            moments=MOMENTS,
            models=args.models,
            config=config,
        )
        for destination, values in zip(
            [metric_rows, prediction_rows, queue_rows], result
        ):
            destination.extend(values)

    if not args.skip_grouped:
        # Macro fields are excluded from the feature set so the model cannot
        # simply recover the group key used to construct the split.
        grouped_moments = {
            name: [c for c in columns if c not in MACRO_FIELDS]
            for name, columns in MOMENTS.items()
        }
        group_cv = GroupKFold(n_splits=min(5, groups.nunique()))
        for fold, (train_idx, test_idx) in enumerate(group_cv.split(X, y, groups)):
            result = evaluate_split(
                analysis="proxy_cohort_group_cv",
                split_id=f"group_f{fold:02d}",
                repeat=0,
                fold=fold,
                train_idx=train_idx,
                test_idx=test_idx,
                X=X,
                y=y,
                moments=grouped_moments,
                models=args.models,
                config=config,
            )
            for destination, values in zip(
                [metric_rows, prediction_rows, queue_rows], result
            ):
                destination.extend(values)

    if not args.skip_ablation:
        # This is a direct stress test of the claimed registration moment.
        # It does not determine when the fields were actually frozen.
        ablations = {
            "registration_full": ADMISSION,
            "registration_without_payment_scholarship": [
                c for c in ADMISSION if c not in TIMING_SENSITIVE_FIELDS
            ],
            "registration_without_macro_or_payment": [
                c for c in ADMISSION
                if c not in TIMING_SENSITIVE_FIELDS + MACRO_FIELDS
            ],
        }
        # Reuse the exact same repeated folds, but evaluate registration only.
        for split_number, (train_idx, test_idx) in enumerate(repeated.split(X, y)):
            repeat, fold = divmod(split_number, config.outer_splits)
            for label, features in ablations.items():
                result = evaluate_split(
                    analysis="registration_contract_ablation",
                    split_id=f"{label}_r{repeat:02d}_f{fold:02d}",
                    repeat=repeat,
                    fold=fold,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    X=X,
                    y=y,
                    moments={label: features},
                    models=args.models,
                    config=config,
                )
                for destination, values in zip(
                    [metric_rows, prediction_rows, queue_rows], result
                ):
                    destination.extend(values)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    queues = pd.DataFrame(queue_rows)
    pairs = paired_summaries(metrics, config)
    break_even, decisions = break_even_tables(queues)
    subgroups = subgroup_table(predictions, X, config)

    metrics.to_csv(out / "fold_metrics.csv", index=False)
    predictions.to_csv(out / "out_of_fold_predictions.csv", index=False)
    queues.to_csv(out / "queue_capacity_by_fold.csv", index=False)
    pairs.to_csv(out / "paired_effects_corrected_intervals.csv", index=False)
    break_even.to_csv(out / "timing_break_even.csv", index=False)
    decisions.to_csv(out / "timing_net_value_scenarios.csv", index=False)
    subgroups.to_csv(out / "subgroup_diagnostics_with_intervals.csv", index=False)
    pd.DataFrame(
        {
            "row_id": np.arange(len(X)),
            "target": target_text,
            "proxy_cohort": groups,
        }
    ).to_csv(out / "proxy_cohort_manifest.csv", index=False)

    make_figures(metrics, break_even, out)
    (out / "run_configuration.json").write_text(
        json.dumps(
            {
                "profile": args.profile,
                "models": args.models,
                "config": config.__dict__,
                "python": sys.version,
                "warning": (
                    "Proxy cohorts are inferred from macroeconomic tuples, not "
                    "verified academic-year identifiers. Net-value outputs are "
                    "scenario assumptions, not estimated treatment effects."
                ),
            },
            indent=2,
            default=list,
        ),
        encoding="utf-8",
    )
    print(metrics.groupby(["analysis", "moment", "model"])["macro_f1"].mean())
    print(f"\nOutputs written to {out.resolve()}")


if __name__ == "__main__":
    run()
