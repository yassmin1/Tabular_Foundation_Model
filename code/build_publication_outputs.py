"""Turn raw evaluation outputs into article-ready tables and figures.

This script deliberately reads the exported CSV files rather than importing
in-memory model objects.  That makes every sentence and chart in the article
traceable to a stable artifact that another analyst can inspect directly.

The post-processing rules are intentionally simple:

* average repeated outer-fold scores only within the same model and checkpoint;
* use corrected paired intervals already calculated by the evaluation script;
* retain each student's held-out probability from the five-fold binary audit
  before calibration and subgroup allocation summaries;
* label the break-even calculation as a scenario threshold, never as an
  estimated intervention effect.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve


MODEL_LABELS = {
    "logistic": "Logistic regression",
    "catboost": "CatBoost",
    "histgb": "HistGradientBoosting",
    "tabpfn_1": "TabPFN (1 estimator)",
    "logistic_binary": "Binary logistic regression",
    "tabpfn_1_binary": "Binary TabPFN (1 estimator)",
}

COLORS = {
    "logistic": "#2A9D8F",
    "catboost": "#457B9D",
    "histgb": "#8E6C8A",
    "tabpfn_1": "#E76F51",
    "logistic_binary": "#2A9D8F",
    "tabpfn_1_binary": "#E76F51",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results")
    )
    return parser.parse_args()


def save_primary_summary(results: Path) -> pd.DataFrame:
    folds = pd.read_csv(results / "primary_multiclass_fold_metrics.csv")
    summary = (
        folds.groupby(["moment", "model"], as_index=False)
        .agg(
            folds=("split_id", "nunique"),
            macro_f1=("macro_f1", "mean"),
            macro_f1_sd=("macro_f1", "std"),
            balanced_accuracy=("balanced_accuracy", "mean"),
            multiclass_log_loss=("multiclass_log_loss", "mean"),
            dropout_ap_from_multiclass=("dropout_ap_from_multiclass", "mean"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
        )
    )
    summary["model_label"] = summary.model.map(MODEL_LABELS)
    summary.to_csv(results / "publication_primary_summary.csv", index=False)
    return summary


def plot_paired_effects(results: Path) -> None:
    paired = pd.read_csv(results / "primary_paired_effects_corrected_intervals.csv")
    checkpoint = paired.loc[
        paired.comparison_type.eq("checkpoint_effect")
        & paired.metric.eq("macro_f1")
    ].copy()
    model = paired.loc[
        paired.comparison_type.eq("model_effect_vs_logistic")
        & paired.metric.eq("macro_f1")
        & paired["first"].eq("tabpfn_1")
    ].copy()

    checkpoint["label"] = checkpoint.model.map(MODEL_LABELS)
    model["label"] = model.moment.map(
        {
            "registration": "Registration",
            "after_first_semester": "After first semester",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, frame, title, xlabel in [
        (
            axes[0], checkpoint,
            "Checkpoint effect within each model",
            "Macro-F1 change: first semester minus registration",
        ),
        (
            axes[1], model,
            "TabPFN effect relative to logistic regression",
            "Macro-F1 change: TabPFN minus logistic",
        ),
    ]:
        frame = frame.reset_index(drop=True)
        y = np.arange(len(frame))
        mean = frame.mean_difference.to_numpy()
        lower = mean - frame.ci_low.to_numpy()
        upper = frame.ci_high.to_numpy() - mean
        ax.errorbar(
            mean, y, xerr=np.vstack([lower, upper]), fmt="o", capsize=5,
            color="#17365D", ecolor="#6B7C93", markersize=7,
        )
        ax.axvline(0, color="#777777", linestyle="--", linewidth=1)
        ax.set_yticks(y, frame.label)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle(
        "Repeated paired evaluation separates the information effect from the model effect",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(results / "paired_checkpoint_and_model_effects.png", dpi=200)
    plt.close(fig)


def save_queue_summary_and_plot_calibration(results: Path) -> pd.DataFrame:
    queues = pd.read_csv(results / "binary_queue_capacity_by_fold.csv")
    queue20 = queues.loc[np.isclose(queues.capacity, 0.20)]
    summary = (
        queue20.groupby(["moment", "model"], as_index=False)
        .agg(
            folds=("split_id", "nunique"),
            mean_selected_records=("selected_n", "mean"),
            mean_dropout_labels_in_test_fold=("dropout_n", "mean"),
            mean_dropout_labels_reached=("true_dropout_labels_reached", "mean"),
            queue_recall=("queue_recall", "mean"),
            selected_share_with_dropout_label=("queue_dropout_label_share", "mean"),
            dropout_ap=("dropout_ap", "mean"),
            dropout_roc_auc=("dropout_roc_auc", "mean"),
            dropout_brier=("dropout_brier", "mean"),
        )
    )
    summary["model_label"] = summary.model.map(MODEL_LABELS)
    summary.to_csv(results / "publication_binary_queue_summary_20pct.csv", index=False)

    predictions = pd.read_csv(results / "binary_out_of_fold_predictions.csv")
    averaged = (
        predictions.groupby(["model", "moment", "row_id", "dropout"], as_index=False)
        .p_dropout.mean()
    )
    calibration_rows = []
    for (model_name, moment), group in averaged.groupby(["model", "moment"]):
        observed, predicted = calibration_curve(
            group.dropout, group.p_dropout, n_bins=10, strategy="quantile"
        )
        for bin_number, (pred, obs) in enumerate(zip(predicted, observed), 1):
            calibration_rows.append(
                {
                    "model": model_name,
                    "moment": moment,
                    "bin": bin_number,
                    "mean_predicted_dropout": pred,
                    "observed_dropout_rate": obs,
                }
            )
    calibration = pd.DataFrame(calibration_rows)
    calibration.to_csv(results / "publication_binary_calibration_bins.csv", index=False)

    # Pair the reliability view with the break-even curve because both concern
    # operational use of probabilities, not just classification labels.
    break_even = pd.read_csv(results / "binary_timing_break_even.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].plot([0, 1], [0, 1], "--", color="#777777", label="Ideal")
    for (model_name, moment), group in calibration.groupby(["model", "moment"]):
        linestyle = "-" if moment == "after_first_semester" else ":"
        label = f"{MODEL_LABELS[model_name]} — " + (
            "semester 1" if moment == "after_first_semester" else "registration"
        )
        axes[0].plot(
            group.mean_predicted_dropout,
            group.observed_dropout_rate,
            marker="o", markersize=4, linestyle=linestyle,
            color=COLORS[model_name], label=label,
        )
    axes[0].set(
        title="Dropout probability reliability",
        xlabel="Mean predicted probability",
        ylabel="Observed Dropout-label rate",
        xlim=(0, 1), ylim=(0, 1),
    )
    axes[0].grid(alpha=0.25)

    for model_name, group in break_even.groupby("model"):
        axes[1].plot(
            100 * group.capacity,
            group.early_effectiveness_break_even_ratio,
            marker="o", color=COLORS[model_name],
            label=MODEL_LABELS[model_name],
        )
    axes[1].axhline(1.0, color="#777777", linestyle="--", linewidth=1)
    axes[1].axvline(20, color="#777777", linestyle=":", linewidth=1)
    axes[1].set(
        title="Effectiveness early support needs to break even",
        xlabel="Records selected for review (%)",
        ylabel="Required early-to-late effectiveness ratio",
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Probability quality and the assumed value of acting earlier")
    fig.tight_layout()
    fig.savefig(results / "binary_calibration_and_timing_decision.png", dpi=200)
    plt.close(fig)
    return summary


def save_robustness_summary_and_chart(
    results: Path, primary: pd.DataFrame
) -> pd.DataFrame:
    raw = pd.read_csv(results / "robustness_fold_metrics.csv")
    robustness = (
        raw.groupby(["analysis", "moment", "model"], as_index=False)
        .agg(macro_f1=("macro_f1", "mean"), folds=("split_id", "nunique"))
    )
    main = primary[["moment", "model", "macro_f1", "folds"]].copy()
    main.insert(0, "analysis", "primary_repeated_stratified_cv")
    combined = pd.concat([main, robustness], ignore_index=True)
    combined["model_label"] = combined.model.map(MODEL_LABELS)
    combined.to_csv(results / "publication_robustness_summary.csv", index=False)

    # The chart focuses on logistic and TabPFN because those are the two models
    # carried through every stress test.
    chosen = combined.loc[combined.model.isin(["logistic", "tabpfn_1"])].copy()
    chosen["scenario"] = chosen.analysis + " | " + chosen.moment
    order = list(dict.fromkeys(chosen.scenario))
    x = np.arange(len(order))
    width = 0.36
    fig, ax = plt.subplots(figsize=(13, 6))
    for offset, model_name in [(-width / 2, "logistic"), (width / 2, "tabpfn_1")]:
        values = (
            chosen.loc[chosen.model.eq(model_name)]
            .set_index("scenario")
            .reindex(order)
            .macro_f1
        )
        ax.bar(
            x + offset, values, width=width,
            label=MODEL_LABELS[model_name], color=COLORS[model_name],
        )
    short_labels = [
        label.replace("primary_repeated_stratified_cv | ", "Main: ")
        .replace("proxy_cohort_group_cv | ", "Proxy groups: ")
        .replace("registration_contract_ablation | ", "Ablation: ")
        .replace("after_first_semester", "semester 1")
        .replace("registration_without_payment_scholarship", "no payment/scholarship")
        .replace("registration_without_macro_or_payment", "no macro/payment")
        .replace("registration_full", "full registration")
        for label in order
    ]
    ax.set_xticks(x, short_labels, rotation=25, ha="right")
    ax.set_ylabel("Mean macro F1")
    ax.set_title("Robustness checks change the level, not the timing pattern")
    ax.set_ylim(0.40, 0.72)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(results / "robustness_comparison.png", dpi=200)
    plt.close(fig)
    return combined


def save_subgroup_allocation_reference(results: Path) -> pd.DataFrame:
    diagnostics = pd.read_csv(
        results / "binary_subgroup_diagnostics_with_intervals.csv"
    )
    wide = diagnostics.pivot_table(
        index=["model", "attribute", "level", "n", "dropout_n"],
        columns="metric", values="estimate"
    ).reset_index()
    global_prevalence = 1421 / 4424
    queue_capacity = 0.20
    # A base-rate-proportional allocation is only a descriptive reference.  It
    # is not asserted to be fair or optimal.  It asks how many selections a
    # group would receive if the fixed queue were distributed in proportion to
    # each group's observed Dropout-label prevalence.
    wide["base_rate_proportional_selection_rate"] = (
        queue_capacity * wide.dropout_prevalence / global_prevalence
    ).clip(upper=1.0)
    wide["selection_rate_minus_base_rate_reference"] = (
        wide.selection_rate - wide.base_rate_proportional_selection_rate
    )
    wide.to_csv(results / "publication_subgroup_allocation_reference.csv", index=False)
    return wide


def save_article_values(
    results: Path,
    primary: pd.DataFrame,
    queues: pd.DataFrame,
) -> None:
    paired = pd.read_csv(results / "primary_paired_effects_corrected_intervals.csv")
    break_even = pd.read_csv(results / "binary_timing_break_even.csv")
    values = {
        "primary_summary": primary.to_dict(orient="records"),
        "macro_f1_checkpoint_effects": paired.loc[
            paired.comparison_type.eq("checkpoint_effect")
            & paired.metric.eq("macro_f1")
        ].to_dict(orient="records"),
        "macro_f1_model_effects": paired.loc[
            paired.comparison_type.eq("model_effect_vs_logistic")
            & paired.metric.eq("macro_f1")
        ].to_dict(orient="records"),
        "queue_20pct": queues.to_dict(orient="records"),
        "break_even_20pct": break_even.loc[
            np.isclose(break_even.capacity, 0.20)
        ].to_dict(orient="records"),
        "interpretation": (
            "Break-even ratios are assumption thresholds, not estimated "
            "intervention effects. Corrected repeated-CV intervals remain "
            "approximate and do not establish external validity."
        ),
    }
    (results / "article_values.json").write_text(
        json.dumps(values, indent=2, default=float), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    results = args.results_dir
    primary = save_primary_summary(results)
    plot_paired_effects(results)
    queues = save_queue_summary_and_plot_calibration(results)
    save_robustness_summary_and_chart(results, primary)
    save_subgroup_allocation_reference(results)
    save_article_values(results, primary, queues)
    print(f"Publication tables and figures written to {results.resolve()}")


if __name__ == "__main__":
    main()
