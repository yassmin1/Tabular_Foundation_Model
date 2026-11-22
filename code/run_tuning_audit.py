"""Run the expensive tuning and TabPFN ensemble audit without rerunning 5x5 CV.

The main publishable driver uses fixed, prespecified settings across 25 paired
outer folds.  This companion script asks a narrower question on the first
five folds: would training-only hyperparameter tuning of the conventional
baselines, or increasing TabPFN from one to four estimators, materially change
the conclusion?  Keeping the audit separate prevents an accidental rerun of
the full repeated evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

import higher_ed_tfm_publishable as publishable
import higher_ed_tfm_revised as core


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/uci697_student_dropout_snapshot.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/nested_tuning_and_tabpfn_ensemble_audit.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    X, y, _ = core.load_snapshot(args.input)
    cfg = core.RunConfig(
        outer_splits=5,
        outer_repeats=5,
        inner_splits=3,
        subgroup_bootstraps=1000,
        tabpfn_estimators=(1, 4),
    )

    # Recreate the exact deterministic split sequence used by the main run.
    splitter = RepeatedStratifiedKFold(
        n_splits=cfg.outer_splits,
        n_repeats=cfg.outer_repeats,
        random_state=core.SEED,
    )
    first_repeat = list(splitter.split(X, y))[: cfg.outer_splits]
    result = publishable.tuning_audit(first_repeat, X, y, cfg)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(
        result.groupby(["moment", "model"])
        .agg(
            macro_f1=("macro_f1", "mean"),
            log_loss=("multiclass_log_loss", "mean"),
            runtime_seconds=("runtime_seconds", "sum"),
        )
        .round(4)
        .to_string()
    )
    print(f"\nAudit written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
