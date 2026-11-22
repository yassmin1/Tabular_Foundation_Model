"""Progressive training-only tuning audit plus a bounded TabPFN ensemble check.

Conventional baselines are tuned on all five outer folds. Because CPU TabPFN-4
is much more expensive, the 1-vs-4 ensemble sensitivity is deliberately scoped
to the first deterministic outer fold at each checkpoint. Every completed row
is written immediately, so the artifact is recoverable and auditable.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

import higher_ed_tfm_publishable as publishable
import higher_ed_tfm_revised as core


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    X, y, _ = core.load_snapshot(args.input)
    cfg = core.RunConfig(5, 5, 3, 1000, (1, 4))
    splitter = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=5, random_state=core.SEED
    )
    folds = list(splitter.split(X, y))[:5]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Resume safely after an interrupted run. The file is rewritten after every
    # new result, so completed model/fold/checkpoint combinations are never
    # recomputed merely because the terminal session disconnected.
    if args.output.exists() and args.output.stat().st_size > 1:
        rows = pd.read_csv(args.output).to_dict("records")
    else:
        rows = []

    completed = {
        (str(row["audit_scope"]), int(row["fold"]), str(row["moment"]), str(row["model"]))
        for row in rows
    }

    # Training-only tuning for every conventional model on all five outer
    # folds. Test folds never participate in hyperparameter selection.
    for fold, (train_idx, test_idx) in enumerate(folds):
        for moment, features in core.MOMENTS.items():
            for model in ("logistic", "catboost", "histgb"):
                key = (
                    "five_outer_folds_training_only_tuning",
                    fold,
                    moment,
                    model,
                )
                if key in completed:
                    continue
                started = perf_counter()
                p, chosen, runtime = core.predict_model(
                    model, X.iloc[train_idx], y[train_idx],
                    X.iloc[test_idx], features, cfg.inner_splits,
                )
                rows.append({
                    "audit_scope": "five_outer_folds_training_only_tuning",
                    "fold": fold,
                    "moment": moment,
                    "model": model,
                    "chosen_parameters": str(chosen),
                    "runtime_seconds": runtime,
                    **publishable.multiclass_metrics(y[test_idx], p),
                })
                pd.DataFrame(rows).to_csv(args.output, index=False)
                completed.add(key)
                print(f"saved baseline fold={fold} moment={moment} model={model} elapsed={perf_counter()-started:.1f}s", flush=True)

    # The ensemble sensitivity is a declared subset audit: one fixed outer fold
    # at both checkpoints, comparing identical train/test rows.
    train_idx, test_idx = folds[0]
    for moment, features in core.MOMENTS.items():
        for model in ("tabpfn_1", "tabpfn_4"):
            key = (
                "first_outer_fold_ensemble_sensitivity",
                0,
                moment,
                model,
            )
            if key in completed:
                continue
            started = perf_counter()
            p, chosen, runtime = core.predict_model(
                model, X.iloc[train_idx], y[train_idx],
                X.iloc[test_idx], features, cfg.inner_splits,
            )
            rows.append({
                "audit_scope": "first_outer_fold_ensemble_sensitivity",
                "fold": 0,
                "moment": moment,
                "model": model,
                "chosen_parameters": str(chosen),
                "runtime_seconds": runtime,
                **publishable.multiclass_metrics(y[test_idx], p),
            })
            pd.DataFrame(rows).to_csv(args.output, index=False)
            completed.add(key)
            print(f"saved ensemble moment={moment} model={model} elapsed={perf_counter()-started:.1f}s", flush=True)

    print(pd.DataFrame(rows).groupby(["audit_scope", "moment", "model"]).macro_f1.mean(), flush=True)


if __name__ == "__main__":
    main()
