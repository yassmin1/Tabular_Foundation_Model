# Revised Higher-Education Tabular Foundation Model Evaluation

This folder contains a staged, heavily commented evaluation that is designed
to answer the article's actual question: **does model choice matter as much as
the time at which new information becomes available?**

## Recommended command

Run from this package directory after creating the environment described below:

```bash
.venv/bin/python code/higher_ed_tfm_publishable.py \
  --profile full \
  --input data/uci697_student_dropout_snapshot.csv \
  --output-dir results_reproduced
```

The full run is intentionally staged:

1. **Primary repeated paired evaluation:** 5 folds × 5 repeats, identical folds
   for logistic regression, CatBoost, HistGradientBoosting, and TabPFN-1 at
   registration and after semester one. Hyperparameters are prespecified.
2. **Operational binary evaluation:** one five-fold pass with separately fitted
   Dropout-vs-rest logistic and TabPFN-1 models. These predictions drive the
   outreach queue and break-even analysis.
3. **Proxy-cohort stress test:** five GroupKFold splits defined by repeated
   macroeconomic tuples, with those macro fields excluded from the features.
4. **Registration-contract ablation:** five paired folds with and without
   `Debtor`, `Tuition fees up to date`, `Scholarship holder`, and the macro
   variables.
5. **Tuning/ensemble audit:** training-only tuning for the conventional
   baselines on five outer folds, plus a clearly labeled TabPFN
   1-vs-4-estimator sensitivity on the first deterministic outer fold.

Use `--skip-tuning-audit` to obtain the primary and robustness results first.
The audit can be run later because it is saved separately and does not change
the prespecified primary models.

## Smoke test

```bash
.venv/bin/python code/higher_ed_tfm_publishable.py \
  --profile smoke \
  --skip-tuning-audit \
  --input data/uci697_student_dropout_snapshot.csv \
  --output-dir smoke_results
```

The smoke profile checks execution paths; its two-fold outputs are **not**
publication results.

## Environment setup

Python 3.12 was used for the supplied results. The following commands create a
local virtual environment and install the pinned packages:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

TabPFN can download model weights on first use. The publication outputs already
included under `results/` and `figures/` allow readers to inspect the analysis
without rerunning the CPU-intensive experiment.

Dataset SHA-256:
`4b15c8e0b1fc9ea1e7994caaff47db72c07e0049bf9e47758aa53c2fc7fd6376`.
The supplied run used Python 3.12.14 on 64-bit Linux with CPU-only PyTorch.
Exact package versions are pinned in `requirements.txt`; run-level settings and
interpretation warnings are recorded in `results/run_manifest.json`.

## Package map

| Path | Contents |
|---|---|
| `article/` | Publication-ready Markdown article |
| `code/` | Fully commented primary, extended, audit, and chart scripts |
| `data/` | Frozen UCI dataset snapshot used in the run |
| `results/` | Fold-level metrics, predictions, audits, and manifests |
| `figures/` | Publication figures generated from result CSV files |
| `SOURCE_LEDGER.md` | Claim-to-source ledger |

## Interpretation boundaries

- Proxy cohorts are inferred from macroeconomic tuples. They are not verified
  academic-year identifiers.
- The break-even curve varies assumptions about early-versus-late intervention
  effectiveness. It does not estimate a treatment effect.
- The public table does not explain why each student left. Therefore, queue
  precision is named `queue_dropout_label_share`, not “share who needed help.”
- Corrected repeated-CV intervals address overlap among folds. They do not
  measure transportability to another university, country, or future cohort.
- Subgroup intervals are descriptive row-bootstrap intervals conditional on
  the fitted queue; they do not propagate training or threshold uncertainty
  and are not a fairness certificate.

The paired-effects file uses an oriented effect: positive values favor the
entry in its `first` column. For log loss, the raw arithmetic difference is
reversed because lower values are better.

## Main outputs

| File | Purpose |
|---|---|
| `primary_multiclass_fold_metrics.csv` | Every primary model/checkpoint/fold result |
| `primary_paired_effects_corrected_intervals.csv` | Paired model and checkpoint effects |
| `binary_out_of_fold_predictions.csv` | Traceable held-out binary risks |
| `binary_queue_capacity_by_fold.csv` | Queue recall and label share from 5%–50% capacity |
| `binary_timing_break_even.csv` | Early-effectiveness ratio required for break-even |
| `binary_net_value_scenarios.csv` | Transparent benefit/cost assumption grid |
| `binary_subgroup_diagnostics_with_intervals.csv` | Calibration, FNR, recall, and selection intervals |
| `robustness_fold_metrics.csv` | Proxy-cohort and registration-contract stress tests |
| `nested_tuning_and_tabpfn_ensemble_audit.csv` | Training-only tuning and 1-vs-4 TabPFN audit |
| `run_manifest.json` | Exact design and interpretation warnings |

## Runtime guidance

TabPFN CPU runtime varies substantially with hardware and concurrent workload.
On the environment used for the original case study, a one-estimator fit took
roughly 15–25 seconds. A 4-estimator audit is several times slower. The staged
design avoids repeating that audit over all 25 primary folds.

