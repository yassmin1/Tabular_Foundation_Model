# Reproducibility provenance

- Python: 3.12.14
- Platform: Linux 6.18.57 x86_64
- NumPy: 2.3.5
- pandas: 2.2.3
- scikit-learn: 1.6.1
- PyTorch: 2.15.0 CPU
- TabPFN: 2.2.1
- CatBoost: 1.2.8
- Dataset: `data/uci697_student_dropout_snapshot.csv`
- Dataset SHA-256: `4b15c8e0b1fc9ea1e7994caaff47db72c07e0049bf9e47758aa53c2fc7fd6376`

The TabPFN package can download pretrained weights on first use. The result
files preserve the scores produced in the recorded environment; reruns with a
different TabPFN release or weight artifact should be treated as a new run.
