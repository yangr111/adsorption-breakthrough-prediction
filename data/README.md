# Data

Place the full processed data files here or provide a Zenodo/Figshare link.

Expected model-ready files:

- `curve_dataset_clean.csv`: cleaned input descriptors, shape `(N, 24)`
- `curve_clean.csv`: cleaned breakthrough curves, shape `(N, 300)`; columns are 100 time points, 100 C1 concentrations, and 100 C2 concentrations.

The cleaning script removes samples with `Tmax > 500` or `Cmax > 2`.
