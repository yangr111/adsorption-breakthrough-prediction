# Adsorption Breakthrough Curve Prediction

This repository contains scripts for generating RUPTURA breakthrough simulation inputs, cleaning simulated breakthrough data, constructing adsorption-derived descriptors, training dual deep-learning models, and reproducing evaluation figures.

## Important Note About RUPTURA

RUPTURA itself is not bundled in this repository. Install and compile it from the official repository first:

```bash
git clone https://github.com/iraspa/ruptura.git
```

After compilation, pass the path to the RUPTURA executable to `ruptura_generation/run_ruptura_batch.py` with `--ruptura_exe`.

## Expected Data Files

The model-training scripts expect cleaned data files named:

- `curve_dataset_clean.csv`: input descriptors, shape `(N, 24)`
- `curve_clean.csv`: breakthrough curves, shape `(N, 300)`

For `curve_clean.csv`, columns 0-99 are the time axis, columns 100-199 are C1 concentrations, and columns 200-299 are C2 concentrations.

Large generated data files and model checkpoints are intentionally excluded from Git by `.gitignore`. Put full data in `data/` locally or provide a Zenodo/Figshare link for publication.

## Workflow

### 1. Generate RUPTURA input folders

```bash
python ruptura_generation/breakthrough.py --base_dir break_binary --num_iterations 50000 --seed 42
```

This creates binary adsorption breakthrough simulation folders containing `params.json` and `simulation.input`.

### 2. Run RUPTURA simulations

```bash
python ruptura_generation/run_ruptura_batch.py break_binary --ruptura_exe /path/to/RUPTURA-main/src/ruptura --max_workers 8
```

On Windows, use the compiled executable path for your local RUPTURA installation.

### 3. Filter valid simulation folders

```bash
python ruptura_generation/filter_folders.py break_binary --output filtered_result_new.csv
```

### 4. Build model-ready curve files

Convert RUPTURA output files into:

- `curve_dataset.csv`
- `curve.csv`

`curve_dataset.csv` contains the 24 simulation/input descriptors. `curve.csv` contains 100 time points plus 100 C1 and 100 C2 concentration values.

### 5. Clean abnormal simulations

```bash
python data_processing/clean_dataset.py --x_csv curve_dataset.csv --y_csv curve.csv --out_x_csv curve_dataset_clean.csv --out_y_csv curve_clean.csv
```

The cleaning criteria are `Tmax <= 500` and `Cmax <= 2`.

### 6. Train the time model

Copy or symlink `curve_dataset_clean.csv` and `curve_clean.csv` into `time_model/`, then run for example:

```bash
cd time_model
python train_time.py --model lstm --features 50 --target total
```

Random search:

```bash
python random_search_time.py --targets total --trials 8 --epochs 500
```

### 7. Train the concentration model

Copy or symlink the same cleaned data into `concentration_model/`, then run for example:

```bash
cd concentration_model
python train_conc.py --model lstm --features 24
```

Random search:

```bash
python random_search_conc.py --trials 8 --epochs 500
```

### 8. Export and plot the final dual model

After model search, use:

```bash
python analysis/export_best_dual_model.py
python analysis/plot_final_dual_curves.py --final_dir final_dual_model --x_csv curve_dataset_clean.csv --y_csv curve_clean.csv
```

## Feature Sets

- 24 features: original column, gas, transport, and Langmuir-Freundlich parameters
- 38 features: original L-F parameters replaced by uptake values at 13 pressure points for each component
- 50 features: original 24 features plus 26 uptake features

The pressure points are `10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000 Pa`.

## RUPTURA Reference

Sharma, S.; Balestra, S. R. G.; Baur, R.; Agarwal, U.; Zuidema, E.; Rigutto, M. S.; Calero, S.; Vlugt, T. J. H.; Dubbeldam, D. RUPTURA: Simulation Code for Breakthrough, Ideal Adsorption Solution Theory Computations, and Fitting of Isotherm Models. *Molecular Simulation* **2023**, *49*, 893-953. https://doi.org/10.1080/08927022.2023.2202757

## AI-Assisted Coding Disclosure

Some data-processing and model-training scripts were drafted with assistance from an AI-based coding tool. All code, analyses, and interpretations were reviewed, revised, and validated by the authors.
