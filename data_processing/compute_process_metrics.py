import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import simpson


LOW_LIMIT = 0.001


def compute_adsorption_ratio(case_dir):
    case_dir = Path(case_dir)
    file_1 = case_dir / "component_1_Compent1.data"
    file_2 = case_dir / "component_2_Compent2.data"
    params_file = case_dir / "params.json"

    result1 = pd.read_csv(file_1, sep=r"\s+", header=None)
    result2 = pd.read_csv(file_2, sep=r"\s+", header=None)

    with open(params_file, "r", encoding="utf-8") as pf:
        params = json.load(pf)

    valid_result1 = result1[result1.iloc[:, 2] > LOW_LIMIT]
    valid_result2 = result2[result2.iloc[:, 2] > LOW_LIMIT]
    if valid_result1.empty or valid_result2.empty:
        raise ValueError("No valid breakthrough points after threshold filtering.")

    t1 = valid_result1.iloc[0, 0]
    t2 = valid_result2.iloc[0, 0]
    if t1 > t2:
        t1, t2 = t2, t1
    if t1 == t2:
        raise ValueError("The two breakthrough times are equal.")

    filtered_result1 = result1[(result1.iloc[:, 1] > t1) & (result1.iloc[:, 1] < t2)]
    if filtered_result1.empty:
        raise ValueError("No valid data between the two breakthrough times.")

    x_values_1 = filtered_result1.iloc[:, 1]
    y_values_1 = filtered_result1.iloc[:, 2]
    integral_1 = simpson(y=y_values_1, x=x_values_1)
    component_1_adsorption_ratio = t2 - integral_1
    return component_1_adsorption_ratio, t2, params


def load_filtered_folders(filtered_csv):
    filtered = pd.read_csv(filtered_csv, header=None)
    if filtered.shape[1] == 1:
        return filtered.iloc[:, 0].astype(str).tolist()
    return filtered.iloc[:, 1].astype(str).tolist()


def compute_on_filtered_folders(root_dir, filtered_csv, out_x_csv, out_y_csv):
    root_dir = Path(root_dir)
    folder_names = load_filtered_folders(filtered_csv)
    dataset = []
    targets = []
    failed = 0

    for folder_name in folder_names:
        case_dir = root_dir / folder_name
        if not case_dir.is_dir():
            failed += 1
            continue
        try:
            ratio, time, params = compute_adsorption_ratio(case_dir)
            dataset.append(params)
            targets.append([ratio, time])
        except Exception as exc:
            failed += 1
            print(f"Skipping {case_dir}: {exc}")

    if not dataset:
        raise RuntimeError("No valid folders were processed.")

    x = np.asarray(dataset, dtype=float).reshape(-1, 24)
    target = pd.DataFrame(targets, columns=["adsorption_ratio", "time"])

    np.savetxt(out_x_csv, x, delimiter=",", fmt="%.10g")
    target.to_csv(out_y_csv, index=False)

    print(f"Processed folders: {len(dataset)}")
    print(f"Failed/skipped folders: {failed}")
    print(f"Saved: {out_x_csv}")
    print(f"Saved: {out_y_csv}")


def main():
    parser = argparse.ArgumentParser(description="Compute process-level metrics from RUPTURA outputs.")
    parser.add_argument("root_dir", type=Path, help="Directory containing iteration_* simulation folders.")
    parser.add_argument("--filtered_csv", type=Path, default=Path("filtered_result_new.csv"))
    parser.add_argument("--out_x_csv", type=Path, default=Path("time_dataset.csv"))
    parser.add_argument("--out_y_csv", type=Path, default=Path("time.csv"))
    args = parser.parse_args()

    compute_on_filtered_folders(args.root_dir, args.filtered_csv, args.out_x_csv, args.out_y_csv)


if __name__ == "__main__":
    main()
