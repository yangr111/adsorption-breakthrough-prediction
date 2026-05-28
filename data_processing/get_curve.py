import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def fix_negative_concentration(time, concentration):
    concentration = np.asarray(concentration, dtype=float).copy()
    negative_mask = concentration < 0
    if not np.any(negative_mask):
        return concentration

    valid_idx = np.where(~negative_mask)[0]
    if len(valid_idx) < 2:
        concentration[negative_mask] = 0.0
        return concentration

    interpolator = interp1d(
        time[valid_idx],
        concentration[valid_idx],
        kind="linear",
        fill_value="extrapolate",
    )
    concentration[negative_mask] = interpolator(time[negative_mask])
    return concentration


def read_component_data(case_dir, component_name):
    candidates = [
        case_dir / component_name,
        case_dir / component_name.replace("Compent", "Component"),
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path, sep=r"\s+", header=None)
    raise FileNotFoundError(f"Could not find component data file in {case_dir}: {component_name}")


def get_curve(case_dir, num_points=100):
    case_dir = Path(case_dir)
    component_1 = read_component_data(case_dir, "component_1_Compent1.data")
    component_2 = read_component_data(case_dir, "component_2_Compent2.data")

    with open(case_dir / "params.json", "r", encoding="utf-8") as pf:
        params = json.load(pf)

    time = component_1.iloc[:, 1].to_numpy(dtype=float)
    conc_1 = component_1.iloc[:, 2].to_numpy(dtype=float)
    conc_2 = component_2.iloc[:, 2].to_numpy(dtype=float)

    conc_1 = fix_negative_concentration(time, conc_1)
    conc_2 = fix_negative_concentration(time, conc_2)

    new_time = np.linspace(time.min(), time.max(), num_points)
    interp_c1 = interp1d(time, conc_1, kind="linear")
    interp_c2 = interp1d(time, conc_2, kind="linear")

    new_conc_1 = interp_c1(new_time)
    new_conc_2 = interp_c2(new_time)
    curve = np.concatenate([new_time, new_conc_1, new_conc_2])
    return curve, params


def load_filtered_folders(filtered_csv):
    filtered = pd.read_csv(filtered_csv, header=None)
    if filtered.shape[1] == 1:
        return filtered.iloc[:, 0].astype(str).tolist()
    return filtered.iloc[:, 1].astype(str).tolist()


def build_curve_dataset(root_dir, filtered_csv, out_x_csv, out_y_csv, num_points=100):
    root_dir = Path(root_dir)
    folder_names = load_filtered_folders(filtered_csv)
    dataset = []
    curves = []
    failed = 0

    for folder_name in folder_names:
        case_dir = root_dir / folder_name
        if not case_dir.is_dir():
            failed += 1
            continue
        try:
            curve, params = get_curve(case_dir, num_points=num_points)
            dataset.append(params)
            curves.append(curve)
        except Exception as exc:
            failed += 1
            print(f"Skipping {case_dir}: {exc}")

    if not dataset:
        raise RuntimeError("No valid simulation folders were converted.")

    x = np.asarray(dataset, dtype=float).reshape(-1, 24)
    y = np.asarray(curves, dtype=float).reshape(-1, num_points * 3)

    np.savetxt(out_x_csv, x, delimiter=",", fmt="%.10g")
    np.savetxt(out_y_csv, y, delimiter=",", fmt="%.10g")

    print(f"Converted simulations: {len(dataset)}")
    print(f"Failed/skipped folders: {failed}")
    print(f"Saved: {out_x_csv}")
    print(f"Saved: {out_y_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert RUPTURA component output files into model-ready curve CSV files."
    )
    parser.add_argument("root_dir", type=Path, help="Directory containing iteration_* simulation folders.")
    parser.add_argument("--filtered_csv", type=Path, default=Path("filtered_result_new.csv"))
    parser.add_argument("--out_x_csv", type=Path, default=Path("curve_dataset.csv"))
    parser.add_argument("--out_y_csv", type=Path, default=Path("curve.csv"))
    parser.add_argument("--num_points", type=int, default=100)
    args = parser.parse_args()

    build_curve_dataset(
        root_dir=args.root_dir,
        filtered_csv=args.filtered_csv,
        out_x_csv=args.out_x_csv,
        out_y_csv=args.out_y_csv,
        num_points=args.num_points,
    )


if __name__ == "__main__":
    main()
