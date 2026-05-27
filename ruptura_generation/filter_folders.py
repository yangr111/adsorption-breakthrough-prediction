import argparse
import os
from pathlib import Path

import pandas as pd

MIN_CONCENTRATION = -0.001
FINAL_CONCENTRATION_RANGE = (0.99, 1.01)


def valid_component_file(file_path: Path) -> bool:
    try:
        result = pd.read_csv(file_path, sep=r"\s+", header=None)
        concentration = result.iloc[:, 2]
        if not (concentration >= MIN_CONCENTRATION).all():
            return False
        last_values = concentration.iloc[-3:]
        return ((last_values >= FINAL_CONCENTRATION_RANGE[0]) &
                (last_values <= FINAL_CONCENTRATION_RANGE[1])).all()
    except Exception as exc:
        print(f"Skipping {file_path}: {exc}")
        return False


def valid_simulation_folder(folder_path: Path) -> bool:
    component_files = sorted(folder_path.glob("component_*.data"))
    if not component_files:
        return False
    return all(valid_component_file(path) for path in component_files)


def filter_folders(root_dir: Path, output_csv: Path):
    valid = []
    for folder in sorted(root_dir.iterdir()):
        if folder.is_dir() and valid_simulation_folder(folder):
            valid.append(folder.name)
    pd.DataFrame(valid).to_csv(output_csv, index=True, header=False)
    print(f"Valid folders: {len(valid)}")
    print(f"Saved: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Filter valid RUPTURA breakthrough simulation folders.")
    parser.add_argument("root_dir", type=Path, help="Directory containing iteration_* folders.")
    parser.add_argument("--output", type=Path, default=Path("filtered_result_new.csv"))
    args = parser.parse_args()
    filter_folders(args.root_dir, args.output)


if __name__ == "__main__":
    main()
