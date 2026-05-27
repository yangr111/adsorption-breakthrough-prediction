import argparse

import numpy as np
import pandas as pd


def clean_dataset(x_csv, y_csv, out_x_csv, out_y_csv, max_tmax=500.0, max_cmax=2.0):
    X = pd.read_csv(x_csv, header=None).values
    Y = pd.read_csv(y_csv, header=None).values

    print("Original dataset size:", len(X))

    time_axis = Y[:, :100]
    c1 = Y[:, 100:200]
    c2 = Y[:, 200:300]

    tmax = time_axis[:, -1]
    cmax = np.maximum(c1.max(axis=1), c2.max(axis=1))

    mask_t = tmax <= max_tmax
    mask_c = cmax <= max_cmax
    mask = mask_t & mask_c

    print("\n===== Cleaning criteria =====")
    print(f"Tmax <= {max_tmax}")
    print(f"Cmax <= {max_cmax}")

    print("\n===== Before cleaning =====")
    print("Total samples:", len(X))
    print(f"Tmax > {max_tmax}:", int(np.sum(~mask_t)))
    print(f"Cmax > {max_cmax}:", int(np.sum(~mask_c)))

    print("\n===== After cleaning =====")
    print("Remaining samples:", int(np.sum(mask)))
    print("Removed samples:", int(np.sum(~mask)))
    print("Removed ratio:", float(np.mean(~mask)))

    pd.DataFrame(X[mask]).to_csv(out_x_csv, header=False, index=False)
    pd.DataFrame(Y[mask]).to_csv(out_y_csv, header=False, index=False)

    print("\nSaved:")
    print(f"  {out_x_csv}")
    print(f"  {out_y_csv}")


def main():
    parser = argparse.ArgumentParser(description="Remove abnormal breakthrough simulations before model training.")
    parser.add_argument("--x_csv", default="curve_dataset.csv")
    parser.add_argument("--y_csv", default="curve.csv")
    parser.add_argument("--out_x_csv", default="curve_dataset_clean.csv")
    parser.add_argument("--out_y_csv", default="curve_clean.csv")
    parser.add_argument("--max_tmax", type=float, default=500.0)
    parser.add_argument("--max_cmax", type=float, default=2.0)
    args = parser.parse_args()
    clean_dataset(args.x_csv, args.y_csv, args.out_x_csv, args.out_y_csv, args.max_tmax, args.max_cmax)


if __name__ == "__main__":
    main()
