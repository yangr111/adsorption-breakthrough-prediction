import argparse
import concurrent.futures
import glob
import os
from pathlib import Path
import subprocess
import threading

lock = threading.Lock()


def is_done(path: Path, min_component_files: int = 3) -> bool:
    column_file = path / "column.data"
    component_files = glob.glob(str(path / "component_*.data"))
    return (
        column_file.exists()
        and column_file.stat().st_size > 1000
        and len(component_files) >= min_component_files
    )


def run_task(path: Path, ruptura_exe: Path, timeout: int, summary_file: Path, min_component_files: int):
    name = path.name
    print(f"[START] {name}")
    try:
        result = subprocess.run(
            [str(ruptura_exe)],
            cwd=str(path),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
        if result.returncode == 0 and is_done(path, min_component_files):
            status = "SUCCESS"
        elif result.returncode == 0:
            status = "NO_OUTPUT"
        else:
            status = f"FAILED({result.returncode})"
    except subprocess.TimeoutExpired:
        status = "TIMEOUT"
    except Exception as exc:
        status = f"EXCEPTION: {exc}"

    with lock:
        with open(summary_file, "a", encoding="utf-8") as f:
            f.write(f"{name}\t{status}\n")
    print(f"[END] {name} -> {status}")


def main():
    parser = argparse.ArgumentParser(description="Run RUPTURA breakthrough simulations in batch.")
    parser.add_argument("cases_dir", type=Path, help="Directory containing iteration_* simulation folders.")
    parser.add_argument("--ruptura_exe", type=Path, required=True, help="Path to the compiled RUPTURA executable, e.g. /path/to/RUPTURA-main/src/ruptura.")
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=7200, help="Timeout per simulation in seconds.")
    parser.add_argument("--summary", type=Path, default=Path("run_summary.txt"))
    parser.add_argument("--min_component_files", type=int, default=3, help="Binary cases should produce helium plus two adsorbable components.")
    args = parser.parse_args()

    cases_dir = args.cases_dir.resolve()
    ruptura_exe = args.ruptura_exe.resolve()
    if not cases_dir.exists():
        raise FileNotFoundError(f"Cases directory not found: {cases_dir}")
    if not ruptura_exe.exists():
        raise FileNotFoundError(f"RUPTURA executable not found: {ruptura_exe}")

    subfolders = [p for p in cases_dir.iterdir() if p.is_dir()]
    tasks = [p for p in subfolders if not is_done(p, args.min_component_files)]
    summary_file = args.summary if args.summary.is_absolute() else cases_dir / args.summary

    print("Total tasks:", len(subfolders))
    print("Remaining tasks:", len(tasks))
    print("RUPTURA executable:", ruptura_exe)
    print("Summary file:", summary_file)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(run_task, task, ruptura_exe, args.timeout, summary_file, args.min_component_files) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    print("ALL DONE")


if __name__ == "__main__":
    main()
