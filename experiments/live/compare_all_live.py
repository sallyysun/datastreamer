"""Compare live experiment CSVs with their in-process counterparts."""
from pathlib import Path
import sys

import glob
import os

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from report_output import write_report  # noqa: E402


def compare(f, ref):
    if open(f, "rb").read() == open(ref, "rb").read():
        return "identical"
    a, b = pd.read_csv(f), pd.read_csv(ref)
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return f"DIFFERENT SHAPE live {a.shape} vs in-process {b.shape}"
    num = [c for c in a.columns if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c])]
    txt = [c for c in a.columns if c not in num]
    bad_txt = int((a[txt].astype(str) != b[txt].astype(str)).any(axis=1).sum()) if txt else 0
    d = (a[num] - b[num]).abs()
    rel = d / b[num].abs().where(b[num].abs() > 1e-12)
    rows = int(((d > 0) & ~(a[num].isna() & b[num].isna())).any(axis=1).sum())
    return (f"{rows}/{len(a)} rows differ; max |diff| {np.nanmax(d.values):.3g}, "
            f"max relative {np.nanmax(rel.values) if np.isfinite(rel.values).any() else 0:.3g}"
            + (f"; {bad_txt} rows with different labels" if bad_txt else ""))


def main():
    out, tot, same = [], 0, 0
    for f in sorted(glob.glob("6.*/results/*.csv")):
        sec, name = f.split("/")[0], os.path.basename(f)
        ref = f"../{sec}/results/{name}"
        if not os.path.exists(ref):
            out.append(f"{sec}/{name}: no in-process file")
            continue
        tot += 1
        verdict = compare(f, ref)
        same += verdict == "identical"
        out.append(f"{sec}/{name}: {verdict}")
    out.insert(0, f"{same} of {tot} files byte-identical to the in-process (thesis) runs.")
    os.makedirs("results", exist_ok=True)
    write_report("results/live_vs_inprocess.txt", out)


if __name__ == "__main__":
    main()
