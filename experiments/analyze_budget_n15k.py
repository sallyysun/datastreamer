"""Section 6.2 long-stream check (n = 15,000; TAG n15k_<scenario>; probe_budget BUDGET_N=15000, RMPL k up to 16384, plus
mpl_exact = RMPL with the true margins). Same seeds, gains and validation-only selection as the n = 5000 study.
Usage (from experiments/): python3 analyze_budget_n15k.py [scenario ...] -> results/budget_n15k_2026-11-15.md"""
import glob
import math
import sys
import numpy as np
import pandas as pd

R = "results/"
FAMS = ["gaussian", "t", "clayton", "gumbel"]
E = ["mpl_dskll", "xiao", "hermite"]
NAME = {"mpl_dskll": "RMPL", "xiao": "Counting Matrix", "hermite": "Hermite Series", "mpl_exact": "RMPL, true margins"}
BUD = {16: 16384, 64: 65536, 256: 262144}
SCS = sys.argv[1:] or [s for s in ["constant", "drift", "jump"] if len(glob.glob(R + f"budget_seeds_n15k_{s}_*.csv")) == 4]
M = ["# Section 6.2 at n = 15,000 (2026-11-15)\n", f"Scenarios available: {', '.join(SCS)}. RMSE ×10⁻², test seeds 42–141, configurations chosen on validation seeds 1000–1039.\n"]
for s in SCS:
    sw = pd.concat([pd.read_csv(R + f"budget_sweep_n15k_{s}_{f}.csv") for f in FAMS])
    sd = pd.concat([pd.read_csv(R + f"budget_seeds_n15k_{s}_{f}.csv") for f in FAMS])
    val = sw[sw.split == "val"]
    SD = sd.set_index(["family", "estimator", "approx", "gain", "seed"]).rmse.sort_index()
    M.append(f"## {s}\n\n### Accuracy and retained state by approximation size (family-averaged; each family at its validation-best gain for that size)\n")
    M.append("| Estimator | size | max KiB | RMSE ± SE |"); M.append("|---|---|---|---|")
    for e in E + ["mpl_exact"]:
        for a in sorted(val[val.estimator == e].approx.unique()):
            per, kib = [], []
            for f in FAMS:
                v = val[(val.family == f) & (val.estimator == e) & (val.approx == a)]
                r = v.loc[v.rmse.idxmin()]
                per.append(100 * SD.loc[(f, e, a, r.gain)].values); kib.append(r.peak_bytes / 1024)
            x = np.mean(per, axis=0)
            M.append(f"| {NAME[e]} | {int(a)} | {max(kib):.1f} | {x.mean():.3f} ± {x.std(ddof=1) / 10:.3f} |")
    M.append("\n### Budget selections and the effect of more storage (paired, ×10⁻²)\n")
    M.append("| Estimator | sizes chosen at 16 / 64 / 256 KiB | max retained KiB at 16 / 64 / 256 | RMSE 16 / 64 / 256 (family avg) | 16→256 reduction median, max |")
    M.append("|---|---|---|---|---|")
    for e in E:
        ch, kb, mu, red = [], [], [], []
        xs = {}
        for b, cap in BUD.items():
            sizes, kib, per = [], [], []
            for f in FAMS:
                v = val[(val.family == f) & (val.estimator == e) & (val.peak_bytes <= cap)]
                r = v.loc[v.rmse.idxmin()]
                sizes.append(int(r.approx)); kib.append(r.peak_bytes / 1024)
                per.append(100 * SD.loc[(f, e, r.approx, r.gain)].values)
            xs[b] = per
            ch.append("/".join(map(str, sizes))); kb.append(f"{max(kib):.1f}"); mu.append(f"{np.mean(per):.3f}")
        d = [(xs[16][i] - xs[256][i]).mean() for i in range(4)]
        M.append(f"| {NAME[e]} | {' ; '.join(ch)} | {' / '.join(kb)} | {' / '.join(mu)} | {np.median(d):+.3f}, {max(d):+.3f} |")
    M.append("")
open(R + "budget_n15k_2026-11-15.md", "w").write("\n".join(M) + "\n")
print("\n".join(M))
