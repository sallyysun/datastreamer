"""Paired analysis and figures for probe_memory_tradeoff.cpp.

Usage (from experiments/):  python3 analyze_memory_tradeoff.py [dir]
Reads <dir>/eval_windows.csv, traces.csv, tuning.csv; writes <dir>/summary.md and figures.
See notes/studies/memory_tradeoff.md.
"""
import csv
import math
import sys
from collections import defaultdict

DIR = sys.argv[1] if len(sys.argv) > 1 else "../notes/studies/memory_tradeoff"
T975 = 2.001  # t_{0.975, 59}

rows = defaultdict(dict)  # (scenario, window, role) -> {seed: (excess, rmse)}
labels = {}
for r in csv.DictReader(open(f"{DIR}/eval_windows.csv")):
    rows[(r["scenario"], r["window"], r["role"])][int(r["seed"])] = (float(r["excess_x1000"]), float(r["rmse"]))
    labels[r["role"]] = r["method"]


def paired(sc, w, a, b, idx=0):
    A, B = rows[(sc, w, a)], rows[(sc, w, b)]
    d = [A[s][idx] - B[s][idx] for s in A]
    n = len(d)
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    return m, T975 * sd / math.sqrt(n)


def mean(sc, w, a, idx=0):
    v = list(rows[(sc, w, a)].values())
    return sum(x[idx] for x in v) / len(v)


def fmt(p, scale=1.0):
    m, h = p
    star = " *" if abs(m) > h else ""
    return f"{scale * m:+.2f} ± {scale * h:.2f}{star}"


scen_windows = defaultdict(list)
for (sc, w, _) in rows:
    if w not in scen_windows[sc]:
        scen_windows[sc].append(w)
order = ["jump", "drift", "constant", "oscillate", "gauss2t_.6", "clay2gumb_.6", "clay2gumb_.8"]

out = []
out.append("# Memory trade-off: paired results\n")
out.append("Chosen configurations (tuning seeds 42-71, objective = mean of jump and drift excess log loss):\n")
for role in ["fixed", "fs", "disc", "vff", "vff_aggr", "rho_fs"]:
    out.append(f"- `{role}`: `{labels[role]}`")
out.append("\nExcess log loss = log c_true - log c_hat at the true PIT, mean per tick, x1000 nats. "
           "Lower is better, so a negative difference favours the first-named method. "
           "Paired 95% CI over 60 fresh seeds (5042-5101); `*` = CI excludes 0.\n")

pairs = [("disc", "fs"), ("disc", "vff"), ("disc", "fixed"), ("fs", "fixed"), ("vff", "fixed"), ("vff_aggr", "fixed")]
for sc in order:
    out.append(f"\n## {sc}\n")
    out.append("| window | " + " | ".join(f"mean {r}" for r in ["fixed", "fs", "disc", "vff"]) + " | "
               + " | ".join(f"{a} − {b}" for a, b in pairs) + " |")
    out.append("|" + "---|" * (5 + len(pairs)))
    for w in scen_windows[sc]:
        cells = [f"{mean(sc, w, r):.2f}" for r in ["fixed", "fs", "disc", "vff"]]
        cells += [fmt(paired(sc, w, a, b)) for a, b in pairs]
        out.append(f"| {w} | " + " | ".join(cells) + " |")

out.append("\n## Spearman RMSE (secondary; whole run, mean over seeds, x1000)\n")
out.append("| scenario | " + " | ".join(["fixed", "fs", "disc", "vff", "vff_aggr", "rho_fs"]) + " | disc − fs | disc − vff |")
out.append("|" + "---|" * 9)
for sc in order:
    cells = [f"{1000 * mean(sc, 'whole', r, 1):.1f}" for r in ["fixed", "fs", "disc", "vff", "vff_aggr", "rho_fs"]]
    cells += [fmt(paired(sc, "whole", "disc", "fs", 1), 1000), fmt(paired(sc, "whole", "disc", "vff", 1), 1000)]
    out.append(f"| {sc} | " + " | ".join(cells) + " |")

# Tuning surface for VFF: does a faster rate buy recovery, and at what stationary cost?
tun = list(csv.DictReader(open(f"{DIR}/tuning.csv")))
out.append("\n## Tuning surface (tuning seeds; mean excess log loss x1000, whole run)\n")
out.append("| method | jump | drift | constant | objective |")
out.append("|---|---|---|---|---|")
for r in tun:
    out.append(f"| {r['method']} | {1000 * float(r['jump']):.2f} | {1000 * float(r['drift']):.2f} | "
               f"{1000 * float(r['constant']):.2f} | {1000 * float(r['objective']):.2f} |")
open(f"{DIR}/summary.md", "w").write("\n".join(out) + "\n")
print("\n".join(out[:60]))

# ---- figures ----------------------------------------------------------------------------------
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

tr = defaultdict(lambda: defaultdict(list))
for r in csv.DictReader(open(f"{DIR}/traces.csv")):
    d = tr[(r["scenario"], r["role"])]
    d["t"].append(int(r["t"]))
    d["excess"].append(float(r["excess"]))
    d["mem"].append(float(r["mem"]))
    d["e2"].append(float(r["vff_e2"]))


def roll(x, w=50):
    out_, s = [], 0.0
    for i, v in enumerate(x):
        s += v
        if i >= w:
            s -= x[i - w]
        out_.append(s / min(i + 1, w))
    return out_


style = {"fixed": ("#888888", "fixed (tuned)"), "fs": ("#1b9e77", "fixed share (tuned)"),
         "disc": ("#7570b3", "discounted (tuned)"), "vff": ("#e7298a", "VFF (tuned)"),
         "vff_aggr": ("#e6ab02", "VFF aggressive r=0.5"), "rho_fs": ("#222222", "ρ-scored fixed share")}
scen = ["jump", "drift", "constant", "clay2gumb_.8"]
titles = {"jump": "jump: knob 0.2 → 0.8 at t = 2500", "drift": "drift: knob −0.5 → 0.8",
          "constant": "constant: knob 0.6", "clay2gumb_.8": "Clayton → Gumbel at t = 2500, ρ_S ≈ 0.79"}
fig, ax = plt.subplots(3, len(scen), figsize=(18, 10), sharex=True)
for j, s in enumerate(scen):
    for role, (col, lab) in style.items():
        d = tr[(s, role)]
        if not d["t"]:
            continue
        if role != "rho_fs":
            ax[0, j].plot(d["t"], [1000 * v for v in roll(d["excess"])], color=col, lw=1.2, label=lab)
        ax[1, j].plot(d["t"], d["mem"], color=col, lw=1.2, label=lab, ls="--" if role == "rho_fs" else "-")
        if role.startswith("vff"):
            ax[2, j].plot(d["t"], roll(d["e2"], 20), color=col, lw=1.2, label=lab)
    for a in ax[:, j]:
        if s != "drift" and s != "constant":
            a.axvline(2500, color="k", ls=":", lw=1)
    ax[2, j].axhline(1.0, color="k", lw=0.8, ls="--")
    ax[0, j].set_title(titles[s])
    ax[1, j].set_yscale("log")
    ax[2, j].set_xlabel("tick")
ax[0, 0].set_ylabel("excess log loss, ×10⁻³ nats/tick\n(50-tick rolling mean, 60 seeds)")
ax[1, 0].set_ylabel("memory in ticks (not an ESS)\nbanks: Σ π_k M_k; VFF: current M_t")
ax[2, 0].set_ylabel("VFF Rosenblatt e² (block mean)\nideal null = 1 (dashed);\nmeasured baseline ≠ 1, note §4")
ax[0, 0].legend(fontsize=8)
ax[1, 0].legend(fontsize=7)
fig.suptitle("Ticks 2000–4000 only (traces.csv); the post window runs to 5000. Linux / libstdc++.", fontsize=10)
fig.tight_layout()
fig.savefig(f"{DIR}/traces.png", dpi=100)
print("figure saved")
