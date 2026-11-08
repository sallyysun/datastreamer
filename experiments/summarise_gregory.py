"""Gregory forgetting bank vs the Xiao, Hermite and MPL banks, paired on identical data
(results/thesis_bank_gregory[_fam].csv from COMPONENTS=gregory,xiao,hermite,mle TAG=gregory).
-> results/gregory_bank_summary.txt.  Notes: notes/studies/budget_mpl.md §23."""
import csv, math, collections, sys
fams = ["gaussian", "clayton", "gumbel", "t"]
rows = collections.defaultdict(dict)   # (fam, sc, window, method, comp) -> {seed: rmse}
exc = collections.defaultdict(dict)
for f in fams:
    p = "results/thesis_bank_gregory" + ("" if f == "gaussian" else "_" + f) + ".csv"
    for r in csv.DictReader(open(p)):
        k = (f, r["scenario"], r["window"], r["method"], r["component"])
        rows[k][r["seed"]] = float(r["rmse"]); exc[k][r["seed"]] = float(r["excess_x1000"])
comps = ["gregory", "xiao", "hermite", "mle"]
def ms(v):
    v = [x for x in v if not math.isnan(x)]; n = len(v); m = sum(v) / n
    return m, math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1) / n)
out = []
P = lambda s="": out.append(s)
for metric, D, scale in (("RMSE x1000", rows, 1000), ("excess log loss x1000", exc, 1)):
    P(f"== {metric}, window 'whole' (ticks 500-5000), mean over 60 seeds; paired diff = gregory - other ==")
    for meth in ("bank", "bank_log", "fixed_39", "fixed_199", "fixed_1999"):
        P(f"-- {meth}")
        P(f"{'family':9s}{'scenario':10s}" + "".join(f"{c:>14s}" for c in comps) + "".join(f"{'G-'+c:>16s}" for c in comps[1:]))
        for f in fams:
            for sc in ("constant", "drift", "jump", "oscillate"):
                d = {c: D[(f, sc, "whole", meth, c)] for c in comps}
                line = f"{f:9s}{sc:10s}" + "".join(f"{scale*ms(list(d[c].values()))[0]:14.1f}" for c in comps)
                for c in comps[1:]:
                    m, se = ms([d["gregory"][s] - d[c][s] for s in d["gregory"]])
                    line += f"{scale*m:+10.1f}±{scale*se:4.1f}"
                P(line)
    P()
P("== jump recovery windows, squared bank, RMSE x1000 ==")
for f in fams:
    for w in ("pre", "rec_0_100", "rec_100_500", "rec_500_1500"):
        P(f"{f:9s}{w:14s}" + "".join(f"{c:>9s} {1000*ms(list(rows[(f,'jump',w,'bank',c)].values()))[0]:6.1f}" for c in comps))
open("results/gregory_bank_summary.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
