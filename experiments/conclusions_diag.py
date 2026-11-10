"""Chapter 6 headline conclusions for each diagnostic tag (seeds 1-30, 4 families): log bank vs hindsight-best
fixed memory, log vs squared bank, bank vs uniform, bank memory around the jump, regret violations.
Usage: python3 conclusions_diag.py TAG [TAG ...]   notes/studies/ew_marginals.md."""
import sys, math
import pandas as pd
FAM = ["gaussian", "clayton", "gumbel", "t"]
MEM = {"fixed_39": 39, "fixed_99": 99, "fixed_199": 199, "fixed_399": 399, "fixed_1999": 1999}

def load(kind, tag):
    fr = []
    for f in FAM:
        suf = "" if f == "gaussian" else "_" + f
        if "/" in tag:   # path prefix of an old results set: <dir>/thesis_bank, <dir>/bank_regret_42
            d = tag.rsplit("/", 1)[0]
            p = {"bank": f"{d}/thesis_bank{suf}.csv", "w": f"{d}/thesis_bank_weights{suf}.csv",
                 "reg": f"{d}/bank_regret_42{suf}.csv"}[kind]
        elif tag.startswith(("final", "matched")):
            p = {"bank": f"results/thesis_bank_{tag}{suf}.csv", "w": f"results/thesis_bank_weights_{tag}{suf}.csv",
                 "reg": f"results/bank_regret_42_{tag}{suf}.csv"}[kind]
        else:
            p = {"bank": f"results/thesis_bank_diag_{tag}{suf}.csv", "w": f"results/thesis_bank_weights_diag_{tag}{suf}.csv",
                 "reg": f"results/bank_regret_1_{tag}{suf}.csv"}[kind]
        try: fr.append(pd.read_csv(p))
        except FileNotFoundError: pass
    return pd.concat(fr) if fr else None

def paired(a, b):
    d = (a - b).dropna(); return d.mean(), d.std(ddof=1) / math.sqrt(len(d))

for tag in sys.argv[1:]:
    d = load("bank", tag); d = d[d.window == "whole"]
    print(f"\n######## {tag}")
    rows = []
    for (c, f, s), g in d[d.scenario.isin(["drift", "jump"])].groupby(["component", "copula", "scenario"]):
        p = g.pivot_table(index="seed", columns="method", values="rmse")
        best = min(MEM, key=lambda m: p[m].mean())
        m1, se1 = paired(p["bank_log"], p[best]); m2, se2 = paired(p["bank_log"], p["bank"]); m3, se3 = paired(p["bank"], p["uniform"])
        rows.append(dict(comp=c, fam=f, scen=s, best=MEM[best], log_vs_best=1000 * m1, se1=1000 * se1,
                         log_vs_sq=1000 * m2, se2=1000 * se2, sq_vs_uni=1000 * m3, se3=1000 * se3))
    r = pd.DataFrame(rows)
    def wlt(col, se):
        w = (r[col] < -2 * r[se]).sum(); l = (r[col] > 2 * r[se]).sum(); return f"{w}/{l}/{len(r) - w - l}"
    for s in ["drift", "jump"]:
        q = r[r.scen == s]
        w = (q.log_vs_best < -2 * q.se1).sum(); l = (q.log_vs_best > 2 * q.se1).sum()
        print(f"  log bank vs hindsight-best fixed, {s}: W/L/T {w}/{l}/{len(q)-w-l}; mean diff {q.log_vs_best.mean():+.1f} (range {q.log_vs_best.min():+.1f} .. {q.log_vs_best.max():+.1f})")
    print(f"  log vs squared bank (24 cells): W/L/T {wlt('log_vs_sq', 'se2')}; mean {r.log_vs_sq.mean():+.1f}")
    print(f"  squared bank vs uniform (24 cells): W/L/T {wlt('sq_vs_uni', 'se3')}; mean {r.sq_vs_uni.mean():+.1f}")
    w = load("w", tag)
    if w is not None:
        w = w.assign(nw=w.weight * w.mem).groupby(["component", "bank", "t"]).nw.sum().reset_index()
        w = w.groupby(["component", "bank", "t"]).nw.mean().reset_index()   # mean over families
        for (c, b), g in w.groupby(["component", "bank"]):
            g = g.set_index("t").nw
            vals = " ".join(f"t{t}:{g.loc[t] / 4:.0f}" if False else f"t{t}:{g.loc[t]:.0f}" for t in [2495, 2550, 2700, 3000, 3500])
            print(f"  weighted memory sum_k w_k N_eff,k ({c}, {b}; mean over families, jump): {vals}")
    reg = load("reg", tag)
    if reg is not None:
        g = reg.groupby("bank").apply(lambda x: pd.Series({"runs": len(x), "violations": int((x.regret > x.bound).sum()),
                                                            "mean_regret/bound": (x.regret / x.bound).mean()}))
        print("  regret:"); print(g.to_string())
