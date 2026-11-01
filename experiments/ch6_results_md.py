"""Tables for docs/chapter6_results.md: every Chapter 6 bank result, from probe_thesis_bank
(results/thesis_bank*.csv, tuned eta/delta) and probe_bank_tune (results/bank_tune_42*.csv, for the
untuned comparison). Paired differences are mean +- 2 SE over seeds 42-101.
Usage (from experiments/): python3 ch6_results_md.py > tables.md"""
import pandas as pd
sfx=["","_clayton","_gumbel","_t"]
d=pd.concat(pd.read_csv(f"results/thesis_bank{s}.csv") for s in sfx)
tune=pd.concat(pd.read_csv(f"results/bank_tune_42{s}.csv") for s in sfx)
OLD={"sq":("bank",0.5,1e-4),"log":("bank_log",1.0,4e-4)}
old=pd.concat(tune[(tune.method==m)&(tune.eta==e)&((tune.delta-dl).abs()<1e-12)].assign(method=n) for m,(n,e,dl) in OLD.items())
F=["fixed_39","fixed_99","fixed_199","fixed_399","fixed_1999"]
fam={"gaussian":"Gaussian","clayton":"Clayton","gumbel":"Gumbel","t":"t₄"}
comp={"xiao":"Xiao","hermite":"Hermite","mle":"MPL"}
def ci(x): x=x.dropna(); return x.mean(), 2*x.std(ddof=1)/len(x)**.5
def f(mu,h): return f"{mu:+.1f} ± {h:.1f}"
out=[]
def wide(metric,win="whole",src=d):
    return src[src.window==win].pivot_table(index=["copula","scenario","component","seed"],columns="method",values=metric)
for metric,sc,unit in [("rmse",1000,"RMSE ×10⁻³"),("excess_x1000",1,"excess log loss ×10⁻³ nats/tick")]:
  w=wide(metric)
  for scns,title in [(["drift","jump"],"drift and jump"),(["constant","oscillate"],"constant and oscillate (controls)")]:
    out.append(f"\n### Whole run, {unit}: {title}\n")
    out.append("Means over 60 seeds. Δ columns are paired differences, mean ± 2 SE; negative = bank better. VFF = better rate in hindsight.\n")
    out.append("| Family | Scenario | Est. | Best fixed (M) | M=199 | Squared | Log | Uniform | VFF (rate) | Log − best | Log − M199 | Log − VFF | Sq − best | Sq − VFF |")
    out.append("|"+"---|"*15)
    for c in ["gaussian","clayton","gumbel","t"]:
      for s in scns:
        for k in ["xiao","hermite","mle"]:
          g=w.loc[(c,s,k)]; b=g[F].mean().idxmin(); v=min(["vff_slow","vff_fast"],key=lambda x:g[x].mean())
          m=lambda x:f"{g[x].mean()*sc:.1f}"
          out.append(f"| {fam[c]} | {s} | {comp[k]} | {m(b)} ({b[6:]}) | {m('fixed_199')} | {m('bank')} | {m('bank_log')} | {m('uniform')} | {m(v)} ({v[4:]}) | "
            f"{f(*ci((g.bank_log-g[b])*sc))} | {f(*ci((g.bank_log-g.fixed_199)*sc))} | {f(*ci((g.bank_log-g[v])*sc))} | {f(*ci((g.bank-g[b])*sc))} | {f(*ci((g.bank-g[v])*sc))} |")
  # tallies
  out.append(f"\n### Win / loss / tie counts, {unit} (12 family × estimator cells per row)\n")
  out.append("| Bank | Scenario | M=39 | M=99 | M=199 | M=399 | M=1999 | Uniform | VFF | Hindsight best |")
  out.append("|"+"---|"*10)
  pooled=d[(d.window=="whole")&d.scenario.isin(["drift","jump"])].pivot_table(index=["copula","component","seed"],columns="method",values=metric,aggfunc="mean")
  for bk,lab in [("bank_log","Log"),("bank","Squared")]:
    for s in ["jump","drift","pooled","constant","oscillate"]:
      cells=[]
      for ref in F+["uniform","vff","best"]:
        n={"W":0,"L":0,"T":0}
        groups = pooled.groupby(level=[0,1]) if s=="pooled" else w.xs(s,level=1).groupby(level=[0,1])
        for _,g in groups:
          r = ref
          if ref=="vff": r=min(["vff_slow","vff_fast"],key=lambda x:g[x].mean())
          if ref=="best": r=g[F].mean().idxmin()
          mu,h=ci(g[bk]-g[r]); n["W" if mu+h<0 else "L" if mu-h>0 else "T"]+=1
        cells.append(f"{n['W']}/{n['L']}/{n['T']}")
      out.append(f"| {lab} | {s} | "+" | ".join(cells)+" |")
# log minus squared
out.append("\n### Log bank − squared bank (paired, mean ± 2 SE; negative = log bank better)\n")
out.append("| Family | Scenario | Est. | RMSE ×10⁻³ | Excess log loss ×10⁻³ |")
out.append("|---|---|---|---|---|")
wr,we=wide("rmse"),wide("excess_x1000")
for c in ["gaussian","clayton","gumbel","t"]:
  for s in ["drift","jump","constant","oscillate"]:
    for k in ["xiao","hermite","mle"]:
      a=wr.loc[(c,s,k)]; e=we.loc[(c,s,k)]
      out.append(f"| {fam[c]} | {s} | {comp[k]} | {f(*ci((a.bank_log-a.bank)*1000))} | {f(*ci(e.bank_log-e.bank))} |")
# recovery
out.append("\n### Recovery around the jump at t = 2500, Xiao, RMSE ×10⁻³\n")
for c in ["gaussian","clayton","gumbel","t"]:
  r=d[(d.copula==c)&(d.scenario=="jump")&(d.component=="xiao")].groupby(["method","window"]).rmse.mean().unstack()*1000
  out.append(f"\n**{fam[c]}**\n")
  out.append("| Method | before (2000) | 0–100 after | 100–500 | 500–1500 | all after | whole |")
  out.append("|---|---|---|---|---|---|---|")
  for mth in F+["bank","bank_log","uniform","vff_slow","vff_fast"]:
    x=r.loc[mth]; out.append(f"| {mth} | {x.pre:.1f} | {x.rec_0_100:.1f} | {x.rec_100_500:.1f} | {x.rec_500_1500:.1f} | {x.post:.1f} | {x.whole:.1f} |")
# tuned vs old
out.append("\n### Tuned − untuned (same seeds 42–101, RMSE ×10⁻³, paired)\n")
out.append("| Family | Scenario | Est. | Squared: tuned − old | Log: tuned − old |")
out.append("|---|---|---|---|---|")
wo=wide("rmse",src=old)
for c in ["gaussian","clayton","gumbel","t"]:
  for s in ["drift","jump"]:
    for k in ["xiao","hermite","mle"]:
      a=wr.loc[(c,s,k)]; o=wo.loc[(c,s,k)]
      out.append(f"| {fam[c]} | {s} | {comp[k]} | {f(*ci((a.bank-o.bank)*1000))} | {f(*ci((a.bank_log-o.bank_log)*1000))} |")
print("\n".join(out))
