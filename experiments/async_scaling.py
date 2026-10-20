# M* scaling check for async_memory: python3 async_scaling.py results/async_memory_drift.csv
# Ratios of observed M*_clock to f^(-1/3) (sync) and f^(-1/3) c^(-2/3) (corr) predictions.
import csv,math,collections,sys
rows=list(csv.DictReader(open(sys.argv[1])))
def mstar(Ms,ys):
    k=min(range(len(ys)),key=lambda i:ys[i])
    if k in (0,len(ys)-1): return Ms[k]
    x=[math.log2(Ms[i]) for i in (k-1,k,k+1)]; y=[ys[i] for i in (k-1,k,k+1)]
    den=(x[0]-x[1])*(x[0]-x[2])*(x[1]-x[2])
    A=(x[2]*(y[1]-y[0])+x[1]*(y[0]-y[2])+x[0]*(y[2]-y[1]))/den
    B=(x[2]**2*(y[0]-y[1])+x[1]**2*(y[2]-y[0])+x[0]**2*(y[1]-y[2]))/den
    return 2**min(max(-B/(2*A),x[0]),x[2])
cell=collections.defaultdict(list); meta={}
for r in rows:
    liq=(float(r["px"]),float(r["py"]))
    cell[(liq,r["arm"],r["estimator"])].append((float(r["M_pairs"]),float(r["rmse_mean"])))
    meta[liq]=(float(r["steps_per_pair"]),float(r["mean_c"]))
ests=["xiao_ew","hermite_ew","gaussian_mle_ew","windowed_rank","kalman"]
base={e:mstar(*zip(*cell[((1.0,1.0),"sync",e)])) for e in ests}
print("liq        spp     c     | sync/pred(f^-1/3) | async/sync | corr/pred(f^-1/3 c^-2/3) | RMSE* async/sync | corr/sync")
for liq,(spp,c) in meta.items():
    if liq==(1.0,1.0): continue
    s=[];a=[];k=[];ra=[];rc=[]
    for e in ests:
        ms={arm:mstar(*zip(*cell[(liq,arm,e)]))*spp for arm in ("sync","async","corr")}
        rm={arm:min(y for _,y in cell[(liq,arm,e)]) for arm in ("sync","async","corr")}
        ps=base[e]*spp**(1/3)
        s.append(ms["sync"]/ps); a.append(ms["async"]/ms["sync"]); k.append(ms["corr"]/(ps*c**(-2/3)))
        ra.append(rm["async"]/rm["sync"]); rc.append(rm["corr"]/rm["sync"])
    f=lambda v:f"{min(v):.2f}-{max(v):.2f}"
    print(f"{liq[0]:.2f},{liq[1]:.2f}  {spp:5.2f}  {c:.3f}  | {f(s):>17} | {f(a):>10} | {f(k):>24} | {f(ra):>16} | {f(rc)}")
