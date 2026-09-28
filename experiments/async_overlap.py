# Overlap distribution under Bernoulli arrivals + refresh-time pairing: cbar vs E[C^2].
# python3 async_overlap.py   (see notes/studies/async_memory.md, "Random overlaps")
import random, math
def sim(px,py,T=400000,seed=1):
    r=random.Random(seed); lx=ly=-1; nx=ny=False; ptx=pty=None; cs=[]
    for t in range(T):
        if r.random()<px: lx=t; nx=True
        if r.random()<py: ly=t; ny=True
        if nx and ny:
            if ptx is not None:
                Lx=lx-ptx; Ly=ly-pty; I=max(0,min(lx,ly)-max(ptx,pty)); cs.append(I/math.sqrt(Lx*Ly))
            ptx,pty=lx,ly; nx=ny=False
    n=len(cs); m=sum(cs)/n; m2=sum(c*c for c in cs)/n
    return m, m2, sum(c==0 for c in cs)/n
print("px   py    cbar   E[C^2]  E[C^2]/cbar^2  frac c=0")
for px,py in [(0.5,0.5),(0.25,0.25),(0.1,0.1),(0.5,0.1)]:
    m,m2,z=sim(px,py); print(f"{px:<4} {py:<4}  {m:.3f}  {m2:.3f}   {m2/m/m:.3f}         {z:.3f}")
