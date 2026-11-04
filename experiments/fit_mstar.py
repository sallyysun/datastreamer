# Absolute check of M* = (2 sigma^2 / d^2)^(1/3) on probe_memory output. See notes/studies/memory_sweep.md §9.
# Usage (from repo root): python3 experiments/fit_mstar.py experiments/results/memory_sweep.csv
import numpy as np, pandas as pd, sys
df = pd.read_csv(sys.argv[1]); df = df[df.window == "all"]
N, burn = 5000, 500
t = np.arange(burn, N)
r = -0.5 + 1.3 * t / N
rs = 6/np.pi*np.arcsin(r/2)
d = (6/np.pi)*(0.5/np.sqrt(1-r**2/4))*1.3/N          # d rho_S / dt per tick
d2 = np.mean(d**2)
print(f"d: mean {d.mean():.3e}/tick, rms {np.sqrt(d2):.3e}")

def interp_min(m, y):
    i = int(np.argmin(y)); i = min(max(i,1), len(y)-2)
    x = np.log(m[i-1:i+2]); a, b, _ = np.polyfit(x, y[i-1:i+2], 2)
    return float(np.exp(-b/(2*a)))

for est in ["xiao_ew", "hermite_ew", "kalman"]:
    dr = df[(df.estimator==est)&(df.scenario=="drift")].sort_values("mem_ticks")
    co = df[(df.estimator==est)&(df.scenario=="constant")].sort_values("mem_ticks")
    m = dr.mem_ticks.values
    # sigma^2 = var*M, taken over the memories that are clearly warm (50..2000)
    w = (m >= 50) & (m <= 2000)
    s2_drift = np.median((dr["var"].values*m)[w])
    s2_const = np.median((co["var"].values*co.mem_ticks.values)[(co.mem_ticks.values>=50)&(co.mem_ticks.values<=2000)])
    # lag bias: drift bias^2 minus the memory-dependent floor seen on constant
    lag2 = dr.bias2.values - np.interp(m, co.mem_ticks.values, co.bias2.values)
    lag_ratio = np.sqrt(np.clip(lag2,0,None)) / (np.sqrt(d2)*m/2)
    m_obs = interp_min(m, dr.mse.values)
    p_d = (2*s2_drift/d2)**(1/3); p_c = (2*s2_const/d2)**(1/3)
    # MSE at predicted M* vs at observed M*
    mse_at = lambda M: np.interp(np.log(M), np.log(m), dr.mse.values)
    print(f"\n{est}: sigma2 (drift var*M) {s2_drift:.3f}, (constant var*M) {s2_const:.3f}")
    print("  lag |bias| / (d_rms*M/2) over M in [50,2000]:",
          " ".join(f"{x:.2f}" for x,mm in zip(lag_ratio, m) if 50<=mm<=2000))
    print(f"  M* observed {m_obs:.0f};  predicted {p_d:.0f} (drift sigma2, ratio {p_d/m_obs:.2f}),"
          f" {p_c:.0f} (constant sigma2, ratio {p_c/m_obs:.2f})")
    print(f"  MSE at observed {mse_at(m_obs):.5f}, at predicted(drift) {mse_at(p_d):.5f}"
          f" (+{100*(mse_at(p_d)/mse_at(m_obs)-1):.1f}%)")
