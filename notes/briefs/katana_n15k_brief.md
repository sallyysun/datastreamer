# Brief: run the Section 6.2 memory-budget sweep at n = 15,000 on Katana

From the cloud Chapter 6 session (2026-11-15). Read this whole brief first; ask the user before anything not listed here.

## Goal
Re-run the Section 6.2 (E1) memory-budget experiment with streams of **15,000 observations instead of 5,000**, so RMPL's KLL
margin sketches actually compress at large capacities (at n = 5000, k ≥ 8192 just stores every observation, which is why RMPL's
retained state stops near 64 KiB). Same estimators, seeds, gains and selection protocol as the existing E1 run; only the stream
length and the RMPL capacity grid (k up to 16384) change. A local 4-core run of the same thing is going in the cloud session but
will take hours; Katana should do it in well under an hour as 12 parallel jobs.

## Rules
- Repository `sallyysun/datastreamer`, branch **`bank-vff-study`** at commit `6c197ab`.
- Don't commit or push anything (the user hasn't approved commits for this work). Don't commit raw CSVs ever.
- Don't edit the thesis LaTeX. Don't change any estimator code; the only change is the harness patch below.
- Build on Linux with g++ and libstdc++ (Katana is fine: the seeds generate identical data to the cloud run only on libstdc++).

## 1. Get the code
```bash
git clone https://github.com/sallyysun/datastreamer.git && cd datastreamer   # or fetch into an existing clone
git checkout bank-vff-study && git reset --hard 6c197ab   # only in a fresh clone; don't discard the user's local work
git submodule update --init external/datasketches-cpp
```
Apply the harness patch (it only adds environment-variable options to `experiments/probe_budget.cpp`; defaults are unchanged):
save the block at the end of this brief as `probe_budget_n15k.patch` and run `git apply probe_budget_n15k.patch`.
**Shortcut:** the latest `bank-vff-study` (the commit that added this brief) already contains the patch; if you check that out
instead of `6c197ab`, skip `git apply`. The reproduction check in step 3 holds for either.

New options it adds: `BUDGET_N` (stream length), `BUDGET_EST` (estimator subset), `BUDGET_KLL` (RMPL capacity grid),
`BUDGET_TIME_GAIN` (timing only), `BUDGET_THREADS` (worker threads; **set this to the cores PBS allocates**, otherwise it uses
every core it can see on the node).

## 2. Build
Eigen and DataSketches are header-only. On Katana, load a recent GCC (C++20) and point `EIGEN` at an Eigen install
(`module avail eigen`; or download Eigen 3.4 headers into your home directory).
```bash
module load gcc            # name/version per `module avail gcc`; needs C++20
cd experiments
make -B probe_budget EIGEN=/path/to/eigen/prefix CXX=g++   # the prefix whose include/eigen3 holds Eigen
```

## 3. Check the build reproduces the cloud results (default n = 5000), ~1 minute
```bash
SCENARIOS=matched BUDGET_THREADS=4 BUDGET_EST=mpl_dskll BUDGET_KLL=1024 BUDGET_GAINS=0.01 BUDGET_SCENARIOS=jump TAG=katana_check \
  ./probe_budget sweep 100 40 gaussian
python3 -c "import pandas as pd; x=pd.read_csv('results/budget_seeds_katana_check_gaussian.csv'); print(round(x.rmse.mean(), 7))"
```
It must print **0.0790964** (first three seeds by seed order: 0.0770336, 0.0793138, 0.074311). If it doesn't, stop and tell the
user: the platform is generating different data and the results won't be comparable.

## 4. The run: 12 independent tasks (4 families × 3 scenarios)
Each task is one call:
```bash
SCENARIOS=matched BUDGET_N=15000 BUDGET_THREADS=$NCPUS \
BUDGET_EST=xiao,hermite,mpl_dskll,mpl_exact \
BUDGET_KLL=8,16,32,64,128,256,512,1024,2048,4096,8192,16384 \
BUDGET_GAINS=<G> BUDGET_SCENARIOS=<scenario> TAG=n15k_<scenario> \
./probe_budget sweep 100 40 <family>
```
- `<family>` ∈ gaussian, t, clayton, gumbel; `<scenario>` ∈ constant, drift, jump.
- `<G>` = `0,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1` for constant and drift; `0.005,0.01,0.02,0.05,0.1` for jump.
- Run from `experiments/` (it writes to `results/`). Outputs per task: `results/budget_sweep_n15k_<scenario>_<family>.csv`,
  `results/budget_seeds_n15k_<scenario>_<family>.csv` (and `budget_paths_…` for jump).

A PBS array script along these lines (adjust queue/walltime/memory to Katana's current policy; check `man qsub` or the UNSW
Katana docs):
```bash
#!/bin/bash
#PBS -N e1_n15k
#PBS -l select=1:ncpus=16:mem=8gb
#PBS -l walltime=03:00:00
#PBS -J 0-11
cd "$PBS_O_WORKDIR"            # submit from datastreamer/experiments
module load gcc
FAMS=(gaussian t clayton gumbel); SCS=(constant drift jump)
f=${FAMS[$((PBS_ARRAY_INDEX % 4))]}; s=${SCS[$((PBS_ARRAY_INDEX / 4))]}
G=0,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1; [ "$s" = jump ] && G=0.005,0.01,0.02,0.05,0.1
SCENARIOS=matched BUDGET_N=15000 BUDGET_THREADS=${NCPUS:-16} BUDGET_EST=xiao,hermite,mpl_dskll,mpl_exact \
BUDGET_KLL=8,16,32,64,128,256,512,1024,2048,4096,8192,16384 BUDGET_GAINS=$G BUDGET_SCENARIOS=$s TAG=n15k_$s \
./probe_budget sweep 100 40 $f
```
Rough cost: a few minutes to ~20 minutes per task on 16 cores; constant and drift (9 gains) are slower than jump (5 gains).
Memory is small (well under 1 GB per task).

## 5. Afterwards
- Check all 12 sweep files exist and each has rows for both `split=val` and `split=test`.
- Record in a short note (`notes/code_changes/`, not committed): Katana node type, GCC version, Eigen version, wall time per task,
  and the reproduction value from step 3.
- **Ask the user how to get the 24–36 CSVs back to the cloud session** (they're too large and raw for the thesis branch). Options:
  a tarball they upload, or a separate data branch if they approve pushing it. Don't push without their say-so.
- The cloud session will analyse them with `experiments/analyze_budget_time.py`-style scripts; no analysis is needed on Katana.

## Patch (`probe_budget_n15k.patch`, against 6c197ab)
```diff
diff --git a/experiments/probe_budget.cpp b/experiments/probe_budget.cpp
index 7048b7f..39f6e03 100644
--- a/experiments/probe_budget.cpp
+++ b/experiments/probe_budget.cpp
@@ -30,9 +30,10 @@
 
 namespace {
 
-constexpr std::size_t kN = 5000;
+// BUDGET_N overrides the stream length (harness only; default 5000). Drift spans the whole stream; the jump is at kN / 2.
+const std::size_t kN = std::getenv("BUDGET_N") ? std::strtoul(std::getenv("BUDGET_N"), nullptr, 10) : 5000;
 constexpr std::size_t kBurn = 500;
-constexpr std::size_t kJump = kN / 2;
+const std::size_t kJump = kN / 2;
 constexpr std::size_t kEvery = 5;          // score every 5th tick
 constexpr std::size_t kPost = 1000;        // post-change window
 constexpr unsigned kTestSeed = 42;
@@ -54,6 +55,32 @@ std::vector<double> gain_grid() {
 }
 const std::vector<double> kGains = gain_grid();
 
+// Harness only (Section 6.2 extension, 2026-11-15): BUDGET_EST=a,b restricts the estimators; BUDGET_KLL=k1,k2 replaces the
+// mpl_dskll capacity grid; BUDGET_TIME_GAIN sets the gain timed by `time` (default 0.02).
+std::vector<std::string> csv_env(const char* name) {
+    std::vector<std::string> v;
+    const char* e = std::getenv(name);
+    if (!e) return v;
+    std::string str(e);
+    for (std::size_t a = 0, b; a <= str.size(); a = b + 1) {
+        b = str.find(',', a);
+        if (b == std::string::npos) b = str.size();
+        if (b > a) v.push_back(str.substr(a, b - a));
+    }
+    return v;
+}
+bool est_on(const std::string& est) {
+    const auto v = csv_env("BUDGET_EST");
+    return v.empty() || std::find(v.begin(), v.end(), est) != v.end();
+}
+std::vector<int> ds_kll_grid() {
+    const auto v = csv_env("BUDGET_KLL");
+    if (v.empty()) return {8, 16, 32, 64, 128, 256, 512, 1024};
+    std::vector<int> k;
+    for (const auto& x : v) k.push_back(std::atoi(x.c_str()));
+    return k;
+}
+
 struct Cfg {
     std::string est;    // xiao | hermite | mpl_kll | mpl_exact | mpl_ew | mpl_fit_kll
     int approx;         // B, N, k or W
@@ -63,13 +90,13 @@ struct Cfg {
 std::vector<Cfg> build_cfgs() {
     std::vector<Cfg> v;
     for (double g : kGains) {
-        for (int b : {10, 20, 30, 45, 65, 90, 130, 180}) v.push_back({"xiao", b, g});
-        for (int n : {5, 10, 15, 20, 30, 43, 60, 89, 128}) v.push_back({"hermite", n, g});
-        for (int k : {8, 16, 32, 64, 128, 256, 512, 1024}) v.push_back({"mpl_kll", k, g});
-        for (int k : {8, 16, 32, 64, 128, 256, 512, 1024}) v.push_back({"mpl_dskll", k, g});
-        v.push_back({"mpl_exact", 0, g});
-        v.push_back({"mpl_ew", 500, g});
-        v.push_back({"mpl_fit_kll", 256, g});
+        if (est_on("xiao")) for (int b : {10, 20, 30, 45, 65, 90, 130, 180}) v.push_back({"xiao", b, g});
+        if (est_on("hermite")) for (int n : {5, 10, 15, 20, 30, 43, 60, 89, 128}) v.push_back({"hermite", n, g});
+        if (est_on("mpl_kll")) for (int k : {8, 16, 32, 64, 128, 256, 512, 1024}) v.push_back({"mpl_kll", k, g});
+        if (est_on("mpl_dskll")) for (int k : ds_kll_grid()) v.push_back({"mpl_dskll", k, g});
+        if (est_on("mpl_exact")) v.push_back({"mpl_exact", 0, g});
+        if (est_on("mpl_ew")) v.push_back({"mpl_ew", 500, g});
+        if (est_on("mpl_fit_kll")) v.push_back({"mpl_fit_kll", 256, g});
     }
     return v;
 }
@@ -120,7 +147,7 @@ Data simulate(const std::string& family, const std::string& scenario, unsigned s
     return d;
 }
 
-constexpr std::size_t kPts = (kN - kBurn) / kEvery;
+const std::size_t kPts = (kN - kBurn) / kEvery;
 
 struct Acc {
     std::vector<double> rmse, post;   // per seed
@@ -135,7 +162,9 @@ void run_split(const std::string& family, const std::string& scenario, unsigned
     for (auto& a : acc) { a.rmse.assign(n_seeds, 0.0); a.post.assign(n_seeds, 0.0); a.path.assign(kPts, 0.0); }
     truth_path.assign(kPts, 0.0);
     std::mutex mu;
-    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
+    // BUDGET_THREADS caps the worker threads (harness only; on a batch cluster set it to the allocated cores).
+    const unsigned hw = std::getenv("BUDGET_THREADS") ? std::max(1, std::atoi(std::getenv("BUDGET_THREADS")))
+                                                        : std::max(1u, std::thread::hardware_concurrency());
     std::vector<std::thread> pool;
     for (unsigned w = 0; w < hw; ++w) {
         pool.emplace_back([&, w] {
@@ -253,7 +282,8 @@ void sweep(std::size_t n_test, std::size_t n_val, const std::vector<std::string>
 // is the whole online pipeline per observation. Data are generated before the clock starts.
 void timing() {
     std::vector<Cfg> cfgs;
-    for (const Cfg& c : build_cfgs()) if (c.gain == 0.02) cfgs.push_back(c);
+    const double tg = std::getenv("BUDGET_TIME_GAIN") ? std::atof(std::getenv("BUDGET_TIME_GAIN")) : 0.02;
+    for (Cfg c : build_cfgs()) if (c.gain == kGains.front()) { c.gain = tg; cfgs.push_back(c); }
     std::ofstream out("results/budget_timing" + (std::getenv("TAG") ? std::string("_") + std::getenv("TAG") : std::string()) + ".csv");
     out << "estimator,approx,peak_bytes,mean_ns,p95_ns,update_mean_ns\n";
     for (const Cfg& c : cfgs) {
```
