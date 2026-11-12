# Chapter 6: structure and central argument

**This is how I want to develop Chapter 6.**

Section 6.3 is not a contest that MFF has to win in every experiment. It asks three things:
- when adaptive memory is worth having;
- how to choose memory from observable information;
- what the limits of doing so are.

## 1. Central argument

**6.2: the right effective sample size depends on how dependence evolves.**
- Long histories give precision under constant dependence.
- Intermediate histories balance variance against tracking bias under gradual drift.
- After an abrupt change, different memory lengths are best at different stages of recovery.

So a fixed forgetting factor works when the bias–variance compromise it needs stays the same, and becomes restrictive when that compromise changes within the stream.

**6.3: does online adaptation improve this trade-off?**
- VFF adjusts the memory of a single estimator with a prediction-error controller.
- MFF keeps several fixed-memory estimators and combines them using observable prediction losses.
- The comparison asks when the extra flexibility helps, what it costs when conditions are stable, and whether its advantages hold when VFF is favourably tuned.

**From performance to mechanism.** The true estimation error is not available online, so MFF has to weight its arms by a predictive scoring rule. We ask:
- whether squared or logarithmic loss gives more useful feedback;
- how the weights respond to change;
- whether fixed-share reweighting improves recovery.

**Beyond correlation.** Estimating Spearman's ρ accurately does not mean predicting the full dependence distribution accurately. The next section extends MFF from choosing memory lengths to choosing copula families, so it can adapt when the dependence structure changes without any change in correlation.

**Emphasis:** MFF is not valuable simply because it has more estimators. It is valuable if it uses observable evidence to combine memory lengths as their usefulness changes. That makes the comparison of scoring rules central, not a technical detail.

## 2. Structure

| Section | What it establishes | How it leads on |
|---|---|---|
| 6.2 Fixed Forgetting and Effective Sample Size | Different regimes favour different ESS values; an abrupt change can shift the preferred ESS within one stream. | Motivates online adaptation rather than one ESS chosen in hindsight. |
| 6.3.1 Experimental Configuration | Defines fixed forgetting, VFF and MFF; sets out tuning, marginal estimation, prequential evaluation and the hindsight benchmark. | Gives a consistent basis for the comparison. |
| 6.3.2 Comparison of Estimation Accuracy | Does adaptation help? Fixed forgetting is best under constant dependence and drift; MFF is best over the whole jump stream. The VFF sensitivity study supports MFF's advantage over the tested controller under constant dependence and jumps. | Having shown when MFF helps, asks whether it also improves density prediction and what feedback should guide it. |
| 6.3.3 Effect of the Weighting Loss | Uniform vs squared-loss vs log-loss weighting: does density-based feedback improve correlation tracking, predictive density, or both? | Shows how the choice of feedback shapes which memory lengths MFF favours. |
| 6.3.4 Weight Adaptation and Fixed-Share | Weight trajectories; whether fixed share lets candidates regain influence after a change. | Explains the adaptation mechanism and links to switching regret. |
| Regret (discussion or appendix) | Cumulative predictive performance against fixed and switching comparators, kept separate from estimation RMSE. | Completes the assessment of adaptive memory selection. |
| Copula-family adaptation (next section) | Whether adapting the copula family improves density prediction when dependence strength alone misses a change in structure. | Extends adaptation from historical weighting to the dependence model. |

**Order:** 6.3.3 comes before 6.3.4. First establish which feedback rule is useful, then how the algorithm uses it.

## 3. The four driving questions

1. **When does adaptive forgetting improve estimation?**
   - It depends on the regime: MFF is strongest under abrupt jumps.
   - A well-chosen fixed memory is stronger under constant dependence and gradual drift.
2. **What observable loss should guide adaptation?**
   - Squared loss targets a rank-correlation statistic; log loss scores the predictive density.
   - Do these objectives lead to materially different accuracy?
3. **How does MFF respond when the useful memory length changes?**
   - The weight trajectories show the reallocation.
   - Fixed share isolates whether keeping weight on previously poor candidates helps after a change.
4. **What happens when dependence changes in ways correlation can't describe?**
   - Tracking ρ_S doesn't identify the copula shape; the family bank tests whether adapting the family improves distributional prediction.

## 4. Distinctions to preserve

**Memory adaptation vs model adaptation.**
- 6.3 keeps the Gaussian working copula fixed and adapts the historical weighting.
- The family-bank section adapts the shape of the dependence.
- The density metrics in 6.3 use Gaussian plug-in forecasts, so they reflect both correlation error and any mismatch between the Gaussian working family and the true copula.
- So 6.3 is not a complete test of nonparametric copula-density estimation. It tests how forgetting strategies affect correlation tracking and the Gaussian predictive densities that follow from it.

**Performance vs explanation.** 6.3.2 reports results without attributing them early to a mechanism.
- The jump advantage alone doesn't show that MFF moves from long to short memory at the right time; the weight trajectories must show that.
- The VFF sensitivity study shows MFF beats the tested VFF rates under constant dependence and jumps. It doesn't isolate whether the cause is the parallel candidates, the scoring rule, fixed share or their combination. That is what 6.3.3 and 6.3.4 are for.

## 5. What not to add

- **No separate main-text subsections** for correlation accuracy, density accuracy, VFF sensitivity or regret unless a result really needs extended discussion.
- **6.3.2** moves from Spearman RMSE to the density metrics and folds in the VFF sensitivity finding; the detailed figures go to the appendix.
- **Weight adaptation and fixed share** belong together, because both are about how MFF changes its allocation.

## Three claims, kept separate

1. **Performance:** when does MFF help?
2. **Feedback:** which loss should set the weights?
3. **Mechanism:** how do the weights respond to changing dependence?

The family-bank section then introduces the next limitation: choosing the historical memory isn't enough when the dependence structure itself changes.

## Current evidence (12 November 2026)

Sources: `experiments/results/strategies_2026-11-12.txt`, `vffsens_2026-11-12.txt`, `fixed_ess_2026-11-12.txt`, `jump_recovery_2026-11-12.txt`.

**Construction:** RMPL on cumulative DataSketches KLL margins (k = 256), shared by the RMPL arms, the VFF controller and MFF scoring. 60 seeds; prequential forecasts.

**Findings:**
- **6.2:** the best ESS is no forgetting under constant dependence, about 360–490 under drift (Proposition prediction within 10%), and changes from no forgetting to 39 to 99 across a jump. Recovery time is about 1.15 N_eff.
- **6.3.2, RMSE** (family mean, ×10⁻²): MFF − hindsight-best fixed ESS is +0.5 (constant), +0.45 (drift) and −1.6 to −1.75 (jump). VFF (α = 0.02) ties MFF under drift and is worse under constant dependence and jumps.
- **VFF sensitivity** (α from 0.005 to 0.2):
  - the hindsight-best α differs by scenario: 0.02 constant, 0.01 drift, 0.075 jump;
  - MFF has lower RMSE at every α under constant dependence and jumps, and ties slow VFF under drift;
  - under drift, a hindsight-tuned slow VFF has slightly lower density error.
- **Density metrics** (Gaussian plug-in): same ordering as RMSE. Hermite's excess log loss is unreliable because of overshoot to |ρ̂| ≈ 1.
