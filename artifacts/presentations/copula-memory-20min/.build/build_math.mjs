import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { Presentation, PresentationFile } from '@oai/artifact-tool';

const SKILL='/Users/sallysun/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations';
const WD='/Users/sallysun/Documents/unsw/6thYear/datastreamer/artifacts/presentations/copula-memory-20min';
const OUT=path.join(WD,'output','copula-memory-mathematical-v3.pptx');
const {applyPresentationChartFont,finalizePresentation}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
const P=Presentation.create({slideSize:{width:1280,height:720}});
const C={bg:'#FBFBF8',ink:'#202B35',blue:'#246D8E',rust:'#BD654B',muted:'#60717A',pale:'#BFD6E0',line:'#CFD9DC',light:'#E9EFF0',green:'#5A786E'};
const font='Helvetica Neue';
function tx(s,v,x,y,w,h,z=27,b=false,c=C.ink){const a=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});a.text=v;a.text.style={typeface:font,fontSize:z,bold:b,color:c,autoFit:'none'};return a;}
function line(s,x,y,w,c=C.line,th=2){return s.shapes.add({geometry:'line',position:{left:x,top:y,width:w,height:0},fill:'none',line:{style:'solid',fill:c,width:th}});}
function box(s,x,y,w,h,c=C.light){return s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill:c,line:{fill:'none',width:0}});}
function base(t,n){const s=P.slides.add();s.background.fill=C.bg;tx(s,t,68,45,1140,71,36,true);line(s,68,128,1143);tx(s,String(n).padStart(2,'0'),1164,660,44,25,15,false,C.muted);return s;}
function notes(s,t){s.speakerNotes.textFrame.setText(t);}
const ax={fill:C.muted,fontSize:16,typeface:font};
function chart(s,type,cfg){const a=s.charts.add(type,{...cfg,chartFill:C.bg,plotAreaFill:C.bg,chartLine:{fill:'none',width:0},plotAreaLine:{fill:'none',width:0}});applyPresentationChartFont(a,{fontFamily:font});return a;}

// 1 — central question
{
 const s=P.slides.add();s.background.fill=C.bg;
 tx(s,'Choosing memory in online copula estimation',69,87,1125,70,44,true);
 tx(s,'How should an estimator trade statistical efficiency against adaptation when dependence changes?',69,208,1080,143,35,false);
 line(s,69,426,1136);
 tx(s,'A mathematical account of the estimators, adaptation rules, and their observed failure modes',69,460,1070,95,26,false,C.blue);
 tx(s,'20-minute thesis update',69,640,425,26,17,false,C.muted);
 notes(s,'Opening: 30 seconds. The target throughout is time-varying Spearman dependence of a bivariate copula. The common design variable is memory. I will show the estimators as update equations, derive the memory tradeoff, then examine whether the adaptive policies improve the decision criterion. Sources: notes/studies/memory_sweep.md; adaptive_memory.md; results_2026_09_22.md.');
}

// 2 — target and data-generating process
{
 const s=base('Target, observations, and online constraint',2);
 tx(s,'Latent pair',83,180,240,42,24,true,C.muted);
 tx(s,'(Uₜ,Vₜ) ∼ Cₜ     and     ρₛ,ₜ = 12 E[UₜVₜ] − 3',84,226,1080,72,35,true,C.blue);
 tx(s,'Observed prices',83,348,285,42,24,true,C.muted);
 tx(s,'log Pᵢ,ₜ − log Pᵢ,ₜ₋₁ = μᵢ + σᵢ Zᵢ,ₜ',84,391,1080,68,34,false,C.ink);
 line(s,84,501,1075);
 tx(s,'Online rule: ρ̂ₛ,ₜ may depend on observations 1,…,t only.',84,532,1030,60,28,true);
 tx(s,'Synthetic tests reveal ρₛ,ₜ to the scorer; a real feed does not.',84,600,1040,43,22,false,C.muted);
 notes(s,'1.5 minutes. U,V are copula uniforms. Spearman rho equals 12 E[UV]−3, so every estimator is reported on one target scale. The synthetic source draws normal scores from a selected copula and integrates returns into prices. The consumer recovers log returns. The truth is for offline scoring only and never enters the estimator. For the Gaussian copula, ρ_S=(6/π)arcsin(r/2). Source: experiments/copula_source.hpp, src/streamgenerator/datasource/synthetic_source.cpp.');
}

// 6 — ESS derivation
{
 const s=base('Fixed estimators: choose one memory M',3);
 tx(s,'Fixed means one preselected forgetting rate for the whole stream.',83,168,1100,55,27,true,C.blue);
 tx(s,'Weights: wⱼ = α(1−α)ʲ',83,233,1080,53,30,false,C.ink);
 tx(s,'ESS = 1 / ∑ⱼ wⱼ² = (2−α)/α',83,297,1080,62,32,true,C.blue);
 line(s,83,380,1086);
 tx(s,'Xiao, MLE',83,402,320,50,28,true);
 tx(s,'α ≈ 1−λ',420,402,470,50,28,false,C.blue);
 tx(s,'Hermite',83,465,320,50,28,true);
 tx(s,'α = λ',420,465,470,50,28,false,C.blue);
 tx(s,'Kalman',83,528,320,50,28,true);
 tx(s,'M in ticks ≈ m(2−K)/K',420,528,700,50,28,false,C.blue);
 tx(s,'Xiao and MLE select the same best memory in 20 of 24 tested cells.',83,618,1086,42,24,false,C.ink);
 notes(s,'2 minutes. Squared weights sum to α/(2−α), so ESS is its reciprocal. The two λ conventions are mirror images: Xiao retains previous counts, Hermite uses λ as the current update weight. MLE discounts Fisher information, giving nominal α≈1−λ. Kalman updates every block, so multiply block ESS by m. This is a nominal variance-sense axis; short Kalman memories hit the one-block floor. Step-response experiments verified the mapping in each method’s linear domain. Source: notes/studies/memory_sweep.md sections 1 and 3; results_2026_09_22.md section 1.');
}

// 3 — nonparametric routes
{
 const s=base('Two nonparametric representations',4);
 tx(s,'Xiao: empirical copula histogram',83,170,620,49,28,true,C.ink);
 tx(s,'Nᵢⱼ,ₜ = λNᵢⱼ,ₜ₋₁ + (1−λ)1{(Uₜ,Vₜ) ∈ Bᵢⱼ}',84,223,1080,63,30,false,C.blue);
 tx(s,'ρ̂ₛ,ₜ ≈ 12 ∑ᵢⱼ p̂ᵢⱼ,ₜ uᵢvⱼ − 3',84,288,1080,61,29,false,C.ink);
 line(s,84,376,1085);
 tx(s,'Hermite: smooth density expansion',83,405,620,49,28,true,C.ink);
 tx(s,'Aₜ = (1−α)Aₜ₋₁ + α h(xₜ)h(yₜ)ᵀ',84,458,1080,60,30,false,C.blue);
 tx(s,'ρ̂ₛ,ₜ = 12 (u−z/2)ᵀ Aₜ (v−z/2)',84,528,1080,62,30,false,C.ink);
 tx(s,'Different state representations; the same dependence functional.',84,615,1080,38,22,false,C.muted);
 notes(s,'2 minutes. Xiao discretizes the empirical copula in a 20-by-20 count grid, with calibrated marginal cutpoints. Forgetting multiplies past counts by λ. The displayed cell-centre formula is schematic for its rank-correlation readout. Hermite projects the joint structure into an order-10 basis; h(x)h(y)^T updates a coefficient matrix and its CDF projections yield Spearman rho. The exact implementation also tracks marginal coefficient vectors and debiases their EW warm-up, so the displayed quadratic is the compact functional form, not a verbatim listing of all accumulators. Source: src/estimators/dependence/xiao_spearman.hpp, hermiter.hpp; notes/studies/memory_sweep.md.');
}

// 4 — parametric route
{
 const s=base('Gaussian copula MLE as a recursive Fisher step',5);
 tx(s,'θ = atanh(r)        ρₛ = (6/π)asin(r/2)',86,173,1070,59,31,true,C.blue);
 tx(s,'sₜ = ∂ log cᵣ(Uₜ,Vₜ) / ∂θ       Iθ = 1 + r²',86,277,1070,68,31,false,C.ink);
 tx(s,'Hₜ = λHₜ₋₁ + Iθ',86,384,480,58,31,true,C.ink);
 tx(s,'θₜ = θₜ₋₁ + sₜ/Hₜ',636,384,530,58,31,true,C.ink);
 line(s,86,490,1079);
 tx(s,'At equilibrium: H ≈ Iθ/(1−λ), so the step has α ≈ 1−λ.',86,520,1090,91,27,false,C.blue);
 notes(s,'1.5 minutes. The Gaussian copula log-density yields a score in the unconstrained Fisher parameter θ=atanh r. Its information is 1+r². The algorithm discounts accumulated information and takes a score divided by current information. At stationary information, H is approximately I/(1−λ), exposing an update weight α=1−λ. The code standardizes marginals online first. This is a parametric advantage only when the Gaussian family is credible. Source: src/estimators/dependence/gaussian_mle.hpp; notes/studies/memory_sweep.md section 9.');
}

// 5 — Kalman and detector paths
{
 const s=base('Kalman filtering and detector attempts',6);
 tx(s,'Block observation',83,170,340,43,25,true,C.muted);
 tx(s,'yᵦ = atanh(ρ̂ₛ,ᵦ),    zᵦ = zᵦ₋₁ + ωᵦ,    yᵦ = zᵦ + vᵦ',84,221,1090,65,31,false,C.blue);
 tx(s,'Kᵦ = P⁻ᵦ/(P⁻ᵦ + R),     ẑᵦ = ẑ⁻ᵦ + Kᵦ(yᵦ − ẑ⁻ᵦ)',84,308,1090,64,29,false,C.ink);
 line(s,84,408,1079);
 tx(s,'Change detection on the standardized innovation νᵦ',84,439,1065,52,26,true,C.ink);
 tx(s,'CUSUM: gᵦ = max{0, gᵦ₋₁ + δνᵦ − δ²/2}',84,495,1090,64,29,false,C.blue);
 tx(s,'Tried CUSUM, GLR, and BOCPD; tuning memory mattered more than choosing a detector.',84,589,1075,65,22,false,C.muted);
 notes(s,'1.5 minutes. The local-level model runs on Fisher-transformed block Spearman estimates. R is approximately 1.06/(m−3), with m=20. The Kalman innovation can drive CUSUM, GLR, or BOCPD. These were measured at common false-alarm rates. BOCPD gave a structural improvement on the jump, but across the suite the forgetting variance affected RMSE about four times more than detector choice. Detector thresholds and process variance remain separate knobs. Source: notes/studies/kalman_changepoint_results.md sections 0.9–0.11 and findings; src/estimators/dependence/kalman_local_level.hpp.');
}

// 10 — adaptive alternatives
{
 const s=base('Adaptive rules: detector, VFF, and bank',7);
 tx(s,'Detector',85,178,250,49,29,true);
 tx(s,'Threshold innovation; reset or boost gain after an alarm',355,178,831,69,25,false,C.blue);
 line(s,85,264,1088,C.light,1);
 tx(s,'VFF',85,298,250,49,29,true);
 tx(s,'Cₜ ← Cₜ₋₁ + r(eₜ² − Cₜ₋₁); adjust α ∝ Cₜʳ',355,298,831,69,25,false,C.blue);
 line(s,85,386,1088,C.light,1);
 tx(s,'Bank',85,420,250,49,29,true);
 tx(s,'ℓₖ,ₜ = sₖ,ₜ²/(2v);  wₖ,ₜ ∝ wₖ,ₜ₋₁ e⁻ηℓₖ,ₜ',355,420,830,67,30,false,C.blue);
 tx(s,'wₖ,ₜ ← (1−p)wₖ,ₜ + p/K',355,490,830,61,29,false,C.ink);
 line(s,85,581,1088,C.line,2);
 tx(s,'Fixed share p keeps arms alive; η controls reaction speed.',85,608,1090,47,22,false,C.muted);
 notes(s,'1.5 minutes. A detector changes the update rule after an innovation crosses a threshold; we tried Kalman CUSUM, GLR and BOCPD, and score-based detectors on other estimators. VFF changes one forgetting parameter continuously from surprise. A bank runs K fixed-memory arms in parallel and scores their pre-update predictions; exponentiated losses update weights, and a fixed-share mixing step preserves the possibility of switching. η controls responsiveness, p is the assumed switching rate. The generic EstimatorBank uses squared rank residuals, not a genuine predictive likelihood. KalmanBank uses model predictive densities. Both are established algorithms, not a new algorithmic invention. Source: notes/studies/adaptive_memory.md sections 1, 5 and 17; results_2026_09_22.md section 9.');
}

// 15 — regret and comparator definitions
{
 const s=base('Regret needs a comparator and a window',8);
 tx(s,'Per-tick error:  ℓₜ(a) = (ρ̂ₜ⁽ᵃ⁾ − ρₜ)²',84,168,1080,56,29,true,C.blue);
 tx(s,'L[a,b)(a) = ∑ₜ₌ₐᵇ⁻¹ ℓₜ(a)',84,246,1080,54,29,false,C.ink);
 line(s,84,333,1085);
 tx(s,'Static regret',84,370,325,48,27,true);
 tx(s,'L[a,b)(π) − minₐ L[a,b)(a)',415,370,742,59,29,false,C.blue);
 tx(s,'Switching regret',84,463,325,48,27,true);
 tx(s,'L[a,b)(π) − minₐ₁:ₜ, switches≤K ∑ₜ ℓₜ(aₜ)',415,463,748,86,27,false,C.blue);
 tx(s,'The comparator uses hindsight; the online policy never sees ρₜ.',84,611,1080,42,23,false,C.muted);
 notes(s,'1.5 minutes. Define loss before quoting any regret. Static regret compares the adaptive rule with the best single fixed arm selected after seeing the same interval. Switching regret compares with a hindsight sequence allowed K switches, so it is harder. The window [a,b) is part of the metric, not a plotting detail. In synthetic runs rho_t is known to the scorer only. On a real feed neither RMSE nor these hindsight regrets exist; cumulative excess predictive loss versus a named running estimator can be reported instead. Later worst-case regret is a per-scenario MSE form, while capture and decomposition use cumulative squared loss.');
}

// 7 — U curve and observed optima
{
 const s=base('The target sets the best memory',9);
 const Ms=[20,30,45,70,100,140,200,300,450,700,1000],rd=x=>Math.round(x*1e6)/1e6;
 const v=Ms.map(m=>rd(1/m)),b=Ms.map(m=>rd((0.0005*m/2)**2)),tot=Ms.map(m=>rd(1/m+(0.0005*m/2)**2));
 chart(s,'line',{position:{left:80,top:175,width:810,height:396},categories:Ms.map(String),series:[{name:'MSE',values:tot,line:{style:'solid',fill:C.ink,width:4}},{name:'Variance',values:v,line:{style:'solid',fill:C.blue,width:3}},{name:'Lag²',values:b,line:{style:'solid',fill:C.rust,width:3}}],hasLegend:true,legend:{position:'bottom',overlay:false,textStyle:ax},xAxis:{title:'Memory M (log-spaced values)',textStyle:ax,majorGridlines:null},yAxis:{title:'Illustrative MSE',numberFormatCode:'0.000',textStyle:ax,majorGridlines:{style:'solid',fill:C.light,width:1}}});
 tx(s,'MSE ≈ σ²/M + (dM/2)²',916,181,290,118,28,true,C.blue);
 tx(s,'Best M, measured',916,329,287,44,23,true);
 tx(s,'Oscillate  36',916,381,288,43,25,false);
 tx(s,'Drift          235',916,431,288,43,25,false);
 tx(s,'Jump settled  495',916,481,288,72,25,false);
 tx(s,'Constant: optimum beyond 30,000-tick grid',84,612,1100,43,22,false,C.muted);
 notes(s,'2 minutes. The U curve is a theoretical local-linear illustration, not a fitted empirical curve. Variance decreases with M; drift lag grows roughly with dM/2. The measured optima move greatly by scenario. Constant asks for very long memory, while the 400-tick oscillation wants M=36. The jump has separate recovery and settled optima, so one whole-run best M hides the temporal conflict. The Xiao 20-bin histogram also has a small discretization bias floor. Source: notes/studies/memory_sweep.md section 4.');
}

// 8 — misspecification
{
 const s=base('The MLE’s edge is payment for knowing the family',10);
 chart(s,'bar',{position:{left:82,top:180,width:785,height:421},categories:['Gaussian','Gumbel','Clayton','Student t'],series:[{name:'MLE wins vs Xiao',values:[6,5,3,1],fill:C.blue}],barOptions:{direction:'bar',grouping:'clustered',gapWidth:70},hasLegend:false,xAxis:{min:0,max:6,majorUnit:1,title:'Wins among 6 matched-memory windows',textStyle:ax,majorGridlines:{style:'solid',fill:C.light,width:1}},yAxis:{textStyle:{...ax,fontSize:20},majorGridlines:null},dataLabels:{showValue:true,position:'outEnd',textStyle:{...ax,fontSize:20,fill:C.ink,bold:true}}});
 tx(s,'6 / 6',907,216,284,75,48,true,C.blue);
 tx(s,'on-family wins',908,294,286,52,24,false);
 tx(s,'+66.7%',907,407,284,74,43,true,C.rust);
 tx(s,'worst off-family MLE loss',908,480,280,82,22,false);
 tx(s,'All numbers are synthetic Gaussian/Clayton/Gumbel/t comparisons.',84,620,1050,36,19,false,C.muted);
 notes(s,'1.5 minutes. At matched memory, the Gaussian MLE beats Xiao in every Gaussian window. Across the other three copula families it wins 9 of 18 and can lose badly, with +66.7% in the worst observed Clayton cell. Xiao offers robustness when family is unknown. The main issue is not algorithm quality divorced from model specification. Source: notes/studies/results_2026_09_22.md section 2; memory_sweep.md section 9.');
}

// 9 — rate budget
{
 const s=base('Processing budget changes the memory optimum',11);
 tx(s,'MSE(Mₜᵢₘₑ) = σ²/(f Mₜᵢₘₑ) + (d Mₜᵢₘₑ/2)²',83,174,1110,77,31,true,C.blue);
 tx(s,'∂MSE/∂M = 0  ⇒  M*ₜᵢ꜀ₖₛ = [2σ²/(f d²)]¹ᐟ³ ∝ f⁻¹ᐟ³',83,260,1110,75,30,false,C.ink);
 chart(s,'bar',{position:{left:85,top:352,width:815,height:270},categories:['All ticks','Half','Quarter'],series:[{name:'Predicted',values:[1,1.26,1.59],fill:C.pale},{name:'Measured',values:[1,1.25,1.67],fill:C.blue}],barOptions:{direction:'column',grouping:'clustered',gapWidth:75},hasLegend:true,legend:{position:'bottom',overlay:false,textStyle:ax},xAxis:{textStyle:{...ax,fontSize:18},majorGridlines:null},yAxis:{min:0,max:2,majorUnit:0.5,title:'M* ratio',numberFormatCode:'0.00',textStyle:ax,majorGridlines:{style:'solid',fill:C.light,width:1}}});
 tx(s,'Thinning beat batching',940,411,257,90,27,true,C.ink);
 tx(s,'at the same update budget',940,509,260,81,23,false,C.muted);
 notes(s,'1.5 minutes. If only fraction f of incoming ticks is processed, M observations span M/f wall-clock ticks. Expressing variance in tick units gives σ²/(f M_ticks); lag bias remains d M_ticks/2. Minimization yields the cube-root law. At f=1/2 measured optimal memory ratio is 1.25 versus predicted 1.26; at f=1/4 it is 1.67 versus 1.59. This is for linear drift; jumps overshoot the prediction. Thinning uniformly in time beat contiguous batching in the tested conditions. Source: notes/studies/rate_budget.md.');
}

// 11 — minimax regret
{
 const s=base('A decision criterion for unknown regimes',12);
 tx(s,'R(π,e) = L(π,e) − minₐ L(a,e)',84,169,1080,60,30,true,C.blue);
 tx(s,'Choose π to minimise maxₑ R(π,e)',84,236,1080,55,29,false,C.ink);
 chart(s,'bar',{position:{left:85,top:315,width:845,height:303},categories:['Xiao','Hermite','Hermite db','MLE','Kalman'],series:[{name:'Bank',values:[2.50,2.36,2.33,2.50,4.77],fill:C.blue},{name:'Best fixed',values:[10.37,11.91,11.75,8.88,18.78],fill:C.pale}],barOptions:{direction:'column',grouping:'clustered',gapWidth:70},hasLegend:true,legend:{position:'bottom',overlay:false,textStyle:ax},xAxis:{textStyle:{...ax,fontSize:16},majorGridlines:null},yAxis:{title:'Worst-case regret × 10⁻³ MSE',min:0,max:20,majorUnit:5,textStyle:ax,majorGridlines:{style:'solid',fill:C.light,width:1}}});
 tx(s,'VFF regret',966,356,233,47,27,true,C.ink);
 tx(s,'Xiao 39.29',966,412,233,48,25,false,C.rust);
 tx(s,'Kalman 15.39',966,464,239,48,25,false,C.rust);
 tx(s,'×10⁻³ MSE',966,526,239,43,20,false,C.muted);
 notes(s,'2 minutes. For each scenario e, subtract the best loss achieved in that scenario by the comparator class. Then examine the worst regret across constant, drift, jump, oscillate. This avoids the hardest scenario swallowing the comparison. The bank ranks first among the tested candidate policies on this criterion in all five estimator families. The plotted values are ×10^-3 MSE, whereas a later slide uses cumulative squared loss; keep those units distinct. The best fixed setting is selected in hindsight over the scenario suite; the bank itself acts online without scenario labels. Source: notes/studies/results_2026_09_22.md section 4; adaptive_memory.md section 12.');
}

// 12 — switching comparator
{
 const s=base('How much of one ideal switch does the bank capture?',13);
 tx(s,'L*(K) = minₐ₁:ₜ:  switches(a₁:ₜ)≤K  ∑ₜ ℓₜ(aₜ)',84,172,1090,78,29,true,C.blue);
 tx(s,'Capture = [L*(0) − Lbank] / [L*(0) − L*(1)]',84,266,1090,65,29,false,C.ink);
 line(s,84,370,1090);
 tx(s,'Xiao',98,416,240,47,27,true);tx(s,'14.3%',409,416,210,47,28,false,C.rust);
 tx(s,'Gaussian MLE',98,483,278,47,27,true);tx(s,'14.0%',409,483,210,47,28,false,C.rust);
 tx(s,'Kalman',98,550,240,47,27,true);tx(s,'82.5%',409,550,210,47,28,false,C.blue);
 tx(s,'The K=1 comparator knows the jump time in hindsight.',760,450,435,118,24,false,C.muted);
 notes(s,'2 minutes. The K-switch comparator is a dynamic-programming oracle over arm sequences. K=0 is one fixed memory; K=1 permits one hindsight switch. Capture is the fraction of the one-switch improvement achieved by the online bank on the full jump run. Xiao and MLE banks captured about 14%; KalmanBank 82.5%. That gap reflects how the different bank weight updates react, but the comparator knows the jump time and cannot itself be deployed. Window choice changes the numbers, especially for generic banks. Source: notes/studies/results_2026_09_22.md section 5; adaptive_memory.md sections 13–14.');
}

// 13 — time decomposition
{
 const s=base('The bank wins before the jump and pays during recovery',14);
 chart(s,'bar',{position:{left:83,top:173,width:788,height:429},categories:['Before','Recovery','Settled'],series:[{name:'Excess loss',values:[-7647,4860,905],fill:C.blue,points:[{idx:0,fill:C.blue},{idx:1,fill:C.rust},{idx:2,fill:C.rust}]}],barOptions:{direction:'column',grouping:'clustered',gapWidth:80},hasLegend:false,xAxis:{textStyle:{...ax,fontSize:20},tickLabelPosition:'low',majorGridlines:null},yAxis:{min:-9000,max:6000,majorUnit:3000,title:'Excess squared loss × 10⁻³',numberFormatCode:'#,##0',textStyle:ax,majorGridlines:{style:'solid',fill:C.light,width:1}}});
 tx(s,'−7,647',912,214,280,66,39,true,C.blue);tx(s,'before the jump',914,282,282,40,22,false);
 tx(s,'+4,860',912,385,280,66,39,true,C.rust);tx(s,'during recovery',914,453,282,40,22,false);
 tx(s,'Whole run: −1,881',85,611,615,47,26,true);
 notes(s,'2 minutes. These are Xiao-bank cumulative squared-loss differences against the one best fixed Xiao memory selected over the whole jump run. Negative means the bank is ahead. It earns 7,647 units before the jump, loses 4,860 in recovery and 905 after settling. Thus a whole-run gain is not a recovery win. A faster learning rate η=4 can reverse recovery regret, but makes worst-case regret 13 times worse because of oscillate. This is the central failure mode and tradeoff. Source: notes/studies/results_2026_09_22.md sections 6 and 8; adaptive_memory.md section 15.');
}

// 14 — scoring windows and live validation
{
 const s=base('Scoring windows are part of the result',15);
 tx(s,'RMSE[a,b) = √{ ∑ₜ₌ₐᵇ⁻¹ (ρ̂ₜ − ρₜ)² / (b−a) }',84,169,1100,65,31,true,C.blue);
 line(s,84,262,1084);
 tx(s,'Bank, first 400 recovery ticks',84,291,560,44,27,true);
 tx(s,'Offline 0.2155     Live 0.2231',84,344,1050,52,29,false,C.ink);
 tx(s,'Offline, first 1000 recovery ticks',84,425,690,43,27,true);
 tx(s,'0.1437 — the recovered tail changes the aggregate.',84,476,1080,62,27,false,C.rust);
 line(s,84,560,1084);
 tx(s,'One unseeded run; same-input parity and pair IDs remain to be checked.',84,588,1085,60,23,false,C.muted);
 notes(s,'1 minute. Correct the initial live/offline impression. On the same 400-tick post-change window, Xiao bank RMSE is 0.2155 offline and 0.2231 live, 3.5 percent apart. Fixed Xiao differs more: 0.1944 offline versus 0.2216 live, about 14 percent. Comparing the live 400-tick number with offline 1000-tick 0.1437 created the false impression of live degradation. These are separate unseeded paths, so even matched windows establish only rough consistency, not parity. For real trades rho truth is unavailable, so show weights, mean memory, and prequential score; use fixed sampling plus trade-intensity and staleness controls for the Epps effect. Source: notes/studies/online_run_2026_09_22.md section 3.3.');
}

await fs.mkdir(path.join(WD,'.codex-finalizer'),{recursive:true});await fs.mkdir(path.join(WD,'output'),{recursive:true});await fs.mkdir(path.join(WD,'.build','math-render'),{recursive:true});
const candidate=path.join(WD,'.codex-finalizer','candidate-math.pptx');await(await PresentationFile.exportPptx(P)).save(candidate);
for(let i=0;i<P.slides.items.length;i++){const png=await P.export({slide:P.slides.items[i],format:'png',scale:1});await fs.writeFile(path.join(WD,'.build','math-render',`slide-${i+1}.png`),new Uint8Array(await png.arrayBuffer()));}
const result=await finalizePresentation({explicitTotalSlideCount:15,requiredNativeChartOwnerSlides:[9,10,11,12,14],materializeLiteralChartWorkbooks:true,workspaceDir:WD,candidatePath:candidate,finalPath:OUT,pythonExecutable:'/Users/sallysun/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit'],fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,receiptPath:path.join(WD,'.codex-finalizer','validation-math-v3.json')});
console.log(JSON.stringify({out:OUT,slideCount:P.slides.items.length,passed:result.presentationLayout?.finding_count===0&&result.nativeChartValidation?.passed}));
