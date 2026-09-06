import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { Presentation, PresentationFile } from '@oai/artifact-tool';

const SKILL_DIR = '/Users/sallysun/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations';
const workspaceDir = '/Users/sallysun/Documents/unsw/6thYear/datastreamer/artifacts/presentations/copula-memory-20min';
const FINAL_PPTX = path.join(workspaceDir, 'output', 'copula-memory-20min-v2.pptx');
const { applyPresentationChartFont, finalizePresentation } = await import(pathToFileURL(path.join(SKILL_DIR, 'container_tools/artifact_tool_utils.mjs')).href);

const C = { bg:'#FBFBF8', ink:'#202B35', blue:'#246D8E', pale:'#BFD6E0', rust:'#BD654B', muted:'#61717B', line:'#CED8DC', light:'#E9EFF0', green:'#5A786E' };
const font = 'Helvetica Neue';
const pres = Presentation.create({ slideSize:{ width:1280, height:720 } });

function text(slide, value, x,y,w,h, size=27, bold=false, color=C.ink) {
  const s=slide.shapes.add({ geometry:'textbox', position:{left:x,top:y,width:w,height:h}, fill:'none', line:{fill:'none',width:0} });
  s.text=value;
  s.text.style={typeface:font,fontSize:size,bold,color,autoFit:'none'};
  return s;
}
function rect(slide,x,y,w,h,fill=C.light) { return slide.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:'none',width:0}}); }
function rule(slide,x,y,w,color=C.line,width=2) { return slide.shapes.add({geometry:'line',position:{left:x,top:y,width:w,height:0},fill:'none',line:{style:'solid',fill:color,width}}); }
function base(title,n) {
  const s=pres.slides.add(); s.background.fill=C.bg;
  text(s,title,66,46,1148,70,37,true);
  rule(s,66,128,1148,C.line,2);
  text(s,String(n).padStart(2,'0'),1163,661,50,25,15,false,C.muted);
  return s;
}
function note(s,body) { s.speakerNotes.textFrame.setText(body); }
function chart(slide,type,cfg) {
  const ch=slide.charts.add(type,{...cfg,chartFill:C.bg,plotAreaFill:C.bg,chartLine:{fill:'none',width:0},plotAreaLine:{fill:'none',width:0}});
  applyPresentationChartFont(ch,{fontFamily:font}); return ch;
}
const axisStyle={fill:C.muted,fontSize:16,typeface:font};

// 1. Research question
{
  const s=pres.slides.add(); s.background.fill=C.bg;
  text(s,'Memory in online dependence estimation',66,88,1150,74,45,true);
  text(s,'How much history should an estimator remember when dependence changes?',67,210,1040,140,36,false,C.ink);
  rule(s,67,412,1146,C.line,2);
  text(s,'Estimate ρₛ,t from a stream, without knowing whether the next regime is stable or changing.',67,444,1030,100,27,false,C.blue);
  text(s,'20-minute research presentation',67,642,450,25,16,false,C.muted);
  note(s,'Opening, 1 minute. The target is time-varying Spearman copula dependence. The problem is memory: too little history gives noisy estimates; too much history causes lag. Today I will show the mathematics, the synthetic evidence, and what the adaptive method actually achieved. Source: notes/studies/results_2026_09_22.md and manuscript.md.');
}

// 2. System
{
  const s=base('The online estimation path',2);
  const boxes=[['Pair source',75],['Producer',340],['Broker',605],['Consumer',870]];
  for(const [label,x] of boxes){ rect(s,x,250,205,113,C.light); text(s,label,x+17,279,180,62,30,true,C.ink); }
  for(const x of [290,555,820]) text(s,'→',x,277,45,65,39,false,C.blue);
  text(s,'Raw symbol ticks',73,389,220,42,21,false,C.muted);
  text(s,'Per-symbol topics',607,389,230,42,21,false,C.muted);
  text(s,'ρ̂ₛ,t and memory',874,389,275,42,21,false,C.blue);
  rule(s,75,470,1110,C.line,2);
  text(s,'Today’s measured results: reproducible synthetic experiments using the same estimator classes.',75,505,1060,75,25,false,C.ink);
  text(s,'Adaptive bank integration into the live consumer is the next implementation step.',75,582,1060,54,21,false,C.muted);
  note(s,'2 minutes. The producer and broker already publish paired synthetic observations as two per-symbol ticks. The standalone consumer reconstructs returns and runs online estimators. The reported adaptive-memory results come from experiments/probe_adaptive.cpp, not the browser or a live feed. The live consumer currently runs fixed Xiao and Hermite variants; connecting the new banks is future work. Source: src/streamgenerator/main.cpp, src/consumer/device_state.hpp, experiments/probe_adaptive.cpp.');
}

// 3. Theory U curve
{
  const s=base('The memory tradeoff',3);
  const Ms=[20,30,45,70,100,140,200,300,450,700,1000];
  const rnd=x=>Math.round(x*1e6)/1e6;
  const variance=Ms.map(m=>rnd(1/m)), lag=Ms.map(m=>rnd((0.0005*m/2)**2)), total=Ms.map((m,i)=>rnd(1/m+(0.0005*m/2)**2));
  chart(s,'line',{position:{left:86,top:170,width:1090,height:405},categories:Ms.map(String),
    series:[
      {name:'Total MSE',values:total,line:{style:'solid',fill:C.ink,width:4},marker:{symbol:'circle',size:5}},
      {name:'Variance',values:variance,line:{style:'solid',fill:C.blue,width:3}},
      {name:'Lag²',values:lag,line:{style:'solid',fill:C.rust,width:3}}
    ],hasLegend:true,legend:{position:'bottom',overlay:false,textStyle:axisStyle},
    xAxis:{title:'Effective memory M (log-spaced values)',textStyle:axisStyle,majorGridlines:null},
    yAxis:{title:'MSE (illustrative units)',textStyle:axisStyle,numberFormatCode:'0.000',majorGridlines:{style:'solid',fill:C.light,width:1}}});
  text(s,'MSE(M) ≈ σ²/M + (dM/2)²',295,603,720,50,30,true,C.ink);
  note(s,'3 minutes. This is the analytical approximation, not a fit to the experiment. For locally linear drift with slope d, variance decreases as 1/M and squared lag bias grows as (dM/2)^2. Their sum has an interior optimum; for a jump, this local linear approximation is not exact. The chart uses σ²=1 and d=0.0005 solely to illustrate the shape. Source: notes/studies/memory_sweep.md; notes/studies/rate_budget.md.');
}

// 4. Common axis
{
  const s=base('Different estimators, one memory axis',4);
  const rows=[
    ['Xiao','α = 1 − λ'],
    ['Hermite','α = λ'],
    ['Gaussian MLE','α ≈ 1 − λ  (Fisher step)'],
    ['Kalman','α ≈ K  (steady gain)']
  ];
  rows.forEach(([name,mapping],i)=>{
    const y=180+i*92;
    text(s,name,91,y,310,58,29,true,C.ink);
    text(s,mapping,425,y,690,58,27,false,C.blue);
    rule(s,91,y+62,1055,C.light,1);
  });
  text(s,'M ≈ (2 − α)/α',89,565,465,54,30,true,C.ink);
  text(s,'20 / 24',790,555,240,65,46,true,C.blue);
  text(s,'Xiao–MLE matched optima',792,617,365,36,21,false,C.ink);
  note(s,'2 minutes. Each method has its own parameterization. Xiao and Gaussian MLE use a retention λ, with α approximately 1−λ. Hermite uses λ as the update weight. Kalman has a gain implied by process and observation variance. Converting all of them to nominal effective memory creates one comparison axis. In 20 of 24 scenario-window-family cells, Xiao and Gaussian MLE picked the same best memory. This does not mean their errors are identical. Source: notes/studies/results_2026_09_22.md sections 1 and 9; notes/studies/memory_sweep.md section 9.');
}

// 5. Parametric risk
{
  const s=base('The parametric advantage depends on the family',5);
  chart(s,'bar',{position:{left:82,top:184,width:785,height:413},categories:['Gaussian','Gumbel','Clayton','Student t'],
    series:[{name:'Windows where MLE beats Xiao',values:[6,5,3,1],fill:C.blue}],
    barOptions:{direction:'bar',grouping:'clustered',gapWidth:70},hasLegend:false,
    xAxis:{min:0,max:6,majorUnit:1,title:'Wins out of 6 matched-memory windows',textStyle:axisStyle,majorGridlines:{style:'solid',fill:C.light,width:1}},
    yAxis:{textStyle:{...axisStyle,fontSize:20},majorGridlines:null},
    dataLabels:{showValue:true,position:'outEnd',textStyle:{...axisStyle,fill:C.ink,bold:true,fontSize:20}}});
  text(s,'6 / 6',907,215,250,85,50,true,C.blue);
  text(s,'on-family wins',908,298,290,49,25,false,C.ink);
  text(s,'+66.7%',907,399,275,78,43,true,C.rust);
  text(s,'worst observed off-family MLE loss',907,480,290,95,23,false,C.ink);
  text(s,'Matched-memory comparisons; 40 seeds per setting',84,619,720,28,17,false,C.muted);
  note(s,'2 minutes. The Gaussian MLE is well specified on Gaussian-copula data and beats Xiao in all six windows. Across the other three generating families it wins 9 of 18 cells, with a worst observed increase in loss of 66.7% on Clayton. The argument for nonparametric Xiao is robustness to unknown family, not that it always has the smallest error. Data: notes/studies/results_2026_09_22.md section 2, from memory_sweep.md section 9.');
}

// 6. Rate budget
{
  const s=base('Fewer updates shift the optimum',6);
  chart(s,'bar',{position:{left:86,top:180,width:833,height:410},categories:['All ticks','Half','Quarter'],
    series:[
      {name:'Predicted',values:[1,1.26,1.59],fill:C.pale},
      {name:'Measured',values:[1,1.25,1.67],fill:C.blue}
    ],barOptions:{direction:'column',grouping:'clustered',gapWidth:85},hasLegend:true,
    legend:{position:'bottom',overlay:false,textStyle:axisStyle},
    xAxis:{textStyle:{...axisStyle,fontSize:20},majorGridlines:null},
    yAxis:{min:0,max:2,majorUnit:0.5,title:'Optimal memory in ticks / full-rate optimum',textStyle:axisStyle,numberFormatCode:'0.00',majorGridlines:{style:'solid',fill:C.light,width:1}}});
  text(s,'M*ₜᵢ꜀ₖₛ ∝ f⁻¹ᐟ³',950,290,280,83,36,true,C.blue);
  text(s,'Linear drift',952,384,257,40,22,false,C.muted);
  text(s,'Thinning beat batching at equal budget in the tested cases.',84,611,1050,50,22,false,C.ink);
  note(s,'2 minutes. Here f is the fraction of incoming ticks the estimator processes, not the broker rate limit. An observation memory M spans M/f wall-clock ticks. On linear drift, minimizing σ²/(f M_ticks)+(d M_ticks/2)^2 gives M*_ticks proportional to f^(-1/3). Measured ratios were 1.25 at f=1/2 and 1.67 at f=1/4; predictions 1.26 and 1.59. This law is not expected to hold exactly at jumps. Thinning beat contiguous batching by 1.3 to 8.4 percent in the tested settings. Source: notes/studies/rate_budget.md.');
}

// 7. Bank and trajectory
{
  const s=base('The bank chooses memory from predictive loss',7);
  text(s,'7 memory settings',84,166,325,55,29,true,C.ink);
  text(s,'→',399,170,50,55,35,false,C.blue);
  text(s,'update weights',464,166,330,55,29,true,C.ink);
  text(s,'→',788,170,50,55,35,false,C.blue);
  text(s,'weighted estimate',848,166,350,55,29,true,C.ink);
  const ts=[500,1000,1500,2000,2400,2500,2550,2600,2700,3000,3500,4000,4500];
  const mem=[496.693,618.681,699.392,741.947,789.437,800.318,383.978,99.3825,55.1617,67.6524,105.103,154.379,209.442];
  chart(s,'scatter',{position:{left:82,top:242,width:1100,height:340},
    series:[{name:'Mean memory, Xiao bank',xValues:ts,values:mem,line:{style:'solid',fill:C.blue,width:4},marker:{symbol:'circle',size:5}}],
    scatterOptions:{style:'lineWithMarkers'},
    hasLegend:false,xAxis:{title:'Tick; jump at 2,500',min:0,max:5000,majorUnit:1000,textStyle:{...axisStyle,fontSize:15},majorGridlines:null},
    yAxis:{title:'Posterior-mean memory (ticks)',min:0,max:900,majorUnit:200,textStyle:axisStyle,majorGridlines:{style:'solid',fill:C.light,width:1}}});
  text(s,'Worst-case regret across tested scenarios: 3.6–5.0× lower than the best fixed setting',83,605,1105,50,23,true,C.ink);
  note(s,'3 minutes. Each bank runs the same estimator at several forgetting rates. The generic bank scores each arm by pre-update squared rank residual, applies exponential weights, and adds fixed-share weight so an arm can recover. It reports a weighted rho and posterior-mean memory. This plot uses the actual seed-averaged Xiao bank trajectory on the Gaussian jump experiment. It first favours long memory, then contracts after the change. The worst-case regret ratio compares candidate policies over four synthetic scenarios; it is not a real-feed result or a universal guarantee. Data: experiments/results/adaptive_memory_traj.csv; notes/studies/results_2026_09_22.md sections 4 and 9.');
}

// 8. Regret windows
{
  const s=base('The whole-run gain is earned before the jump',8);
  chart(s,'bar',{position:{left:83,top:180,width:780,height:420},categories:['Before','Recovery','Settled'],
    series:[{name:'Excess loss vs best fixed memory',values:[-7647,4860,905],fill:C.blue,
      points:[{idx:0,fill:C.blue},{idx:1,fill:C.rust},{idx:2,fill:C.rust}]}],
    barOptions:{direction:'column',grouping:'clustered',gapWidth:80},hasLegend:false,
    xAxis:{textStyle:{...axisStyle,fontSize:20},tickLabelPosition:'low',majorGridlines:null},
    yAxis:{min:-9000,max:6000,majorUnit:3000,title:'Cumulative excess squared loss × 10⁻³',textStyle:axisStyle,numberFormatCode:'#,##0',majorGridlines:{style:'solid',fill:C.light,width:1}}});
  text(s,'−7,647',907,214,278,70,40,true,C.blue);
  text(s,'before the jump',909,282,287,42,23,false,C.ink);
  text(s,'+4,860',907,379,278,70,40,true,C.rust);
  text(s,'during recovery',909,448,287,42,23,false,C.ink);
  text(s,'Total: −1,881',85,611,560,45,25,true,C.ink);
  note(s,'3 minutes. This is seed-averaged cumulative squared error against one best fixed Xiao arm selected over the complete jump run. Negative means the bank is ahead. The bank gains 7,647 ×10^-3 before the jump, then loses 4,860 ×10^-3 in recovery and another 905 ×10^-3 after settling; rounded values sum to about −1,881 ×10^-3 overall. Thus the whole-run result conceals the very period where rapid adaptation matters. This is why the talk reports time decomposition as well as scalar regret. Source: notes/studies/results_2026_09_22.md section 6 and notes/studies/adaptive_memory.md section 15.');
}

// 9. Conclusion
{
  const s=base('What the results establish',9);
  const ys=[188,302,416];
  const lines=[
    'Memory is a common statistical choice across estimator families.',
    'Adaptive selection is robust across the tested synthetic regimes.',
    'Recovery after a jump remains the main weakness.'
  ];
  ys.forEach((y,i)=>{ text(s,String(i+1),75,y,55,54,34,true,C.blue); text(s,lines[i],150,y,1035,75,28,false,C.ink); });
  rule(s,75,539,1110,C.line,2);
  text(s,'Next: run the bank in the live consumer and inspect memory beside trade intensity and price staleness.',75,566,1065,82,24,false,C.muted);
  note(s,'2 minutes. The contribution is empirical: a common memory axis, a measured bias-variance and rate tradeoff, and a time-resolved assessment of memory selection. The bank algorithms themselves are established forms of exponential weights or multiple-model averaging plus fixed share. The live integration remains to be done. On real trades there is no latent rho for RMSE or truth-based regret. Asynchronous sampling can change measured correlation without a change in latent dependence, so the real-feed demonstration must show trade intensity and price age alongside mean memory. Sources: notes/studies/results_2026_09_22.md; notes/studies/online_study_protocol.md; Epps (1979) https://doi.org/10.1080/01621459.1979.10482508; Zhang (2011) https://doi.org/10.1016/j.jeconom.2010.03.012.');
}

await fs.mkdir(path.join(workspaceDir,'.codex-finalizer'),{recursive:true});
await fs.mkdir(path.join(workspaceDir,'output'),{recursive:true});
const candidatePath=path.join(workspaceDir,'.codex-finalizer','candidate.pptx');
await (await PresentationFile.exportPptx(pres)).save(candidatePath);

// Render images for human review as well as package validation.
for (let i=0;i<pres.slides.items.length;i++) {
  const png=await pres.export({slide:pres.slides.items[i],format:'png',scale:1});
  await fs.writeFile(path.join(workspaceDir,'.build',`slide-${i+1}.png`),new Uint8Array(await png.arrayBuffer()));
}

const result=await finalizePresentation({
  explicitTotalSlideCount:9,
  requiredNativeChartOwnerSlides:[3,5,6,7,8],
  materializeLiteralChartWorkbooks:true,
  workspaceDir,
  candidatePath,
  finalPath:FINAL_PPTX,
  pythonExecutable:'/Users/sallysun/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',
  integrityValidatorPath:path.join(SKILL_DIR,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(SKILL_DIR,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit'],
  fontPolicy:{basis:'design',families:[font]},
  verifyArtifactToolImport:true,
  receiptPath:path.join(workspaceDir,'.codex-finalizer','validation-v2.json')
});
console.log(JSON.stringify({final:FINAL_PPTX,result},null,2));
