import type { ReactNode } from "react";
import {
  Area, AreaChart, CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";

import { COLOURS, correlationDomain, formatNumber, type StreamPoint, type Timeline } from "./charts";
import "./StreamCharts.css";

const TICK = { fill: "#a5b9cb", fontSize: 11 };
const TIP = { background: "#0c1420", border: "1px solid #27394d", borderRadius: 6, fontSize: 12 };
const MARGIN = { top: 8, right: 12, bottom: 0, left: 0 };
const GRID = <CartesianGrid stroke="#27394d" vertical={false} />;
const MEMORY_COLOURS: Record<number, string> = {
  11: "#c0b3ff", 39: "#8d9ceb", 99: "#6dbee5", 199: "#79e1cb", 399: "#f7c378", 999: "#bd8ee4", 1999: "#ff8fa9",
};

function timeAxis(rows: StreamPoint[], timeline: Timeline) {
  if (timeline.formatStep) {
    return <XAxis dataKey="step" tick={TICK} minTickGap={36} tickFormatter={timeline.formatStep} />;
  }
  const start = rows[0]?.step ?? 0;
  const end = Math.max(start + 1, rows.at(-1)?.step ?? start);
  return <XAxis dataKey="step" type="number" domain={[start, end]} allowDataOverflow allowDecimals={false} tick={TICK} minTickGap={26} />;
}

function tooltip(timeline: Timeline, format: (value: number) => string) {
  return <Tooltip
    contentStyle={TIP}
    labelFormatter={(label) => timeline.formatStep?.(Number(label)) ?? `Observation ${label}`}
    formatter={(value) => format(Number(value))}
  />;
}

function eventLines(timeline: Timeline, showLabels = false) {
  return timeline.marks?.map(({ step, label }, i) => (
    <ReferenceLine key={i} x={step} stroke="#8297ad" strokeDasharray="4 3"
      label={showLabels ? { value: label, fill: "#b7c9db", fontSize: 10, position: "insideTopLeft" } : undefined} />
  ));
}

function Panel({ title, detail, children }: { title: string; detail?: ReactNode; children: ReactNode }) {
  return <section className="stream-panel"><h2>{title}{detail}</h2>{children}</section>;
}

function Plot({ timeline, short = false, children }: { timeline: Timeline; short?: boolean; children: ReactNode }) {
  return <div className={short ? "stream-plot short" : "stream-plot"}>
    {timeline.active !== false && <ResponsiveContainer>{children}</ResponsiveContainer>}
  </div>;
}

export function PairCharts({ rows, timeline, labels, title, detail, showTruth, showVff, fixedMemories }: {
  rows: StreamPoint[];
  timeline: Timeline;
  labels: [string, string];
  title: string;
  detail?: ReactNode;
  showTruth?: boolean;
  showVff?: boolean;
  fixedMemories?: [number, number];
}) {
  const last = rows.at(-1);
  const single = rows.length === 1;
  const rhoDomain = correlationDomain(rows.flatMap(({ mff, vff, truth, fixedShort, fixedLong }) =>
    [mff, truth, showVff ? vff : null, fixedMemories ? fixedShort : null, fixedMemories ? fixedLong : null]
  ));

  return (
    <div className="stream-chart-grid">
      <Panel title={title} detail={detail}>
        {last?.x == null || last.y == null ? (
          <p className="stream-plot stream-chart-message">Rebuild the consumer to display simulated prices.</p>
        ) : (
          <Plot timeline={timeline}>
            <LineChart data={rows} margin={MARGIN}>
              {GRID}{timeAxis(rows, timeline)}
              <YAxis domain={["auto", "auto"]} tick={TICK} width={44} tickFormatter={(n) => formatNumber(n, 1)} />
              {tooltip(timeline, formatNumber)}
              {eventLines(timeline)}
              <Line name={labels[0]} dataKey="x" stroke={COLOURS.x} strokeWidth={1.8} dot={single} isAnimationActive={false} />
              <Line name={labels[1]} dataKey="y" stroke={COLOURS.y} strokeWidth={1.8} dot={single} isAnimationActive={false} />
            </LineChart>
          </Plot>
        )}
      </Panel>
      <Panel title="MFF estimate of Spearman’s ρ" detail={last?.mff == null ? <small>Warming up</small> :
        fixedMemories && <small className="stream-fixed-legend">
          <span style={{ color: COLOURS.fixedShort }}>Short · {fixedMemories[0]}</span>
          <span style={{ color: COLOURS.fixedLong }}>Long · {fixedMemories[1]}</span>
        </small>}>
        <Plot timeline={timeline}>
          <LineChart data={rows} margin={MARGIN}>
            {GRID}{timeAxis(rows, timeline)}
            <YAxis domain={rhoDomain} allowDataOverflow tickCount={5} tick={TICK} width={44} tickFormatter={(n) => formatNumber(n)} />
            <ReferenceLine y={0} stroke="#495d72" />
            {tooltip(timeline, (n) => formatNumber(n, 3))}
            {eventLines(timeline, true)}
            {showTruth && <Line name="Simulated truth" dataKey="truth" stroke={COLOURS.truth} strokeDasharray="4 3" dot={single} isAnimationActive={false} />}
            {fixedMemories && <Line name={`Fixed memory ${fixedMemories[0]}`} dataKey="fixedShort" stroke={COLOURS.fixedShort} strokeDasharray="3 3" dot={false} isAnimationActive={false} />}
            {fixedMemories && <Line name={`Fixed memory ${fixedMemories[1]}`} dataKey="fixedLong" stroke={COLOURS.fixedLong} strokeDasharray="3 3" dot={false} isAnimationActive={false} />}
            <Line name="MFF estimate" dataKey="mff" stroke={COLOURS.mff} strokeWidth={2} dot={single} isAnimationActive={false} />
            {showVff && <Line name="VFF" dataKey="vff" stroke={COLOURS.vff} strokeDasharray="5 3" dot={single} isAnimationActive={false} />}
          </LineChart>
        </Plot>
      </Panel>
    </div>
  );
}

type Series = { label: string; colour: string };

export function MemoryCharts({ rows, timeline, memories, showVff }: {
  rows: StreamPoint[];
  timeline: Timeline;
  memories: number[];
  showVff?: boolean;
}) {
  const series = memories.map((memory) => ({ label: String(memory), colour: MEMORY_COLOURS[memory] ?? "#a5b9cb" }));
  const last = rows.at(-1);
  const ticks = showVff ? [39, 99, 199, 399, 999, 1999] : memories;
  return (
    <Panel title="MFF memory weights" detail={<small>Candidate effective sample sizes</small>}>
      <Allocation weights={last?.weights} series={series} initial={last?.mff == null} />
      <div className="stream-weight-grid">
        <div>
          <h3>Allocation over time</h3>
          <WeightHistory rows={rows} timeline={timeline} series={series} weightKey="weights" />
        </div>
        <div>
          <h3>Effective memory <small>{showVff ? "MFF · VFF" : "Weight-averaged candidate memory"}</small></h3>
          <Plot timeline={timeline} short>
            <LineChart data={rows} margin={MARGIN}>
              {GRID}{timeAxis(rows, timeline)}
              <YAxis scale="log" domain={[Math.min(...ticks) * 0.8, Math.max(...ticks) * 1.1]} ticks={ticks} tick={TICK} width={44} allowDataOverflow />
              {tooltip(timeline, (n) => formatNumber(n, 0))}
              {eventLines(timeline)}
              <Line name="MFF (weight-averaged)" dataKey={({ weights }: StreamPoint) =>
                weights?.reduce((sum, weight, i) => sum + weight * memories[i], 0) ?? null
              } stroke={COLOURS.mff} strokeWidth={2} dot={false} isAnimationActive={false} />
              {showVff && <Line name="VFF" dataKey="vffMemory" stroke={COLOURS.vff} strokeDasharray="5 3" dot={false} isAnimationActive={false} />}
            </LineChart>
          </Plot>
        </div>
      </div>
    </Panel>
  );
}

export function FamilyWeights({ rows, timeline, series }: { rows: StreamPoint[]; timeline: Timeline; series: Series[] }) {
  return (
    <Panel title="MFF family weights">
      <Allocation weights={rows.at(-1)?.familyWeights} series={series} initial={rows.at(-1)?.mff == null} />
      <div className="stream-family-history">
        <h3>Allocation over time</h3>
        <WeightHistory rows={rows} timeline={timeline} series={series} weightKey="familyWeights" />
      </div>
    </Panel>
  );
}

function Allocation({ weights, series, initial }: { weights?: number[]; series: Series[]; initial: boolean }) {
  return (
    <div className="stream-allocation">
      <h3>{initial ? "Initial weights" : "Current allocation"}</h3>
      {weights ? (
        <>
          <div className="stream-allocation-bar" aria-hidden="true">
            {series.map(({ label, colour }, i) => <span key={label} style={{ flexBasis: `${weights[i] * 100}%`, background: colour }} />)}
          </div>
          <ul>
            {series.map(({ label, colour }, i) => (
              <li key={label}><i style={{ background: colour }} />{label} <b>{formatNumber(weights[i] * 100, 1)}%</b></li>
            ))}
          </ul>
        </>
      ) : <small>Weights appear after warm-up.</small>}
    </div>
  );
}

function WeightHistory({ rows, timeline, series, weightKey }: {
  rows: StreamPoint[];
  timeline: Timeline;
  series: Series[];
  weightKey: "weights" | "familyWeights";
}) {
  return <Plot timeline={timeline} short>
    <AreaChart data={rows} margin={MARGIN}>
      {GRID}{timeAxis(rows, timeline)}
      <YAxis domain={[0, 1]} ticks={[0, 0.5, 1]} tickFormatter={(n) => `${n * 100}%`} tick={TICK} width={44} allowDataOverflow />
      {tooltip(timeline, (n) => `${formatNumber(n * 100, 1)}%`)}
      {eventLines(timeline)}
      {series.map(({ label, colour }, i) =>
        <Area key={label} dataKey={(row: StreamPoint) => row[weightKey]?.[i] ?? null} name={label}
          type="stepAfter" stackId="weights" stroke="none" fill={colour} fillOpacity={0.9} isAnimationActive={false} />
      )}
    </AreaChart>
  </Plot>;
}
