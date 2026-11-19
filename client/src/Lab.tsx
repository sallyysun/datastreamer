import { useEffect, useMemo, useRef, useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import "./Lab.css";
import { correlationDomain } from "./charts";
import { wsUrl } from "./server";
import StreamMetric from "./StreamMetric";

// The MFF lab: start a private stream (the server's steerable generator) and a consumer running the MFF you build,
// steer the dependence (jump, drift, family switch, automatic events) and watch the MFF re-weight. Every number shown
// is computed by the consumer and arrives as a "state" message; the browser only draws it.

type Config = { estimator: string; memories: number[]; families: string[]; models: string[] };
type State = {
  t: number; rho: number; family: string; mff: number; vff: number; vff_neff: number; mff_neff: number;
  w_mem: number[]; w_fam: number[]; mff_ready?: boolean; vff_ready?: boolean;
};
type Row = Record<string, number | null> & { t: number; rho: number };
type EventKind = "jump" | "drift" | "family";
type Mark = { t: number; label: string };

const MAX_ROWS = 180;
const PLAYBACK_RATES = [
  [0.5, "1 obs / 2 sec"], [1, "1 obs / sec"], [2, "1 obs / 0.5 sec"], [5, "5 obs / sec"],
  [25, "25 obs / sec"], [150, "150 obs / sec"], [500, "500 obs / sec"], [1000, "1,000 obs / sec"],
] as const;
const FAMILIES = ["gaussian", "t", "clayton", "gumbel"];
const FAMILY_LABEL: Record<string, string> = { gaussian: "Gaussian", t: "Student t₄", clayton: "Clayton", gumbel: "Gumbel" };
const MEMORIES = [39, 99, 199, 399, 999, 1999];
const ESTIMATORS: [string, string][] = [["rmpl", "RMPL"], ["xiao", "Counting Matrix"], ["hermite", "Hermite Series"]];
const MEM_COLOURS = ["#0d0887", "#6a00a8", "#b12a90", "#e16462", "#fca636", "#f0f921"];
const FAM_COLOURS: Record<string, string> = { gaussian: "#9a9a9a", t: "#C2185B", clayton: "#8E6BB8", gumbel: "#E3B505" };
const C_TRUE = "#e8eef5", C_MFF = "#ff6b8b", C_VFF = "#b48cff";

export default function Lab() {
  // generator
  const [copula, setCopula] = useState("gaussian");
  const [rho0, setRho0] = useState(0.5);
  const [rate, setRate] = useState(2);
  // MFF construction
  const [estimator, setEstimator] = useState("rmpl");
  const [mems, setMems] = useState<number[]>([39, 99, 199, 399, 1999]);
  const [fams, setFams] = useState<string[]>(["gaussian", "t", "clayton", "gumbel"]);
  const [eta, setEta] = useState(1);
  const [delta, setDelta] = useState(1e-4);
  // events
  const [eventKind, setEventKind] = useState<EventKind>("jump");
  const [jumpTo, setJumpTo] = useState(0.8);
  const [driftTo, setDriftTo] = useState(0.2);
  const [driftSteps, setDriftSteps] = useState(1000);
  const [switchTo, setSwitchTo] = useState("clayton");
  const [auto, setAuto] = useState(false);
  const [paused, setPaused] = useState(false);

  const [connected, setConnected] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [config, setConfig] = useState<Config | null>(null);
  const [rows, setRows] = useState<Row[]>([]);
  const [last, setLast] = useState<State | null>(null);
  const [marks, setMarks] = useState<Mark[]>([]);

  const wsRef = useRef<WebSocket | null>(null);
  const bufRef = useRef<Row[]>([]);
  const lastRef = useRef<State | null>(null);

  useEffect(() => {
    let ws: WebSocket;
    let retry: number | undefined;
    const connect = () => {
      ws = new WebSocket(wsUrl());
      wsRef.current = ws;
      ws.onopen = () => { setConnected(true); setError(null); };
      ws.onclose = () => { setConnected(false); setRunning(false); retry = window.setTimeout(connect, 1500); };
      ws.onerror = () => setError("cannot reach the server");
      ws.onmessage = (m) => {
        const j = JSON.parse(m.data);
        if (j.type === "error") setError(j.message);
        else if (j.type === "started") { setRunning(true); setError(null); }
        else if (j.type === "config") setConfig(j);
        else if (j.type === "event") setMarks((m) => [...m, { t: j.t, label: j.label }].slice(-20));
        else if (j.type === "state") {
          const s = j as State;
          lastRef.current = s;
          const r: Row = {
            t: s.t, rho: s.rho,
            mff: s.mff_ready === false ? null : s.mff,
            vff: s.vff_ready === false ? null : s.vff,
            mffN: s.mff_neff, vffN: s.vff_neff,
          };
          s.w_mem.forEach((w, i) => (r["m" + i] = w));
          s.w_fam.forEach((w, i) => (r["f" + i] = w));
          bufRef.current.push(r);
        }
      };
    };
    connect();
    const timer = window.setInterval(() => {   // draw at most 10 times a second
      if (!bufRef.current.length) return;
      const add = bufRef.current;
      bufRef.current = [];
      setRows((old) => [...old, ...add].slice(-MAX_ROWS));
      setLast(lastRef.current);
    }, 100);
    return () => { window.clearInterval(timer); window.clearTimeout(retry); ws.onclose = null; ws.close(); };
  }, []);

  const send = (msg: object) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
  };

  const start = () => {
    if (!mems.length || !fams.length) { setError("pick at least one memory and one family"); return; }
    setRows([]); setMarks([]); setLast(null); setConfig(null); bufRef.current = []; lastRef.current = null;
    setAuto(false); setPaused(false);
    send({ action: "start", copula, rho: rho0, rate, quick_start: true, estimator, memories: mems.join(","), families: fams.join(","), eta, delta });
  };
  const toggle = <T,>(list: T[], v: T, set: (x: T[]) => void, order: T[]) =>
    set(list.includes(v) ? list.filter((x) => x !== v) : order.filter((x) => x === v || list.includes(x)));

  const familyMff = (config?.families.length ?? 0) > 1;
  const tMin = rows[0]?.t ?? 0;
  const tMax = rows.at(-1)?.t ?? tMin;
  const visibleMarks = marks.filter((m) => m.t >= tMin && m.t <= tMax);
  const markLines = visibleMarks.map((m, i) => (
    <ReferenceLine key={i} x={m.t} stroke="#8297ad" strokeDasharray="4 3" label={{ value: FAMILY_LABEL[m.label] ?? m.label, fill: "#b7c9db", fontSize: 10, position: "insideTopLeft" }} />
  ));
  const xAxis = <XAxis dataKey="t" type="number" domain={[tMin, Math.max(tMin + 1, tMax)]} allowDecimals={false} allowDataOverflow stroke="#60778f" tick={{ fontSize: 10 }} minTickGap={24} />;
  const grid = <CartesianGrid stroke="#233346" vertical={false} />;
  const rhoDomain = useMemo(() => correlationDomain(rows.flatMap(({ rho, mff, vff }) => [rho, mff, vff])), [rows]);
  const isWarmingUp = last?.mff_ready === false || last?.vff_ready === false;
  const canApply = running && (eventKind !== "family" || familyMff);

  const applyEvent = () => {
    if (eventKind === "jump") send({ action: "jump", rho: jumpTo });
    else if (eventKind === "drift") send({ action: "drift", rho: driftTo, steps: driftSteps });
    else send({ action: "family", family: switchTo });
  };

  return (
    <div className="container lab">
      <header className="lab-header">
        <div>
          <h1>MFF lab</h1>
          <p>Steer a stream and watch its dependence estimates adapt.</p>
        </div>
        <div className="lab-session">
          <span role="status" className={connected ? "lab-status connected" : "lab-status"}>
            {connected ? (running ? (paused ? "Paused" : "Streaming") : "Connected") : "Offline"}
          </span>
          {running && (
            <>
              <button className="lab-button" onClick={() => { send({ action: "pause", on: !paused }); setPaused(!paused); }}>
                {paused ? "Resume" : "Pause"}
              </button>
              <button className="lab-button" onClick={() => { send({ action: "stop" }); setRunning(false); setPaused(false); }}>Stop</button>
            </>
          )}
          <button className="lab-button primary" disabled={!connected} onClick={start}>{running ? "Restart" : "Start"}</button>
        </div>
      </header>

      <div className="lab-grid">
        <aside className="controls">
          <section className="ctl">
            <h2>Stream setup</h2>
            <label>Generating family
              <select value={copula} onChange={(e) => setCopula(e.target.value)}>
                {FAMILIES.map((f) => <option key={f} value={f}>{FAMILY_LABEL[f]}</option>)}
              </select>
            </label>
            <label>Estimator
              <select value={estimator} onChange={(e) => setEstimator(e.target.value)}>
                {ESTIMATORS.map(([key, name]) => <option key={key} value={key}>{name}</option>)}
              </select>
            </label>
            <label>Starting ρ<sub>S</sub> <b>{rho0.toFixed(2)}</b>
              <input type="range" min={0.05} max={0.9} step={0.05} value={rho0} onChange={(e) => setRho0(+e.target.value)} />
            </label>
            <label>Playback
              <select value={rate} onChange={(e) => {
                const speed = Number(e.target.value);
                setRate(speed);
                if (running) send({ action: "rate", rate: speed });
              }}>
                {PLAYBACK_RATES.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
              </select>
            </label>
            <details className="model-options">
              <summary>MFF settings <span>{mems.length * fams.length} models</span></summary>
              <fieldset>
                <legend>Candidate memories</legend>
                <div className="checks">
                  {MEMORIES.map((m) => <label key={m}><input type="checkbox" checked={mems.includes(m)} onChange={() => toggle(mems, m, setMems, MEMORIES)} />{m}</label>)}
                </div>
              </fieldset>
              <fieldset>
                <legend>Working families</legend>
                <div className="checks">
                  {FAMILIES.map((f) => <label key={f}><input type="checkbox" checked={fams.includes(f)} onChange={() => toggle(fams, f, setFams, FAMILIES)} />{FAMILY_LABEL[f]}</label>)}
                </div>
              </fieldset>
              <div className="row2">
                <label>Learning rate η <input type="number" step={0.25} min={0} value={eta} onChange={(e) => setEta(+e.target.value)} /></label>
                <label>Fixed share δ <input type="number" step={0.0001} min={0} max={0.5} value={delta} onChange={(e) => setDelta(+e.target.value)} /></label>
              </div>
            </details>
          </section>

          <section className="ctl">
            <h2>Change dependence</h2>
            <label>Event
              <select value={eventKind} onChange={(e) => setEventKind(e.target.value as EventKind)}>
                <option value="jump">Jump</option>
                <option value="drift">Gradual drift</option>
                <option value="family">Switch family</option>
              </select>
            </label>
            {eventKind === "family" ? (
              <label>New generating family
                <select value={switchTo} onChange={(e) => setSwitchTo(e.target.value)}>
                  {FAMILIES.map((f) => <option key={f} value={f}>{FAMILY_LABEL[f]}</option>)}
                </select>
              </label>
            ) : (
              <label>Target ρ<sub>S</sub> <b>{(eventKind === "jump" ? jumpTo : driftTo).toFixed(2)}</b>
                <input type="range" min={0.05} max={0.9} step={0.05} value={eventKind === "jump" ? jumpTo : driftTo}
                  onChange={(e) => (eventKind === "jump" ? setJumpTo : setDriftTo)(Number(e.target.value))} />
              </label>
            )}
            {eventKind === "drift" && (
              <label>Drift duration <b>{driftSteps} obs</b>
                <input type="range" min={100} max={5000} step={100} value={driftSteps} onChange={(e) => setDriftSteps(Number(e.target.value))} />
              </label>
            )}
            {eventKind === "family" && running && !familyMff && <p className="hint">Select at least two working families and restart to enable switching.</p>}
            <div className="event-actions">
              <button className="lab-button" disabled={!canApply} onClick={applyEvent}>Apply event</button>
              <label className="check-label"><input type="checkbox" checked={auto} disabled={!running} onChange={(e) => {
                const on = e.target.checked;
                send({ action: "auto", on });
                setAuto(on);
              }} /> Auto events</label>
            </div>
          </section>
          {error && <p className="lab-error" role="alert">{error}</p>}
        </aside>

        <main className="lab-main">
          {!last ? (
            <section className="panel lab-empty">
              <h2>{running ? "Waiting for observations" : "Start a stream to see its estimates"}</h2>
              <p>{running ? "The first observation arrives after the consumer connects." : "Start with the default settings, then apply a jump to see how the estimates respond."}</p>
            </section>
          ) : (
            <>
              <dl className="stream-summary" aria-label="Stream summary">
                <StreamMetric label="Observations" value={last.t.toLocaleString()} />
                <StreamMetric label="Family" value={FAMILY_LABEL[last.family] ?? last.family} />
                <StreamMetric label="True ρ" value={last.rho.toFixed(3)} colour={C_TRUE} />
                <StreamMetric label="MFF" value={last.mff_ready === false ? "—" : last.mff.toFixed(3)} colour={C_MFF} />
                <StreamMetric label="VFF" value={last.vff_ready === false ? "—" : last.vff.toFixed(3)} colour={C_VFF} />
              </dl>
              <section className="panel">
                <div className="panel-head"><span>Spearman’s ρ<sub>S</sub></span>
                  {isWarmingUp && <small>Warming up</small>}
                </div>
                <div className="chart">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={rows} margin={{ top: 6, right: 12, left: 0, bottom: 0 }}>
                      {grid}{xAxis}
                      <YAxis domain={rhoDomain} allowDataOverflow tickCount={5} tickFormatter={(v) => Number(v).toFixed(2)} stroke="#60778f" tick={{ fontSize: 10 }} width={44} />
                      <Tooltip contentStyle={tip} formatter={(v) => (typeof v === "number" ? v.toFixed(3) : String(v))} />
                      {markLines}
                      <Line type="stepAfter" dataKey="rho" name="true" stroke={C_TRUE} strokeWidth={1.5} dot={rows.length <= 12 ? { r: 2 } : false} isAnimationActive={false} />
                      <Line type="monotone" dataKey="mff" name="MFF" stroke={C_MFF} strokeWidth={2} dot={rows.length <= 12 ? { r: 2 } : false} isAnimationActive={false} />
                      <Line type="monotone" dataKey="vff" name="VFF" stroke={C_VFF} strokeWidth={1.5} strokeDasharray="5 3" dot={rows.length <= 12 ? { r: 2 } : false} isAnimationActive={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </section>

              <div className="lab-detail-grid">
                {config && (
                  <section className="panel">
                    <div className="panel-head"><span>Weights by memory</span>
                      <span className="legend">
                        {config.memories.map((m, i) => <span key={m} className="legwrap"><i className="sw" style={{ background: memColour(m, i) }} /> N<sub>eff</sub> {m}</span>)}
                      </span>
                    </div>
                    <div className="chart short">
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={rows} margin={{ top: 6, right: 12, left: 0, bottom: 0 }}>
                          {grid}{xAxis}
                          <YAxis domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} allowDataOverflow tickFormatter={(v) => Number(v).toFixed(2)} stroke="#4b5563" tick={{ fontSize: 11 }} width={40} />
                          {markLines}
                          {config.memories.map((m, i) => (
                            <Area key={m} type="monotone" dataKey={"m" + i} name={`N_eff ${m}`} stackId="m" stroke="none" fill={memColour(m, i)} fillOpacity={0.95} isAnimationActive={false} />
                          ))}
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  </section>
                )}

                {config && config.families.length > 1 && (
                  <section className="panel">
                    <div className="panel-head"><span>Weights by family</span>
                      <span className="legend">
                        {config.families.map((f) => <span key={f} className="legwrap"><i className="sw" style={{ background: FAM_COLOURS[f] }} /> {FAMILY_LABEL[f]}</span>)}
                      </span>
                    </div>
                    <div className="chart short">
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={rows} margin={{ top: 6, right: 12, left: 0, bottom: 0 }}>
                          {grid}{xAxis}
                          <YAxis domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} allowDataOverflow tickFormatter={(v) => Number(v).toFixed(2)} stroke="#4b5563" tick={{ fontSize: 11 }} width={40} />
                          {markLines}
                          {config.families.map((f, i) => (
                            <Area key={f} type="monotone" dataKey={"f" + i} name={FAMILY_LABEL[f]} stackId="f" stroke="none" fill={FAM_COLOURS[f]} fillOpacity={0.95} isAnimationActive={false} />
                          ))}
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  </section>
                )}

                <section className="panel">
                  <div className="panel-head"><span>Effective memory</span>
                    <span className="legend">
                      <span className="legwrap"><i className="sw" style={{ background: C_MFF }} /> MFF (weight-averaged N<sub>eff</sub>)</span>
                      <span className="legwrap"><i className="sw" style={{ background: C_VFF }} /> VFF</span>
                    </span>
                  </div>
                  <div className="chart short">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={rows} margin={{ top: 6, right: 12, left: 0, bottom: 0 }}>
                        {grid}{xAxis}
                        <YAxis scale="log" domain={[30, 2100]} ticks={[39, 99, 199, 399, 999, 1999]} stroke="#4b5563" tick={{ fontSize: 11 }} width={44} allowDataOverflow />
                        <Tooltip contentStyle={tip} formatter={(v) => (typeof v === "number" ? v.toFixed(0) : String(v))} />
                        {markLines}
                        <Line type="monotone" dataKey="mffN" name="MFF" stroke={C_MFF} strokeWidth={2} dot={false} isAnimationActive={false} />
                        <Line type="monotone" dataKey="vffN" name="VFF" stroke={C_VFF} strokeWidth={1.5} strokeDasharray="5 3" dot={false} isAnimationActive={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </section>
              </div>
            </>
          )}
        </main>
      </div>
    </div>
  );
}

const memColour = (m: number, i: number) => MEM_COLOURS[Math.max(0, MEMORIES.indexOf(m)) % MEM_COLOURS.length] ?? MEM_COLOURS[i];
const tip = { background: "#0b0e13", border: "1px solid #26313f", borderRadius: 6, fontSize: 12 };
