import { useEffect, useRef, useState } from "react";
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

// The MFF lab: start a private stream (the server's steerable generator) and a consumer running the MFF you build,
// steer the dependence (jump, drift, family switch, automatic events) and watch the MFF re-weight. Every number shown
// is computed by the consumer and arrives as a "state" message; the browser only draws it.

type Config = { estimator: string; memories: number[]; families: string[]; models: string[] };
type State = {
  t: number; rho: number; family: string; mff: number; vff: number; vff_neff: number; mff_neff: number;
  w_mem: number[]; w_fam: number[];
};
type Row = Record<string, number>;
type Mark = { t: number; label: string };

const MAX_ROWS = 900;
const FAMILIES = ["gaussian", "t", "clayton", "gumbel"];
const FAMILY_LABEL: Record<string, string> = { gaussian: "Gaussian", t: "Student t₄", clayton: "Clayton", gumbel: "Gumbel" };
const MEMORIES = [39, 99, 199, 399, 999, 1999];
const ESTIMATORS: [string, string][] = [["rmpl", "RMPL"], ["xiao", "Counting Matrix"], ["hermite", "Hermite Series"]];
const MEM_COLOURS = ["#0d0887", "#6a00a8", "#b12a90", "#e16462", "#fca636", "#f0f921"];
const FAM_COLOURS: Record<string, string> = { gaussian: "#9a9a9a", t: "#C2185B", clayton: "#8E6BB8", gumbel: "#E3B505" };
const C_TRUE = "#e8eef5", C_MFF = "#ff6b8b", C_VFF = "#b48cff";

function wsUrl() {
  const env = import.meta.env.VITE_WS_URL;
  if (env && typeof env === "string") return env;
  return window.location.hostname === "localhost" ? "ws://localhost:9002" : `wss://${window.location.host}`;
}

export default function Lab() {
  // generator
  const [copula, setCopula] = useState("gaussian");
  const [rho0, setRho0] = useState(0.5);
  const [rate, setRate] = useState(150);
  // MFF construction
  const [estimator, setEstimator] = useState("rmpl");
  const [mems, setMems] = useState<number[]>([39, 99, 199, 399, 1999]);
  const [fams, setFams] = useState<string[]>(["gaussian", "t", "clayton", "gumbel"]);
  const [eta, setEta] = useState(1);
  const [delta, setDelta] = useState(1e-4);
  // events
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
          const r: Row = { t: s.t, rho: s.rho, mff: s.mff, vff: s.vff, mffN: s.mff_neff, vffN: s.vff_neff };
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
    send({ action: "start", copula, rho: rho0, rate, estimator, memories: mems.join(","), families: fams.join(","), eta, delta });
  };
  const toggle = <T,>(list: T[], v: T, set: (x: T[]) => void, order: T[]) =>
    set(list.includes(v) ? list.filter((x) => x !== v) : order.filter((x) => x === v || list.includes(x)));

  const familyMff = (config?.families.length ?? 0) > 1;
  const tMin = rows.length ? rows[0].t : 0;
  const visibleMarks = marks.filter((m) => m.t >= tMin);
  const markLines = visibleMarks.map((m, i) => (
    <ReferenceLine key={i} x={m.t} stroke="#7a8798" strokeDasharray="4 3" label={{ value: FAMILY_LABEL[m.label] ?? m.label, fill: "#aab6c4", fontSize: 10, position: "insideTopLeft" }} />
  ));
  const xAxis = <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} stroke="#4b5563" tick={{ fontSize: 11 }} minTickGap={50} />;
  const grid = <CartesianGrid stroke="#1b2430" vertical={false} />;

  return (
    <div className="container lab">
      <header className="header">
        <div className="brand">
          <span className="logo" />
          <span className="brand-title">MFF lab</span>
          <span className="brand-sub">steer the dependence · watch the forgetting factors re-weight</span>
        </div>
        <span className={`status ${connected ? "ok" : ""}`}>{connected ? (running ? "streaming" : "connected") : "offline"}</span>
      </header>

      <div className="lab-grid">
        <aside className="controls">
          <section className="ctl">
            <h3>Generator</h3>
            <label>Copula family
              <select value={copula} onChange={(e) => setCopula(e.target.value)}>
                {FAMILIES.map((f) => <option key={f} value={f}>{FAMILY_LABEL[f]}</option>)}
              </select>
            </label>
            <label>Starting ρ<sub>S</sub> <b>{rho0.toFixed(2)}</b>
              <input type="range" min={0.05} max={0.9} step={0.05} value={rho0} onChange={(e) => setRho0(+e.target.value)} />
            </label>
            <label>Speed <b>{rate}</b> obs/s
              <input type="range" min={20} max={1000} step={10} value={rate}
                onChange={(e) => { setRate(+e.target.value); if (running) send({ action: "rate", rate: +e.target.value }); }} />
            </label>
          </section>

          <section className="ctl">
            <h3>Your MFF</h3>
            <label>Arm estimator
              <select value={estimator} onChange={(e) => setEstimator(e.target.value)}>
                {ESTIMATORS.map(([k, v]) => <option key={k} value={k}>{v}</option>)}
              </select>
            </label>
            <div className="field-label">Memories N<sub>eff</sub></div>
            <div className="chips">
              {MEMORIES.map((m) => (
                <button key={m} className={mems.includes(m) ? "chip on" : "chip"} onClick={() => toggle(mems, m, setMems, MEMORIES)}>{m}</button>
              ))}
            </div>
            <div className="field-label">Working families</div>
            <div className="chips">
              {FAMILIES.map((f) => (
                <button key={f} className={fams.includes(f) ? "chip on" : "chip"} onClick={() => toggle(fams, f, setFams, FAMILIES)}>{FAMILY_LABEL[f]}</button>
              ))}
            </div>
            <div className="row2">
              <label>η <input type="number" step={0.25} min={0} value={eta} onChange={(e) => setEta(+e.target.value)} /></label>
              <label>δ <input type="number" step={0.0001} min={0} max={0.5} value={delta} onChange={(e) => setDelta(+e.target.value)} /></label>
            </div>
            <p className="hint">{mems.length * fams.length} models. Gaussian alone with 39–1999 is the thesis MFF; several families give the family-augmented MFF of §6.5.</p>
            <div className="row2">
              <button className="primary" disabled={!connected} onClick={start}>{running ? "Restart" : "Start"}</button>
              <button disabled={!running} onClick={() => { send({ action: "stop" }); setRunning(false); }}>Stop</button>
            </div>
          </section>

          <section className="ctl">
            <h3>Change the dependence</h3>
            <div className="event">
              <label>Jump to ρ<sub>S</sub> <b>{jumpTo.toFixed(2)}</b>
                <input type="range" min={0.05} max={0.9} step={0.05} value={jumpTo} onChange={(e) => setJumpTo(+e.target.value)} />
              </label>
              <button disabled={!running} onClick={() => send({ action: "jump", rho: jumpTo })}>Jump</button>
            </div>
            <div className="event">
              <label>Drift to ρ<sub>S</sub> <b>{driftTo.toFixed(2)}</b> over <b>{driftSteps}</b> obs
                <input type="range" min={0.05} max={0.9} step={0.05} value={driftTo} onChange={(e) => setDriftTo(+e.target.value)} />
                <input type="range" min={100} max={5000} step={100} value={driftSteps} onChange={(e) => setDriftSteps(+e.target.value)} />
              </label>
              <button disabled={!running} onClick={() => send({ action: "drift", rho: driftTo, steps: driftSteps })}>Drift</button>
            </div>
            <div className="event">
              <label>Switch family to
                <select value={switchTo} onChange={(e) => setSwitchTo(e.target.value)}>
                  {FAMILIES.map((f) => <option key={f} value={f}>{FAMILY_LABEL[f]}</option>)}
                </select>
              </label>
              <button disabled={!running || !familyMff} onClick={() => send({ action: "family", family: switchTo })}>Switch</button>
            </div>
            {running && !familyMff && <p className="hint">Family switches need a family MFF: pick two or more working families and restart.</p>}
            <div className="row2">
              <button className={auto ? "on" : ""} disabled={!running} onClick={() => { send({ action: "auto", on: !auto }); setAuto(!auto); }}>
                {auto ? "Auto events: on" : "Auto events: off"}
              </button>
              <button className={paused ? "on" : ""} disabled={!running} onClick={() => { send({ action: "pause", on: !paused }); setPaused(!paused); }}>
                {paused ? "Resume" : "Pause"}
              </button>
            </div>
          </section>
          {error && <p className="err">{error}</p>}
        </aside>

        <main>
          <div className="tiles">
            <div className="tile"><div className="tile-k">observation</div><div className="tile-v">{last ? last.t : "--"}</div><div className="tile-f">{last ? FAMILY_LABEL[last.family] ?? last.family : ""} generating</div></div>
            <div className="tile"><div className="tile-k">true ρ<sub>S</sub></div><div className="tile-v">{last ? last.rho.toFixed(3) : "--"}</div></div>
            <div className="tile"><div className="tile-k">MFF ρ̂<sub>S</sub></div><div className="tile-v" style={{ color: C_MFF }}>{last ? last.mff.toFixed(3) : "--"}</div><div className="tile-f">memory ≈ {last ? last.mff_neff.toFixed(0) : "--"}</div></div>
            <div className="tile"><div className="tile-k">VFF ρ̂<sub>S</sub></div><div className="tile-v" style={{ color: C_VFF }}>{last ? last.vff.toFixed(3) : "--"}</div><div className="tile-f">memory ≈ {last ? last.vff_neff.toFixed(0) : "--"}</div></div>
          </div>

          {!running && !rows.length ? (
            <section className="panel empty"><div><div className="empty-title">Build an MFF and press Start</div>
              <p className="muted">You get your own stream and your own consumer. Then jump, drift or switch the copula family and watch the weights move.</p></div></section>
          ) : (
            <>
              <section className="panel">
                <div className="panel-head"><span>SPEARMAN'S ρ<sub>S</sub></span>
                  <span className="legend">
                    <span className="legwrap"><i className="sw" style={{ background: C_TRUE }} /> true</span>
                    <span className="legwrap"><i className="sw" style={{ background: C_MFF }} /> MFF</span>
                    <span className="legwrap"><i className="sw" style={{ background: C_VFF }} /> VFF</span>
                  </span>
                </div>
                <div className="chart">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={rows} margin={{ top: 6, right: 12, left: 0, bottom: 0 }}>
                      {grid}{xAxis}
                      <YAxis domain={[0, 1]} stroke="#4b5563" tick={{ fontSize: 11 }} width={40} />
                      <Tooltip contentStyle={tip} formatter={(v) => (typeof v === "number" ? v.toFixed(3) : String(v))} />
                      {markLines}
                      <Line type="stepAfter" dataKey="rho" name="true" stroke={C_TRUE} strokeWidth={1.5} dot={false} isAnimationActive={false} />
                      <Line type="monotone" dataKey="mff" name="MFF" stroke={C_MFF} strokeWidth={2} dot={false} isAnimationActive={false} />
                      <Line type="monotone" dataKey="vff" name="VFF" stroke={C_VFF} strokeWidth={1.5} strokeDasharray="5 3" dot={false} isAnimationActive={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </section>

              {config && (
                <section className="panel">
                  <div className="panel-head"><span>MFF WEIGHT BY MEMORY</span>
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
                  <div className="panel-head"><span>MFF WEIGHT BY WORKING FAMILY</span>
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
                <div className="panel-head"><span>EFFECTIVE MEMORY</span>
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
            </>
          )}
        </main>
      </div>
    </div>
  );
}

const memColour = (m: number, i: number) => MEM_COLOURS[Math.max(0, MEMORIES.indexOf(m)) % MEM_COLOURS.length] ?? MEM_COLOURS[i];
const tip = { background: "#0b0e13", border: "1px solid #26313f", borderRadius: 6, fontSize: 12 };
