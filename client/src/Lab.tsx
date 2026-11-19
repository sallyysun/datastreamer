import { useEffect, useRef, useState } from "react";

import "./Lab.css";
import { COLOURS, formatNumber, type StreamPoint } from "./charts";
import { FamilyWeights, MemoryCharts, PairCharts } from "./StreamCharts";
import { wsUrl } from "./server";
import StreamMetric from "./StreamMetric";

// The MFF lab: start a private stream (the server's steerable generator) and a consumer running the MFF you build,
// steer the dependence (jump, drift, family switch, automatic events) and watch the MFF re-weight. Every number shown
// is computed by the consumer and arrives as a "state" message; the browser only draws it.

type Config = { estimator: string; memories: number[]; families: string[]; models: string[] };
type State = {
  t: number; rho: number; family: string; mff: number; vff: number; vff_neff: number;
  prices?: [number, number];
  arms?: number[];
  w_mem: number[]; w_fam: number[]; mff_ready?: boolean; vff_ready?: boolean;
};
type EventKind = "jump" | "drift" | "family";
type Mark = { t: number; label: string };

const MAX_ROWS = 180;
const PLAYBACK_RATES = [
  [0.5, "1 obs / 2 sec"], [1, "1 obs / sec"], [1.5, "1.5 obs / sec"], [2, "2 obs / sec"], [5, "5 obs / sec"],
  [10, "10 obs / sec"],
] as const;
const FAMILIES = ["gaussian", "t", "clayton", "gumbel"];
const FAMILY_LABEL: Record<string, string> = { gaussian: "Gaussian", t: "Student t₄", clayton: "Clayton", gumbel: "Gumbel" };
const MEMORIES = [11, 39, 99, 199, 399, 999, 1999];
const ESTIMATORS: [string, string][] = [["rmpl", "RMPL"], ["xiao", "Counting Matrix"], ["hermite", "Hermite Series"]];
const FAM_COLOURS: Record<string, string> = { gaussian: "#9a9a9a", t: "#C2185B", clayton: "#8E6BB8", gumbel: "#E3B505" };

export default function Lab() {
  // generator
  const [copula, setCopula] = useState("gaussian");
  const [rho0, setRho0] = useState(0.3);
  const [rate, setRate] = useState(1.5);
  // MFF construction
  const [estimator, setEstimator] = useState("rmpl");
  const [mems, setMems] = useState<number[]>([11, 39, 99, 399]);
  const [fams, setFams] = useState<string[]>(["gaussian", "t", "clayton", "gumbel"]);
  const [eta, setEta] = useState(1);
  const [delta, setDelta] = useState(1e-4);
  // events
  const [eventKind, setEventKind] = useState<EventKind>("jump");
  const [jumpTo, setJumpTo] = useState(0.8);
  const [driftTo, setDriftTo] = useState(0.2);
  const [driftSteps, setDriftSteps] = useState(1000);
  const [switchTo, setSwitchTo] = useState("clayton");
  const [auto, setAuto] = useState(true);
  const [compareFixed, setCompareFixed] = useState(true);
  const [paused, setPaused] = useState(false);

  const [connected, setConnected] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [config, setConfig] = useState<Config | null>(null);
  const [rows, setRows] = useState<StreamPoint[]>([]);
  const [last, setLast] = useState<State | null>(null);
  const [marks, setMarks] = useState<Mark[]>([]);

  const wsRef = useRef<WebSocket | null>(null);
  const bufRef = useRef<StreamPoint[]>([]);
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
          bufRef.current.push({
            step: s.t, truth: s.rho,
            x: s.prices?.[0], y: s.prices?.[1],
            mff: s.mff_ready === false ? null : s.mff,
            vff: s.vff_ready === false ? null : s.vff,
            fixedShort: s.mff_ready === false ? null : s.arms?.[0],
            fixedLong: s.mff_ready === false ? null : s.arms?.at(-1),
            weights: s.w_mem,
            familyWeights: s.w_fam,
            vffMemory: s.vff_ready === false ? null : s.vff_neff,
          });
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
    setPaused(false);
    send({ action: "start", copula, rho: rho0, rate, auto, quick_start: true, estimator, memories: mems.join(","), families: fams.join(","), eta, delta });
  };
  const toggle = <T,>(list: T[], v: T, set: (x: T[]) => void, order: T[]) =>
    set(list.includes(v) ? list.filter((x) => x !== v) : order.filter((x) => x === v || list.includes(x)));

  const familyMff = (config?.families.length ?? 0) > 1;
  const fixedMemories: [number, number] | undefined = compareFixed && config && config.memories.length > 1
    ? [Math.min(...config.memories), Math.max(...config.memories)] : undefined;
  const timeline = {
    marks: marks.filter(({ t }) => t >= (rows[0]?.step ?? 0) && t <= (rows.at(-1)?.step ?? 0))
      .map(({ t, label }) => ({ step: t, label })),
  };
  const latestEvent = marks.at(-1);
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
              <label className="check-label"><input type="checkbox" checked={auto} onChange={(e) => {
                const on = e.target.checked;
                send({ action: "auto", on });
                setAuto(on);
              }} /> Guided events</label>
            </div>
            {auto && <p className="hint">Jump at observation 60. Drift at 155.</p>}
            {latestEvent && <p className="lab-event" role="status">
              <strong>{latestEvent.label}</strong>
              <span>Observation {latestEvent.t.toLocaleString()}</span>
            </p>}
          </section>
          <details className="ctl model-options">
            <summary>MFF settings <span>{mems.length * fams.length} models</span></summary>
            <label>Estimator
              <select value={estimator} onChange={(e) => setEstimator(e.target.value)}>
                {ESTIMATORS.map(([key, name]) => <option key={key} value={key}>{name}</option>)}
              </select>
            </label>
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
            <label className="check-label"><input type="checkbox" checked={compareFixed} onChange={(e) => setCompareFixed(e.target.checked)} /> Compare fixed memories</label>
          </details>
          {error && <p className="lab-error" role="alert">{error}</p>}
        </aside>

        <main className="lab-main">
          {!last ? (
            <section className="lab-empty">
              <h2>{running ? "Waiting for observations" : "Start a stream to see its estimates"}</h2>
              <p>{running ? "The first observation arrives after the consumer connects." : "Start the guided demo or apply your own events."}</p>
            </section>
          ) : (
            <>
              <dl className="stream-summary" aria-label="Stream summary">
                <StreamMetric label="Observations" value={last.t.toLocaleString()} />
                {last.prices && <>
                  <StreamMetric label="Asset X" value={formatNumber(last.prices[0])} colour={COLOURS.x} />
                  <StreamMetric label="Asset Y" value={formatNumber(last.prices[1])} colour={COLOURS.y} />
                </>}
                <StreamMetric label="Family" value={FAMILY_LABEL[last.family] ?? last.family} />
                <StreamMetric label="True ρ" value={last.rho.toFixed(3)} colour={COLOURS.truth} />
                <StreamMetric label="MFF estimate" value={last.mff_ready === false ? "—" : last.mff.toFixed(3)} colour={COLOURS.mff} />
                {!fixedMemories && <StreamMetric label="VFF" value={last.vff_ready === false ? "—" : last.vff.toFixed(3)} colour={COLOURS.vff} />}
              </dl>
              <PairCharts rows={rows} timeline={timeline} labels={["Asset X", "Asset Y"]} title="Simulated prices"
                showTruth showVff={!fixedMemories} fixedMemories={fixedMemories} />
              {config && <MemoryCharts rows={rows} timeline={timeline} memories={config.memories} showVff />}
              {config && familyMff && <FamilyWeights rows={rows} timeline={timeline} series={
                config.families.map((family) => ({ label: FAMILY_LABEL[family], colour: FAM_COLOURS[family] }))
              } />}
            </>
          )}
        </main>
      </div>
    </div>
  );
}
