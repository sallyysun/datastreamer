import { useEffect, useState } from "react";
import {
  CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";

import { replayUrl, wsUrl } from "./server";
import { correlationDomain } from "./charts";
import StreamMetric from "./StreamMetric";
import "./Market.css";

type Point = {
  step: number;
  x: number;
  y: number;
  mff: number | null;
  weights?: number[];
  truth?: number;
  timestamp?: number;
  date?: string;
};
type Replay = { id: string; name: string; labels: [string, string] };
type Stock = { id: string; symbol: string };
type Catalog = { type: "catalog"; symbols: [string, string]; replays?: Replay[]; stocks?: Stock[]; memories?: number[]; market?: boolean };

const WINDOW = 180;
const LIVE = "live";
const STOCKS = "stocks";
const REPLAY_INTERVAL_MS = 500;
const COLOURS = { x: "#60b5be", y: "#a0a5d6", mff: "#ff83a0", truth: "#a5b9cb" };
const WEIGHT_COLOURS = ["#8d9ceb", "#6dbee5", "#79e1cb", "#f7c378", "#ff8fa9"];
const TICK = { fill: "#a5b9cb", fontSize: 11 };
const TOOLTIP = { background: "#0c1420", border: "1px solid #27394d", borderRadius: 6, fontSize: 12 };
const format = (value?: number | null, digits = 2) => value == null ? "—" : value.toFixed(digits);

export default function Market({ active }: { active: boolean }) {
  const [connected, setConnected] = useState(false);
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [live, setLive] = useState<Point[]>([]);
  const [source, setSource] = useState(LIVE);
  const [pair, setPair] = useState<[string, string]>(["", ""]);
  const [recorded, setRecorded] = useState<Point[]>([]);
  const [position, setPosition] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let ws: WebSocket;
    let retry: number | undefined;
    const connect = () => {
      ws = new WebSocket(wsUrl());
      ws.onopen = () => {
        setConnected(true);
        setLive([]);
        ws.send(JSON.stringify({ action: "subscribe", topics: ["market.pair"] }));
      };
      ws.onclose = () => {
        setConnected(false);
        retry = window.setTimeout(connect, 1500);
      };
      ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        if (message.type === "catalog") setCatalog(message);
        if (message.type === "market") setLive((points) => [...points.slice(1 - WINDOW), message]);
      };
    };
    connect();
    return () => {
      window.clearTimeout(retry);
      ws.onclose = null;
      ws.close();
    };
  }, []);

  useEffect(() => {
    if (source === LIVE) return;
    const abort = new AbortController();
    fetch(replayUrl(source, source === STOCKS ? pair : undefined), { signal: abort.signal })
      .then((response) => {
        if (!response.ok) throw new Error("Replay unavailable");
        return response.json();
      })
      .then((points: Point[]) => {
        if (abort.signal.aborted) return;
        if (!points.length) setError("These stocks need at least two shared price dates.");
        setRecorded(points);
      })
      .catch(() => { if (!abort.signal.aborted) setError("Couldn’t load this replay."); })
      .finally(() => { if (!abort.signal.aborted) setLoading(false); });
    return () => abort.abort();
  }, [source, pair]);

  const finished = recorded.length > 0 && position === recorded.length;

  useEffect(() => {
    if (!playing || finished || !active || source === LIVE) return;
    const timer = window.setInterval(() => {
      setPosition((shown) => Math.min(shown + 1, recorded.length));
    }, REPLAY_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [active, finished, playing, recorded.length, source]);

  const isLive = source === LIVE;
  const isStockPair = source === STOCKS;
  const stocks = catalog?.stocks ?? [];
  const replays = catalog?.replays ?? [];

  function selectSource(value: string) {
    setSource(value);
    if (value === STOCKS && !pair[0] && stocks.length >= 2) setPair([stocks[0].id, stocks[1].id]);
    resetReplay(value !== LIVE);
  }

  function resetReplay(load: boolean) {
    setRecorded([]);
    setPosition(0);
    setPlaying(false);
    setLoading(load);
    setError("");
  }

  const replay = replays.find(({ id }) => id === source);
  const labels = isStockPair
    ? pair.map((id) => stocks.find((stock) => stock.id === id)?.symbol ?? "")
    : (isLive ? catalog?.symbols : replay?.labels) ?? ["X", "Y"];
  const points = isLive ? live : recorded.slice(Math.max(0, position - WINDOW), position);
  const last = points.at(-1);
  const memories = catalog?.memories ?? [];
  let waiting = error || (loading ? "Loading replay…" : "Press Play to start");
  if (isLive) {
    if (!connected) waiting = "Waiting for the server";
    else if (catalog && catalog.market === undefined) waiting = "Restart the Datastreamer backend to enable Market MFF.";
    else if (catalog?.market === false) waiting = "Market simulation is available when the server runs in interactive mode.";
    else waiting = "Waiting for observations";
  }
  const rows = points.map((point) => ({
    ...point,
    when: point.date ?? (point.timestamp == null ? "" : new Date(point.timestamp).toLocaleTimeString()),
  }));
  const formatStep = (step: number) => rows.find((point) => point.step === Number(step))?.when ?? "";
  const rhoDomain = correlationDomain(points.flatMap(({ mff, truth }) => [mff, truth]));

  return (
    <main className="container market">
      <header className="market-heading">
        <h1>Market MFF</h1>
        {isLive && <span className={`market-status ${connected && catalog?.market ? "on" : ""}`}>
          {connected ? catalog?.market ? "Live simulation" : "Connected" : "Disconnected"}
        </span>}
      </header>

      <div className="market-toolbar">
        <label className="market-source">
          Data
          <select value={source} onChange={(event) => selectSource(event.target.value)}>
            <option value={LIVE}>Live · {catalog?.symbols?.join(" / ") ?? "market pair"}</option>
            {stocks.length >= 2 && <option value={STOCKS}>Saved stocks</option>}
            {replays.map(({ id, name, labels }) => (
              <option key={id} value={id}>Replay · {labels[0] === "X" && labels[1] === "Y" ? name : labels.join(" / ")}</option>
            ))}
          </select>
        </label>
        {isStockPair && pair.map((id, index) => (
          <select key={index} aria-label={index === 0 ? "First stock" : "Second stock"} value={id} onChange={(event) => {
            const next: [string, string] = [...pair];
            next[index] = event.target.value;
            setPair(next);
            resetReplay(true);
          }}>
            {stocks.filter((stock) => stock.id !== pair[1 - index]).map((stock) =>
              <option key={stock.id} value={stock.id}>{stock.symbol}</option>
            )}
          </select>
        ))}
        {!isLive && (
          <>
            <button type="button" className="market-play" disabled={!recorded.length} onClick={() => {
              if (finished || position === 0) setPosition(1);
              setPlaying(finished || !playing);
            }}>{playing && !finished ? "Pause" : "Play"}</button>
            <button type="button" disabled={position === 0} onClick={() => {
              setPosition(1);
              setPlaying(true);
            }}>Restart</button>
            <span className="market-progress">{REPLAY_INTERVAL_MS / 1000} s per observation</span>
          </>
        )}
        {isLive && catalog?.market && !replays.length && stocks.length < 2 && <small>No saved stock data configured</small>}
      </div>

      {last ? (
        <>
          <dl className="stream-summary" aria-label="Market summary">
            <StreamMetric label="Observations" value={isLive ? last.step.toLocaleString() : `${position.toLocaleString()} / ${recorded.length.toLocaleString()}`} />
            <StreamMetric label={labels[0]} value={format(last.x)} colour={COLOURS.x} />
            <StreamMetric label={labels[1]} value={format(last.y)} colour={COLOURS.y} />
            <StreamMetric label="MFF" value={last.mff == null ? "Warming up" : format(last.mff, 3)} colour={COLOURS.mff} />
            {isLive && <StreamMetric label="True ρ" value={format(last.truth, 3)} colour={COLOURS.truth} />}
          </dl>
          <div className="market-charts">
            <section className="market-panel">
              <h2>{isLive ? "Simulated prices" : isStockPair ? "Log returns (%)" : "Recorded returns"}
                <time dateTime={last.date ?? (last.timestamp == null ? undefined : new Date(last.timestamp).toISOString())}>{rows.at(-1)?.when}</time>
              </h2>
              <div className="market-chart">
                {active && <ResponsiveContainer>
                  <LineChart data={rows} margin={{ top: 9, right: 10, bottom: 0, left: 0 }}>
                    <CartesianGrid stroke="#27394d" vertical={false} />
                    <XAxis dataKey="step" tickFormatter={formatStep} tick={TICK} minTickGap={36} />
                    <YAxis domain={["auto", "auto"]} tick={TICK} width={48} tickFormatter={(n) => format(n, 1)} />
                    <Tooltip contentStyle={TOOLTIP} labelFormatter={(label) => formatStep(Number(label))} formatter={(value) => format(Number(value))} />
                    <Line name={labels[0]} dataKey="x" stroke={COLOURS.x} strokeWidth={1.8} dot={rows.length === 1} isAnimationActive={false} />
                    <Line name={labels[1]} dataKey="y" stroke={COLOURS.y} strokeWidth={1.8} dot={rows.length === 1} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>}
              </div>
            </section>

            <section className="market-panel">
              <h2>Spearman dependence</h2>
              <div className="market-chart">
                {active && <ResponsiveContainer>
                  <LineChart data={rows} margin={{ top: 9, right: 10, bottom: 0, left: 0 }}>
                    <CartesianGrid stroke="#27394d" vertical={false} />
                    <XAxis dataKey="step" tickFormatter={formatStep} tick={TICK} minTickGap={36} />
                    <YAxis domain={rhoDomain} allowDataOverflow tickCount={5} tick={TICK} width={44} tickFormatter={(n) => format(n)} />
                    <ReferenceLine y={0} stroke="#495d72" />
                    <Tooltip contentStyle={TOOLTIP} labelFormatter={(label) => formatStep(Number(label))} formatter={(value) => format(Number(value), 3)} />
                    {isLive && <Line name="Simulated truth" dataKey="truth" stroke={COLOURS.truth} strokeDasharray="4 3" dot={false} isAnimationActive={false} />}
                    <Line name="MFF" dataKey="mff" stroke={COLOURS.mff} strokeWidth={2} dot={false} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>}
              </div>
            </section>
          </div>

          <section className="market-panel market-weights">
            <h2>MFF memory weights <small>Candidate effective sample sizes</small></h2>
            <div className="market-weight-body">
              {last.weights ? (
                <>
                  <div className="market-weight-bar" aria-hidden="true">
                    {last.weights.map((weight, i) => <span key={memories[i]} style={{ flexBasis: `${weight * 100}%`, background: WEIGHT_COLOURS[i] }} />)}
                  </div>
                  <ul>
                    {last.weights.map((weight, i) => (
                      <li key={memories[i]}><i style={{ background: WEIGHT_COLOURS[i] }} />{memories[i]} <b>{format(weight * 100, 1)}%</b></li>
                    ))}
                  </ul>
                </>
              ) : <small>Weights appear once the estimator has enough observations.</small>}
            </div>
          </section>
        </>
      ) : <p className="market-empty" role="status">{waiting}</p>}
    </main>
  );
}
