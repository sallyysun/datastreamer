import { useEffect, useMemo, useRef, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
} from "recharts";

import "./App.css";

// One single-symbol tick as published by the server (topic trades.equity.<SYM>).
type Tick = {
  symbol: string;
  price: number;
  volume?: number;
  timestamp: number;
  true_rho_s?: number;
};

// A merged row keyed by timestamp: { time, AAPL: price, MSFT: price, ... }. Both
// legs of a step share the same timestamp, so they land in the same row.
type Row = { time: number } & Record<string, number>;

const MAX_POINTS = 240; // rows retained in the price series
const MAX_TAPE = 60;
const TOPIC_PREFIX = "trades.equity.";
const topicFor = (sym: string) => TOPIC_PREFIX + sym;

// Deterministic per-symbol colour, stable across renders (index into catalog).
const PALETTE = ["#f6c177", "#a0a5d6", "#74ccaa", "#60b5be", "#e294b9"];

const fmt = (v: number | undefined, d = 2) =>
  typeof v === "number" ? v.toFixed(d) : "--";
const clock = (t: number) => new Date(t).toLocaleTimeString();

function symbolsFromUrl(): string[] {
  const raw = new URLSearchParams(window.location.search).get("symbols");
  if (!raw) return [];
  return raw.split(",").map((s) => s.trim()).filter(Boolean);
}

export default function App() {
  const [status, setStatus] = useState<"connected" | "disconnected">("disconnected");
  const [error, setError] = useState<string | null>(null);

  const [catalog, setCatalog] = useState<string[]>([]);
  const [selected, setSelected] = useState<string[]>(symbolsFromUrl);

  const [series, setSeries] = useState<Row[]>([]);
  const [tape, setTape] = useState<Tick[]>([]);
  const [count, setCount] = useState(0);

  // Display throttle: commit buffered ticks to the screen at most once per
  // `intervalSec`. Capped at 1s (fastest); the user can dial it slower.
  const [intervalSec, setIntervalSec] = useState(1);

  const wsRef = useRef<WebSocket | null>(null);
  const selectedRef = useRef(selected);
  const bufferRef = useRef<Tick[]>([]);
  const seriesMapRef = useRef<Map<number, Row>>(new Map());

  const colorFor = (sym: string) => {
    const i = catalog.indexOf(sym);
    return PALETTE[(i < 0 ? sym.length : i) % PALETTE.length];
  };

  // Keep the ref in sync so the WS open handler (created once) reads the live
  // selection when it resubscribes after a reconnect.
  useEffect(() => {
    selectedRef.current = selected;
  }, [selected]);

  // Reflect the selection in the URL so a link reopens the same view.
  useEffect(() => {
    const url = new URL(window.location.href);
    if (selected.length) url.searchParams.set("symbols", selected.join(","));
    else url.searchParams.delete("symbols");
    window.history.replaceState(null, "", url);
  }, [selected]);

  // ---- WebSocket: connect once, resubscribe on (re)open ----
  useEffect(() => {
    const envUrl = import.meta.env.VITE_WS_URL;
    const fallbackUrl =
      window.location.hostname === "localhost"
        ? "ws://localhost:9002"
        : `wss://${window.location.host}`;
    const wsUrl = envUrl && typeof envUrl === "string" ? envUrl : fallbackUrl;

    let ws: WebSocket;
    try {
      ws = new WebSocket(wsUrl);
    } catch {
      setError("WebSocket init failed");
      return;
    }
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus("connected");
      setError(null);
      const topics = selectedRef.current.map(topicFor);
      if (topics.length) ws.send(JSON.stringify({ action: "subscribe", topics }));
    };
    ws.onclose = () => setStatus("disconnected");
    ws.onerror = () => setError("connection error");
    ws.onmessage = (event) => {
      let m: unknown;
      try {
        m = JSON.parse(event.data);
      } catch {
        return;
      }
      const obj = m as Record<string, unknown>;
      if (obj.type === "catalog") {
        setCatalog(Array.isArray(obj.symbols) ? (obj.symbols as string[]) : []);
        return;
      }
      if (typeof obj.symbol === "string") {
        bufferRef.current.push(m as Tick); // flushed on the throttle interval
      }
    };
    return () => ws.close();
  }, []);

  // ---- Display throttle: flush the buffer into state on a timer ----
  useEffect(() => {
    const flush = () => {
      const buf = bufferRef.current;
      if (buf.length === 0) return;
      bufferRef.current = [];

      const map = seriesMapRef.current;
      for (const t of buf) {
        let row = map.get(t.timestamp);
        if (!row) {
          row = { time: t.timestamp } as Row;
          map.set(t.timestamp, row);
        }
        row[t.symbol] = t.price;
      }
      if (map.size > MAX_POINTS) {
        const keys = Array.from(map.keys()).sort((a, b) => a - b);
        for (let i = 0; i < keys.length - MAX_POINTS; i++) map.delete(keys[i]);
      }
      setSeries(Array.from(map.values()).sort((a, b) => a.time - b.time));
      setTape((prev) => [...[...buf].reverse(), ...prev].slice(0, MAX_TAPE));
      setCount((c) => c + buf.length);
    };

    const id = window.setInterval(flush, Math.max(1, intervalSec) * 1000);
    return () => window.clearInterval(id);
  }, [intervalSec]);

  // ---- Subscription toggling ----
  const toggle = (sym: string) => {
    const ws = wsRef.current;
    const isOn = selected.includes(sym);
    const next = isOn ? selected.filter((s) => s !== sym) : [...selected, sym];
    setSelected(next);
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(
        JSON.stringify({
          action: isOn ? "unsubscribe" : "subscribe",
          topics: [topicFor(sym)],
        })
      );
    }
    if (isOn) {
      // stop drawing a line for a symbol we no longer receive
      seriesMapRef.current.forEach((row) => delete row[sym]);
    }
  };

  const latestBySymbol = useMemo(() => {
    const last: Record<string, Tick> = {};
    for (const t of tape) if (!last[t.symbol]) last[t.symbol] = t; // tape is newest-first
    return last;
  }, [tape]);

  const intervalLabel =
    intervalSec === 1 ? "1s (max)" : `${intervalSec}s`;

  return (
    <div className="container">
      <header className="header">
        <div className="brand">
          <span className="logo" />
          <span className="brand-title">datastreamer</span>
          <span className="brand-sub">subscription viewer</span>
        </div>
        <div className={`status ${status}`}>
          <span className="dot" />
          {status}
          {error && <span className="err">· {error}</span>}
        </div>
      </header>

      {/* SUBSCRIPTION + THROTTLE BAR */}
      <div className="subbar">
        <span className="subbar-label">subscribe</span>
        {catalog.length === 0 && <span className="muted small">waiting for catalog…</span>}
        {catalog.map((sym) => {
          const on = selected.includes(sym);
          return (
            <button
              key={sym}
              className={`chip ${on ? "on" : ""}`}
              style={on ? { borderColor: colorFor(sym), color: "#e6edf3" } : undefined}
              onClick={() => toggle(sym)}
            >
              <span className="chip-dot" style={{ background: on ? colorFor(sym) : "#39424f" }} />
              {sym}
            </button>
          );
        })}

        <div className="throttle">
          <span className="throttle-label">show every</span>
          <input
            type="range"
            min={1}
            max={30}
            step={1}
            value={intervalSec}
            onChange={(e) => setIntervalSec(Number(e.target.value))}
            aria-label="display update interval, seconds"
          />
          <input
            type="number"
            min={1}
            value={intervalSec}
            onChange={(e) => setIntervalSec(Math.max(1, Number(e.target.value) || 1))}
            className="throttle-num"
            aria-label="display update interval in seconds"
          />
          <span className="throttle-unit">{intervalLabel}</span>
        </div>
      </div>

      {selected.length === 0 ? (
        <section className="panel empty">
          <div>
            <div className="empty-title">Pick a symbol to start streaming</div>
            <div className="muted">
              Each choice subscribes this connection to <code>{TOPIC_PREFIX}&lt;SYMBOL&gt;</code>.
              The browser only displays what it receives — no estimation runs here.
            </div>
          </div>
        </section>
      ) : (
        <>
          {/* PRICE CHART — one line per subscribed symbol */}
          <section className="panel">
            <div className="panel-head">
              <span>PRICES</span>
              <span className="legend">
                {selected.map((sym) => (
                  <span key={sym} className="legwrap">
                    <i className="sw" style={{ background: colorFor(sym) }} /> {sym}
                  </span>
                ))}
              </span>
            </div>
            <div className="chart">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={series} margin={{ top: 6, right: 12, left: 0, bottom: 0 }}>
                  <CartesianGrid stroke="#1b2430" vertical={false} />
                  <XAxis
                    dataKey="time"
                    tickFormatter={clock}
                    stroke="#4b5563"
                    tick={{ fontSize: 11 }}
                    minTickGap={60}
                  />
                  <YAxis
                    domain={["auto", "auto"]}
                    stroke="#4b5563"
                    tick={{ fontSize: 11 }}
                    width={54}
                    tickFormatter={(v) => v.toFixed(0)}
                  />
                  <Tooltip
                    contentStyle={tooltipStyle}
                    labelFormatter={(label) => clock(Number(label))}
                    formatter={(v) => (typeof v === "number" ? v.toFixed(2) : String(v))}
                  />
                  {selected.map((sym) => (
                    <Line
                      key={sym}
                      type="monotone"
                      dataKey={sym}
                      stroke={colorFor(sym)}
                      strokeWidth={2}
                      dot={false}
                      isAnimationActive={false}
                      connectNulls
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </section>

          {/* DEPENDENCE — placeholder until a consumer republishes estimates (Phase 3) */}
          {selected.length >= 2 && (
            <section className="panel">
              <div className="panel-head">
                <span>DEPENDENCE ρ̂</span>
                <span className="legend muted">estimator output · not yet wired</span>
              </div>
              <div className="dep-placeholder">
                <span className="badge">Phase 3</span>
                Online ρ̂ (Xiao / Hermite / Gaussian-MLE) runs in a C++ consumer device, not the
                browser. This panel lights up once a consumer republishes its estimates to
                <code>estimates.equity.&lt;PAIR&gt;</code> and the server bridges them here.
              </div>
            </section>
          )}

          {/* TAPE */}
          <section className="panel">
            <div className="panel-head"><span>TAPE</span><span className="muted">{count} ticks</span></div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>time</th><th>symbol</th><th>price</th><th>volume</th>
                  </tr>
                </thead>
                <tbody>
                  {tape.map((t, i) => (
                    <tr key={t.timestamp + t.symbol + i} className={i === 0 ? "flash" : ""}>
                      <td className="muted">{clock(t.timestamp)}</td>
                      <td style={{ color: colorFor(t.symbol) }}>{t.symbol}</td>
                      <td>{fmt(t.price)}</td>
                      <td className="muted">{fmt(t.volume, 0)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="lastrow">
              {selected.map((sym) => {
                const t = latestBySymbol[sym];
                return (
                  <span key={sym} className="lastcell">
                    <i className="sw" style={{ background: colorFor(sym) }} /> {sym}{" "}
                    <b>{fmt(t?.price)}</b>
                  </span>
                );
              })}
            </div>
          </section>
        </>
      )}
    </div>
  );
}

const tooltipStyle = {
  background: "#0d1117",
  border: "1px solid #1f2937",
  borderRadius: 8,
  fontSize: 12,
};
