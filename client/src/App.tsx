import { useEffect, useMemo, useState } from "react";
import {
  AreaChart,
  Area,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";

import "./App.css";

type Tick = {
  time: number;
  price_x: number;
  price_y: number;
  volume_x?: number;
  volume_y?: number;
  return_x?: number;
  return_y?: number;
};

const CORR_WINDOW = 60; // client-side rolling window (display only)

// Rolling Pearson correlation of the two return series -- a lightweight DISPLAY
// metric computed in the browser. The real online estimators (Xiao / Hermite /
// Gaussian-MLE) run in the C++ consumer devices, not here.
function rollingCorr(rows: Tick[], w: number) {
  const out: { time: number; corr: number }[] = [];
  for (let i = w - 1; i < rows.length; i++) {
    const slice = rows.slice(i - w + 1, i + 1);
    const xs = slice.map((r) => r.return_x ?? 0);
    const ys = slice.map((r) => r.return_y ?? 0);
    const mx = xs.reduce((a, b) => a + b, 0) / w;
    const my = ys.reduce((a, b) => a + b, 0) / w;
    let cov = 0, vx = 0, vy = 0;
    for (let k = 0; k < w; k++) {
      const dx = xs[k] - mx;
      const dy = ys[k] - my;
      cov += dx * dy;
      vx += dx * dx;
      vy += dy * dy;
    }
    const d = Math.sqrt(vx * vy);
    out.push({ time: rows[i].time, corr: d > 0 ? cov / d : 0 });
  }
  return out;
}

const fmt = (v: number | undefined, d = 2) =>
  typeof v === "number" ? v.toFixed(d) : "--";
const pct = (v: number | undefined) =>
  typeof v === "number" ? `${(v * 100).toFixed(3)}%` : "--";
const clock = (t: number) => new Date(t).toLocaleTimeString();

export default function App() {
  const [status, setStatus] = useState<"connected" | "disconnected">("disconnected");
  const [error, setError] = useState<string | null>(null);
  const [rows, setRows] = useState<Tick[]>([]);

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

    ws.onopen = () => { setStatus("connected"); setError(null); };
    ws.onclose = () => setStatus("disconnected");
    ws.onerror = () => setError("connection error");
    ws.onmessage = (event) => {
      try {
        const m: Tick = JSON.parse(event.data);
        setRows((prev) => [
          ...prev,
          {
            time: m.timestamp ?? Date.now(),
            price_x: m.price_x,
            price_y: m.price_y,
            volume_x: m.volume_x,
            volume_y: m.volume_y,
            return_x: m.return_x,
            return_y: m.return_y,
          } as Tick & { timestamp?: number },
        ].slice(-300));
      } catch {
        /* ignore malformed */
      }
    };
    return () => ws.close();
  }, []);

  const latest = rows.length ? rows[rows.length - 1] : null;
  const corrSeries = useMemo(() => rollingCorr(rows, CORR_WINDOW), [rows]);
  const latestCorr = corrSeries.length ? corrSeries[corrSeries.length - 1].corr : null;

  return (
    <div className="container">
      <header className="header">
        <div className="brand">
          <span className="logo" />
          <span className="brand-title">datastreamer</span>
          <span className="brand-sub">live market stream</span>
        </div>
        <div className={`status ${status}`}>
          <span className="dot" />
          {status}
          {error && <span className="err">· {error}</span>}
        </div>
      </header>

      {/* KPI TILES */}
      <div className="tiles">
        <div className="tile">
          <div className="tile-label">asset X</div>
          <div className="tile-value x">{fmt(latest?.price_x)}</div>
          <div className="tile-foot">ret {pct(latest?.return_x)}</div>
        </div>
        <div className="tile">
          <div className="tile-label">asset Y</div>
          <div className="tile-value y">{fmt(latest?.price_y)}</div>
          <div className="tile-foot">ret {pct(latest?.return_y)}</div>
        </div>
        <div className="tile">
          <div className="tile-label">rolling corr ({CORR_WINDOW})</div>
          <div className="tile-value" style={{ color: corrColor(latestCorr) }}>
            {latestCorr === null ? "--" : latestCorr.toFixed(3)}
          </div>
          <div className="tile-foot">client-side display</div>
        </div>
        <div className="tile">
          <div className="tile-label">ticks</div>
          <div className="tile-value muted">{rows.length}</div>
          <div className="tile-foot">last {clock(latest?.time ?? Date.now())}</div>
        </div>
      </div>

      {/* PRICE CHART */}
      <section className="panel">
        <div className="panel-head">
          <span>PRICES</span>
          <span className="legend">
            <i className="sw x" /> price_x&nbsp;&nbsp;<i className="sw y" /> price_y
          </span>
        </div>
        <div className="chart">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={rows} margin={{ top: 6, right: 12, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="gx" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#22d3ee" stopOpacity={0.35} />
                  <stop offset="100%" stopColor="#22d3ee" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="gy" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#a78bfa" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="#a78bfa" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="#1b2430" vertical={false} />
              <XAxis dataKey="time" tickFormatter={clock} stroke="#4b5563"
                     tick={{ fontSize: 11 }} minTickGap={60} />
              <YAxis domain={["auto", "auto"]} stroke="#4b5563" tick={{ fontSize: 11 }}
                     width={54} tickFormatter={(v) => v.toFixed(0)} />
              <Tooltip contentStyle={tooltipStyle} labelFormatter={clock}
                       formatter={(v: number) => v.toFixed(2)} />
              <Area type="monotone" dataKey="price_x" stroke="#22d3ee" strokeWidth={2}
                    fill="url(#gx)" dot={false} isAnimationActive={false} />
              <Area type="monotone" dataKey="price_y" stroke="#a78bfa" strokeWidth={2}
                    fill="url(#gy)" dot={false} isAnimationActive={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </section>

      {/* ROLLING CORRELATION CHART */}
      <section className="panel">
        <div className="panel-head">
          <span>ROLLING DEPENDENCE</span>
          <span className="legend muted">
            Pearson corr of returns, window {CORR_WINDOW} · display only
          </span>
        </div>
        <div className="chart short">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={corrSeries} margin={{ top: 6, right: 12, left: 0, bottom: 0 }}>
              <CartesianGrid stroke="#1b2430" vertical={false} />
              <XAxis dataKey="time" tickFormatter={clock} stroke="#4b5563"
                     tick={{ fontSize: 11 }} minTickGap={60} />
              <YAxis domain={[-1, 1]} ticks={[-1, -0.5, 0, 0.5, 1]} stroke="#4b5563"
                     tick={{ fontSize: 11 }} width={54} />
              <ReferenceLine y={0} stroke="#374151" />
              <Tooltip contentStyle={tooltipStyle} labelFormatter={clock}
                       formatter={(v: number) => v.toFixed(3)} />
              <Line type="monotone" dataKey="corr" stroke="#00e676" strokeWidth={2}
                    dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>

      {/* TABLE */}
      <section className="panel">
        <div className="panel-head"><span>TAPE</span></div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>time</th><th>price_x</th><th>price_y</th>
                <th>ret_x</th><th>ret_y</th><th>vol_x</th><th>vol_y</th>
              </tr>
            </thead>
            <tbody>
              {rows.slice(-40).reverse().map((d, i) => (
                <tr key={d.time + "-" + i} className={i === 0 ? "flash" : ""}>
                  <td className="muted">{clock(d.time)}</td>
                  <td className="x">{fmt(d.price_x)}</td>
                  <td className="y">{fmt(d.price_y)}</td>
                  <td className={retClass(d.return_x)}>{pct(d.return_x)}</td>
                  <td className={retClass(d.return_y)}>{pct(d.return_y)}</td>
                  <td className="muted">{fmt(d.volume_x, 0)}</td>
                  <td className="muted">{fmt(d.volume_y, 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

const tooltipStyle = {
  background: "#0d1117",
  border: "1px solid #1f2937",
  borderRadius: 8,
  fontSize: 12,
};

function corrColor(c: number | null) {
  if (c === null) return "#8b949e";
  if (c > 0.15) return "#00e676";
  if (c < -0.15) return "#ff5555";
  return "#e5e7eb";
}

function retClass(r?: number) {
  if (typeof r !== "number") return "muted";
  return r >= 0 ? "up" : "down";
}
