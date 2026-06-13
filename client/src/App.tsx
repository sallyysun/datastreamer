import { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

import "./App.css";

type MarketEvent = {
  price_x: number;
  price_y: number;
  volume_x?: number;
  volume_y?: number;
  return_x?: number;
  return_y?: number;
  timestamp: number;
};

export default function App() {
  const [status, setStatus] = useState("disconnected");
  const [error, setError] = useState<string | null>(null);

  const [data, setData] = useState<
    { time: number; price_x: number; price_y: number }[]
  >([]);

  useEffect(() => {
    // =====================================================
    // SAFE WS URL RESOLUTION (THIS FIXES YOUR ISSUE)
    // =====================================================

    const envUrl = import.meta.env.VITE_WS_URL;

    const fallbackUrl =
      window.location.hostname === "localhost"
        ? "ws://localhost:9002"
        : `wss://${window.location.host}`;

    const wsUrl = (envUrl && typeof envUrl === "string")
      ? envUrl
      : fallbackUrl;

    console.log("🔌 WebSocket connecting to:", wsUrl);

    let ws: WebSocket;

    try {
      ws = new WebSocket(wsUrl);
    } catch (e: any) {
      setError("WebSocket init failed");
      return;
    }

    ws.onopen = () => {
      setStatus("connected");
      setError(null);
    };

    ws.onclose = () => {
      setStatus("disconnected");
    };

    ws.onerror = (e) => {
      console.error("WebSocket error:", e);
      setError("WebSocket connection error");
    };

    ws.onmessage = (event) => {
      try {
        const msg: MarketEvent = JSON.parse(event.data);

        setData((prev) => {
          const next = [
            ...prev,
            {
              time: msg.timestamp ?? Date.now(),
              price_x: msg.price_x,
              price_y: msg.price_y,
            },
          ];

          return next.slice(-300);
        });
      } catch (err) {
        console.log("bad event");
      }
    };

    return () => ws.close();
  }, []);

  return (
    <div className="container">
      {/* HEADER */}
      <div className="header">
        <div>MARKET TERMINAL // LIVE STREAM</div>

        <div>
          <span className={status}>{status}</span>
          {error && <span style={{ color: "red", marginLeft: 10 }}>{error}</span>}
        </div>
      </div>

      {/* CHART */}
      <div className="panel" style={{ height: 350, marginTop: 10 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <XAxis
              dataKey="time"
              tickFormatter={(t) => new Date(t).toLocaleTimeString()}
            />

            <YAxis domain={["auto", "auto"]} />

            <Tooltip labelFormatter={(t) => new Date(t).toLocaleTimeString()} />
            <Legend />

            <Line
              type="monotone"
              dataKey="price_x"
              stroke="#00ff88"
              dot={false}
              strokeWidth={2}
            />

            <Line
              type="monotone"
              dataKey="price_y"
              stroke="#4da3ff"
              dot={false}
              strokeWidth={2}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* TABLE */}
      <div className="panel" style={{ marginTop: 10 }}>
        <table>
          <thead>
            <tr>
              <th>time</th>
              <th>price_x</th>
              <th>price_y</th>
              <th>volume_x</th>
              <th>volume_y</th>
              <th>return_x</th>
              <th>return_y</th>
            </tr>
          </thead>

          <tbody>
            {data.slice(-50).reverse().map((d, i) => (
              <tr key={i}>
                <td>{new Date(d.time).toLocaleTimeString()}</td>
                <td style={{ color: "#00ff88" }}>{d.price_x}</td>
                <td style={{ color: "#4da3ff" }}>{d.price_y}</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}