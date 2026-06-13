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
  type?: string;
};

export default function App() {
  const [status, setStatus] = useState("disconnected");

  const [data, setData] = useState<
    { time: number; price_x: number; price_y: number }[]
  >([]);

  useEffect(() => {
    const ws = new WebSocket(import.meta.env.VITE_WS_URL);

    ws.onopen = () => setStatus("connected");
    ws.onclose = () => setStatus("disconnected");

    ws.onmessage = (event) => {
      try {
        const msg: MarketEvent = JSON.parse(event.data);

        const point = {
          time: msg.timestamp ?? Date.now(),
          price_x: msg.price_x,
          price_y: msg.price_y,
        };

        setData((prev) => [...prev, point].slice(-300));
      } catch {
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
        <div className={status}>{status}</div>
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
            <Tooltip
              labelFormatter={(t) => new Date(t).toLocaleTimeString()}
            />

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

      {/* RAW FEED TABLE */}
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
                <td>{(d as any).volume_x ?? "-"}</td>
                <td>{(d as any).volume_y ?? "-"}</td>
                <td>{(d as any).return_x ?? "-"}</td>
                <td>{(d as any).return_y ?? "-"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}