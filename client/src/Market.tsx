import { useEffect, useRef, useState } from "react";

import { wsUrl } from "./server";
import { COLOURS, formatNumber, type StreamPoint } from "./charts";
import { MemoryCharts, PairCharts } from "./StreamCharts";
import StreamMetric from "./StreamMetric";
import "./Market.css";

type Frame = StreamPoint & {
  observations: number;
  timestamp?: number;
  date?: string;
  event?: string;
};
type MarketSource = {
  symbols: string[];
  memories: number[];
  readyAt: number;
};
type Source = "simulation" | "replay" | "thesis";
type Sources = {
  simulation?: MarketSource;
  replay?: MarketSource;
  thesis?: { pairs: (MarketSource & { id: string })[] };
};
type Pair = [number, number];
type Subscription = { mode: Source; pair: Pair; pairId?: string };

const SOURCE_LABEL: Record<Source, string> = {
  simulation: "Live simulation",
  replay: "Saved-price replay",
  thesis: "Thesis return replay",
};

const WINDOW = 180;

export default function Market({ active }: { active: boolean }) {
  const [connected, setConnected] = useState(false);
  const [sources, setSources] = useState<Sources>({});
  const [mode, setMode] = useState<Source>("simulation");
  const [frames, setFrames] = useState<Frame[]>([]);
  const [selection, setSelection] = useState<number[]>([]);
  const [thesisPair, setThesisPair] = useState("");
  const [subscription, setSubscription] = useState<Subscription | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const socket = useRef<WebSocket | null>(null);
  const nextRequest = useRef(0);
  const requested = useRef<{ id: number; subscription: Subscription } | null>(null);
  const receiving = useRef(false);

  useEffect(() => {
    let ws: WebSocket;
    let retry: number | undefined;
    const connect = () => {
      ws = new WebSocket(wsUrl());
      socket.current = ws;
      ws.onopen = () => {
        setConnected(true);
        setSources({});
        setFrames([]);
        setSelection([]);
        setThesisPair("");
        setSubscription(null);
        setPending(false);
        setError("");
        requested.current = null;
        receiving.current = false;
      };
      ws.onclose = () => {
        setConnected(false);
        setPending(false);
        retry = window.setTimeout(connect, 1500);
      };
      ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        if (message.type === "catalog" && message.market) {
          const catalogue = message.market as Sources;
          setSources(catalogue);
          setMode((current) => catalogue[current] ? current : "simulation");
        } else if (message.type === "market_subscribed" && message.requestId === requested.current?.id && requested.current) {
          setSubscription(requested.current.subscription);
          requested.current = null;
          receiving.current = true;
          setFrames([]);
          setPending(false);
        } else if (message.type === "market_error" && message.requestId === requested.current?.id) {
          requested.current = null;
          setError(message.message);
          setPending(false);
        } else if (message.type === "market" && receiving.current) {
          setFrames((previous) => message.step === 1 ? [message] : [...previous.slice(1 - WINDOW), message]);
        }
      };
    };
    connect();
    return () => {
      window.clearTimeout(retry);
      ws.onclose = null;
      ws.close();
    };
  }, []);

  function chooseSource(next: Source) {
    if (next === mode) return;
    setMode(next);
    setSelection([]);
    setThesisPair("");
    setError("");
  }

  function select(index: number) {
    setSelection((current) => current.includes(index)
      ? current.filter((item) => item !== index)
      : [...current.slice(-1), index]);
  }

  const selected = mode === "thesis" ? (thesisPair ? [0, 1] : []) : [...selection].sort((a, b) => a - b);
  const pairs = sources.thesis?.pairs ?? [];
  const chosenSource = mode === "thesis" ? pairs.find(({ id }) => id === thesisPair) : sources[mode];
  const pairId = mode === "thesis" ? thesisPair : undefined;
  const unchanged = selected.length === 2 && subscription?.mode === mode && subscription.pairId === pairId
    && subscription.pair[0] === selected[0] && subscription.pair[1] === selected[1];
  const canSubscribe = connected && Boolean(chosenSource) && !pending && selected.length === 2 && !unchanged;

  function subscribe() {
    const ws = socket.current;
    if (!canSubscribe || ws?.readyState !== WebSocket.OPEN) return;
    const requestId = ++nextRequest.current;
    requested.current = { id: requestId, subscription: { mode, pair: [selected[0], selected[1]], pairId } };
    setPending(true);
    setError("");
    ws.send(JSON.stringify({ action: "market_subscribe", source: mode, pair: selected, pairId, requestId }));
  }

  function unsubscribe() {
    const ws = socket.current;
    if (!subscription || pending || ws?.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ action: "market_unsubscribe" }));
    receiving.current = false;
    setSubscription(null);
    setFrames([]);
    setError("");
  }

  const showing = subscription?.mode ?? mode;
  const source = showing === "thesis"
    ? pairs.find(({ id }) => id === subscription?.pairId)
    : sources[showing];
  const labels: [string, string] = subscription && source
    ? [source.symbols[subscription.pair[0]], source.symbols[subscription.pair[1]]] : ["", ""];
  const last = frames.at(-1);
  const latestEvent = frames.findLast((item) => item.event);
  const when = (item?: Frame) => item?.date ??
    (item?.timestamp === undefined ? "" : new Date(item.timestamp).toLocaleTimeString());
  const timeline = {
    active,
    formatStep: (step: number) => when(frames.find((item) => item.step === Number(step))),
    marks: frames.flatMap((item) => item.event ? [{ step: item.step, label: item.event }] : []),
  };

  const available = Boolean(sources.simulation);
  const status = !connected ? "Disconnected" : available ? SOURCE_LABEL[showing] : "Connected";
  const waiting = !connected ? "Waiting for the server" : !available ? "Waiting for a Market broadcast" :
    subscription ? "Waiting for the next observation." :
    mode === "thesis" ? "Choose a thesis pair and select Subscribe." : "Choose two assets and select Subscribe.";

  return (
    <main className="container market">
      <header className="market-heading">
        <h1>Market MFF</h1>
        <span className={`market-status ${connected && available ? "on" : ""}`}>{status}</span>
      </header>
      {available && (
        <div className="market-toolbar">
          <label className="market-source">
            Source
            <select value={mode} disabled={pending} onChange={(event) => chooseSource(event.target.value as Source)}>
              <option value="simulation">Simulation (A–E)</option>
              <option value="thesis" disabled={!pairs.length}>Thesis return replay</option>
              <option value="replay" disabled={!sources.replay}>Saved-price replay</option>
            </select>
          </label>
          {mode === "thesis" ? (
            <label className="market-source">
              Pair
              <select value={thesisPair} disabled={pending} onChange={(event) => setThesisPair(event.target.value)}>
                <option value="">Choose a pair</option>
                {pairs.map(({ id, symbols }) => <option key={id} value={id}>{symbols.join(" / ")}</option>)}
              </select>
            </label>
          ) : (
            <div className="market-assets" role="group" aria-label="Select two assets">
              {chosenSource?.symbols.map((symbol, index) => (
                <button type="button" key={symbol} disabled={pending} aria-pressed={selection.includes(index)} onClick={() => select(index)}>
                  {symbol}
                </button>
              ))}
            </div>
          )}
          <button type="button" className="market-subscribe" disabled={!canSubscribe} onClick={subscribe}>
            {pending ? "Subscribing…" : unchanged ? "Subscribed" : "Subscribe"}
          </button>
          {subscription && (
            <button type="button" disabled={!connected || pending} onClick={unsubscribe}>Unsubscribe</button>
          )}
          {latestEvent && <small className="market-event">{latestEvent.event} · observation {latestEvent.step.toLocaleString()}</small>}
        </div>
      )}
      {error && <p className="market-error" role="alert">{error}</p>}
      {source && last ? (
        <>
          <dl className="stream-summary" aria-label="Market summary">
            <StreamMetric label="Observations" value={last.observations.toLocaleString()} />
            <StreamMetric label={labels[0]} value={formatNumber(last.x)} colour={COLOURS.x} />
            <StreamMetric label={labels[1]} value={formatNumber(last.y)} colour={COLOURS.y} />
            <StreamMetric label="MFF estimate" value={last.mff != null ? formatNumber(last.mff, 3) :
              `Warm-up ${last.observations}/${source.readyAt}`} colour={COLOURS.mff} />
            {showing === "simulation" && <StreamMetric label="True ρ" value={formatNumber(last.truth, 3)} colour={COLOURS.truth} />}
          </dl>
          <PairCharts rows={frames} timeline={timeline} labels={labels} showTruth={showing === "simulation"}
            title={showing === "thesis" ? "Volatility-standardised log returns" : showing === "replay" ? "Recorded adjusted prices" : "Simulated prices"}
            detail={<time dateTime={last.date ?? (last.timestamp === undefined ? undefined : new Date(last.timestamp).toISOString())}>
              {when(last)}
            </time>}
          />
          <MemoryCharts rows={frames} timeline={timeline} memories={source.memories} />
        </>
      ) : <div className="market-empty" role="status">
        <p>{waiting}</p>
      </div>}
    </main>
  );
}
