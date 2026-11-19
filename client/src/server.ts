export function wsUrl() {
  const configured = import.meta.env.VITE_WS_URL;
  if (configured) return configured;
  if (window.location.hostname === "localhost") return "ws://localhost:9002";
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}`;
}

export function replayUrl(id: string, pair?: [string, string]) {
  const url = new URL(wsUrl());
  url.protocol = url.protocol === "wss:" ? "https:" : "http:";
  url.pathname = pair
    ? `/api/market/stocks/${pair.map(encodeURIComponent).join("/")}`
    : `/api/market/replays/${encodeURIComponent(id)}`;
  url.search = "";
  return url;
}
