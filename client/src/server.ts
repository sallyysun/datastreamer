export function wsUrl() {
  const configured = import.meta.env.VITE_WS_URL;
  if (configured) return configured;
  if (window.location.hostname === "localhost") return "ws://localhost:9002";
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}`;
}
