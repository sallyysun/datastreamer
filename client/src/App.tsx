import { useState } from "react";

import Lab from "./Lab";
import Market from "./Market";
import "./App.css";

// Two views of the same server: the MFF lab (a private stream you steer, and a consumer running the MFF you build)
// and the market view (the shared live stream's prices). The browser only displays; estimation runs in consumers.
export default function App() {
  const [view, setView] = useState<"lab" | "market">(
    new URLSearchParams(window.location.search).get("view") === "market" ? "market" : "lab",
  );
  return (
    <>
      <nav className="tabs">
        <button className={view === "lab" ? "tab on" : "tab"} onClick={() => setView("lab")}>MFF lab</button>
        <button className={view === "market" ? "tab on" : "tab"} onClick={() => setView("market")}>Market stream</button>
      </nav>
      {view === "lab" ? <Lab /> : <Market />}
    </>
  );
}
