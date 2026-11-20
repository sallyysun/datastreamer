import { useState } from "react";

import Lab from "./Lab";
import Market from "./Market";
import "./App.css";

export default function App() {
  const [view, setView] = useState<"lab" | "market">(
    new URLSearchParams(window.location.search).get("view") === "lab" ? "lab" : "market",
  );
  return (
    <>
      <header className="app-header">
        <div className="app-brand"><span aria-hidden="true" />Datastreamer</div>
        <nav aria-label="Views" className="view-nav">
          <button className="view-link" aria-current={view === "market" ? "page" : undefined}
            onClick={() => setView("market")}>Market stream</button>
          <button className="view-link" aria-current={view === "lab" ? "page" : undefined} onClick={() => setView("lab")}>MFF lab</button>
        </nav>
      </header>
      {view === "lab" && <Lab />}
      <div hidden={view !== "market"}><Market active={view === "market"} /></div>
    </>
  );
}
