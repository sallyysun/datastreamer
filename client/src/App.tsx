import { useState } from "react";

import Lab from "./Lab";
import Market from "./Market";
import "./App.css";

export default function App() {
  const [view, setView] = useState<"lab" | "market">(
    new URLSearchParams(window.location.search).get("view") === "market" ? "market" : "lab",
  );
  const [marketOpened, setMarketOpened] = useState(view === "market");
  return (
    <>
      <header className="app-header">
        <div className="app-brand"><span aria-hidden="true" />Datastreamer</div>
        <nav aria-label="Views" className="view-nav">
          <button className="view-link" aria-current={view === "lab" ? "page" : undefined} onClick={() => setView("lab")}>MFF lab</button>
          <button className="view-link" aria-current={view === "market" ? "page" : undefined}
            onClick={() => { setMarketOpened(true); setView("market"); }}>Market stream</button>
        </nav>
      </header>
      {view === "lab" && <Lab />}
      {marketOpened && <div hidden={view !== "market"}><Market active={view === "market"} /></div>}
    </>
  );
}
