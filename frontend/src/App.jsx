import React, { useState, useEffect, useCallback } from "react";
import DataPanel       from "./components/DataPanel.jsx";
import AlgorithmPanel  from "./components/AlgorithmPanel.jsx";
import SimulationPanel from "./components/SimulationPanel.jsx";
import ResultsPanel    from "./components/ResultsPanel.jsx";
import { FlaskConical } from "lucide-react";

const API = "/api";

export default function App() {
  // ── Server data ────────────────────────────────────────────────────────────
  const [datasets, setDatasets] = useState([]);
  const [variants, setVariants] = useState([]);
  const [cachedCombos, setCachedCombos] = useState([]); // [{dataset_id, variant_id}]

  // ── Selections ─────────────────────────────────────────────────────────────
  const [datasetId,       setDatasetId]       = useState("jj_vaccine");
  const [variantId,       setVariantId]       = useState("mafdm_m3b");
  const [config,          setConfig]          = useState(null);
  const [anchorOverrides, setAnchorOverrides] = useState({});   // {day_str: blurb}

  // ── Demo mode ──────────────────────────────────────────────────────────────
  const [liveRunsEnabled, setLiveRunsEnabled] = useState(false);

  // ── Simulation state ───────────────────────────────────────────────────────
  const [status,     setStatus]     = useState("idle");
  const [source,     setSource]     = useState(null);   // "cache" | "live" | null
  const [dayResults, setDayResults] = useState([]);
  const [evaluation, setEvaluation] = useState(null);
  const [errorMsg,   setErrorMsg]   = useState(null);
  const [totalDays,  setTotalDays]  = useState(31);

  // ── Bootstrap ──────────────────────────────────────────────────────────────
  useEffect(() => {
    Promise.all([
      fetch(`${API}/datasets`).then((r) => r.json()),
      fetch(`${API}/variants`).then((r) => r.json()),
      fetch(`${API}/cache`).then((r) => r.json()),
      fetch(`${API}/health`).then((r) => r.json()).catch(() => ({})),
    ]).then(([ds, vs, cache, health]) => {
      if (health?.live_runs_enabled !== undefined) {
        setLiveRunsEnabled(health.live_runs_enabled);
      }
      setDatasets(ds);
      setVariants(vs);
      setCachedCombos(cache);
      if (vs.length > 0) {
        const def = vs.find((v) => v.id === "mafdm_m3b") || vs[0];
        setVariantId(def.id);
        setConfig(def.default_config);
      }
      if (ds.length > 0) {
        const first = ds[0];
        setDatasetId(first.id);
        setTotalDays(first.sim_end - first.sim_start + 1);
      }
    });
  }, []);

  function refreshCache() {
    fetch(`${API}/cache`).then((r) => r.json()).then(setCachedCombos);
  }

  function resetSimState() {
    setStatus("idle");
    setSource(null);
    setDayResults([]);
    setEvaluation(null);
    setErrorMsg(null);
  }

  function handleSelectVariant(id) {
    setVariantId(id);
    const v = variants.find((x) => x.id === id);
    if (v) setConfig(v.default_config);
    resetSimState();
  }

  function handleSelectDataset(id) {
    setDatasetId(id);
    const d = datasets.find((x) => x.id === id);
    if (d) setTotalDays(d.sim_end - d.sim_start + 1);
    setAnchorOverrides({});   // clear edits when switching dataset
    resetSimState();
  }

  // ── Check if current selection is cached ──────────────────────────────────
  const isCached = cachedCombos.some(
    (c) => c.dataset_id === datasetId && c.variant_id === variantId
  );

  // ── Run simulation ─────────────────────────────────────────────────────────
  const handleRun = useCallback(async (forceRerun = false) => {
    setStatus("running");
    setSource(null);
    setDayResults([]);
    setEvaluation(null);
    setErrorMsg(null);

    try {
      const response = await fetch(`${API}/simulate`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({
          dataset_id:       datasetId,
          variant_id:       variantId,
          config,
          force_rerun:      forceRerun,
          anchor_overrides: Object.keys(anchorOverrides).length > 0 ? anchorOverrides : null,
        }),
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${response.status}`);
      }

      const reader  = response.body.getReader();
      const decoder = new TextDecoder();
      let   buffer  = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const raw = line.slice(6).trim();
          if (!raw) continue;

          let event;
          try { event = JSON.parse(raw); } catch { continue; }

          switch (event.type) {
            case "start":
              setSource(event.data?.source ?? "live");
              break;

            case "day":
              setDayResults((prev) => [...prev, event.data]);
              break;

            case "evaluation":
              setEvaluation(event.data);
              setStatus("done");
              refreshCache();
              break;

            case "error":
              setErrorMsg(event.data?.message || "Unknown error");
              setStatus("error");
              break;

            case "done":
              break;

            default:
              break;
          }
        }
      }
    } catch (err) {
      setErrorMsg(err.message);
      setStatus("error");
    }
  }, [datasetId, variantId, config, anchorOverrides]);

  // ── Clear cache & re-run ──────────────────────────────────────────────────
  const handleClearAndRerun = useCallback(async () => {
    await fetch(`${API}/cache/${datasetId}/${variantId}`, { method: "DELETE" });
    refreshCache();
    handleRun(true);
  }, [datasetId, variantId, handleRun]);

  const selectedVariant = variants.find((v) => v.id === variantId);
  const selectedDataset = datasets.find((d) => d.id === datasetId);

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>

      {/* Navbar */}
      <header style={{
        background: "var(--color-primary)",
        color: "white",
        padding: "0.85rem 1.5rem",
        display: "flex",
        alignItems: "center",
        gap: 12,
        boxShadow: "0 2px 8px rgba(0,0,0,0.15)",
      }}>
        <FlaskConical size={22} />
        <div>
          <h1 style={{ color: "white", fontSize: "1.05rem", fontWeight: 700 }}>
            MA-FDE-LLM Simulation Platform
          </h1>
          <p style={{ fontSize: "0.72rem", opacity: 0.75, marginTop: 1 }}>
            Memory-Augmented Follower-Defection-Equilibrium LLM · Phase 9
          </p>
        </div>
      </header>

      {/* Main layout */}
      <div style={{
        flex: 1,
        display: "flex",
        gap: "1rem",
        padding: "1rem",
        alignItems: "flex-start",
        maxWidth: 1400,
        margin: "0 auto",
        width: "100%",
      }}>

        {/* Left sidebar */}
        <div style={{
          width: 300,
          minWidth: 280,
          display: "flex",
          flexDirection: "column",
          gap: "1rem",
          position: "sticky",
          top: "1rem",
        }}>
          <DataPanel
            datasets        = {datasets}
            selectedId      = {datasetId}
            onSelect        = {handleSelectDataset}
            anchorOverrides = {anchorOverrides}
            onAnchorChange  = {setAnchorOverrides}
            disabled        = {status === "running"}
            variantId       = {variantId}
          />
          <AlgorithmPanel
            variants         = {variants}
            selectedId       = {variantId}
            onSelectVariant  = {handleSelectVariant}
            config           = {config}
            onConfigChange   = {setConfig}
            onRun            = {() => handleRun(false)}
            onClearAndRerun  = {handleClearAndRerun}
            running          = {status === "running"}
            disabled         = {status === "running"}
            isCached         = {isCached}
            liveRunsEnabled  = {liveRunsEnabled}
          />
        </div>

        {/* Main area */}
        <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: "1rem" }}>

          {errorMsg && (
            <div style={{
              background: "#FEF2F2",
              border: "1px solid #FECACA",
              borderRadius: 8,
              padding: "0.8rem 1rem",
              color: "#991B1B",
              fontSize: "0.85rem",
            }}>
              <strong>Simulation error:</strong> {errorMsg}
            </div>
          )}

          {status === "idle" && dayResults.length === 0 && (
            <div className="card" style={{ textAlign: "center", padding: "3rem 2rem" }}>
              <FlaskConical size={40} color="#CBD5E1" style={{ margin: "0 auto 12px" }} />
              <h2 style={{ color: "var(--color-text-muted)", fontWeight: 500 }}>
                Ready to simulate
              </h2>
              <p style={{ color: "var(--color-text-muted)", marginTop: 8, fontSize: "0.85rem" }}>
                Select a dataset and algorithm variant on the left, then click{" "}
                <strong>Run Simulation</strong>.
                {isCached && (
                  <span style={{ color: "#16A34A", display: "block", marginTop: 6, fontWeight: 600 }}>
                    Results are cached — will load instantly.
                  </span>
                )}
              </p>
            </div>
          )}

          <SimulationPanel
            results     = {dayResults}
            totalDays   = {totalDays}
            status      = {status}
            source      = {source}
            variantName = {selectedVariant?.name || variantId}
            datasetName = {selectedDataset?.name || datasetId}
          />

          <ResultsPanel
            evaluation   = {evaluation}
            variantName  = {selectedVariant?.name || variantId}
            datasetName  = {selectedDataset?.name || datasetId}
          />

        </div>
      </div>
    </div>
  );
}
