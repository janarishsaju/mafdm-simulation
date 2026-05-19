import React, { useState } from "react";
import { Settings, ChevronDown, ChevronUp, Play } from "lucide-react";

const CONFIG_GROUPS = [
  {
    label: "CA / Opinion Dynamics",
    fields: [
      { key: "R",     label: "R — Self-persistence",          min: 0,   max: 1,  step: 0.01 },
      { key: "W",     label: "W — Neighbour influence",       min: 0,   max: 2,  step: 0.01 },
      { key: "ALPHA", label: "α — CA/LLM blend (CA weight)", min: 0,   max: 1,  step: 0.01 },
    ],
  },
  {
    label: "SIR Dampening",
    fields: [
      { key: "GAMMA", label: "γ — Dampening probability", min: 0,   max: 1,  step: 0.01 },
      { key: "LAM",   label: "λ — Brake strength",        min: 0,   max: 5,  step: 0.05 },
    ],
  },
  {
    label: "Dynamic ε (Interaction Threshold)",
    fields: [
      { key: "EPS_INIT",         label: "ε_init — Starting ε (Day −5)",       min: 0, max: 2,  step: 0.05 },
      { key: "EPS_MIN_CRISIS",   label: "ε_floor_crisis — Crisis floor",       min: 0, max: 1,  step: 0.05 },
      { key: "EPS_MIN_RECOVERY", label: "ε_floor_recovery — Recovery floor",   min: 0, max: 1,  step: 0.05 },
      { key: "EPS_MAX",          label: "ε_max — Ceiling",                     min: 0.5, max: 2, step: 0.05 },
      { key: "EPS_BETA",         label: "β — Velocity sensitivity",            min: 0.1, max: 10, step: 0.1 },
      { key: "EPS_FIXED",        label: "ε_fixed — Fixed ε (non-dynamic variants)", min: 0, max: 2, step: 0.05 },
    ],
  },
  {
    label: "Episodic Memory",
    fields: [
      { key: "MEMORY_THRESHOLD", label: "Threshold — Min shift to record",   min: 0,  max: 1,  step: 0.01 },
      { key: "MEMORY_ROLL_K",    label: "Roll-K — Rolling buffer capacity",   min: 1,  max: 20, step: 1, integer: true },
      { key: "MEMORY_PROMPT_N",  label: "Prompt-N — Entries shown to LLM",    min: 1,  max: 10, step: 1, integer: true },
    ],
  },
  {
    label: "LLM",
    fields: [
      { key: "LLM_TEMPERATURE", label: "Temperature",           min: 0,   max: 2,  step: 0.1 },
      { key: "LLM_MAX_WORKERS", label: "Parallel workers",      min: 1,   max: 50, step: 1, integer: true },
    ],
  },
];

function VariantBadges({ variant }) {
  if (!variant) return null;
  return (
    <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 6 }}>
      <span className={`badge ${variant.has_memory ? "badge-memory" : "badge-nomemory"}`}>
        {variant.has_memory ? "Memory" : "No Memory"}
      </span>
      <span className={`badge ${variant.has_dynamic_eps ? "badge-dynamic" : "badge-fixed"}`}>
        {variant.has_dynamic_eps ? "Dynamic ε" : "Fixed ε"}
      </span>
    </div>
  );
}

export default function AlgorithmPanel({
  variants,
  selectedId,
  onSelectVariant,
  config,
  onConfigChange,
  onRun,
  onClearAndRerun,
  running,
  disabled,
  isCached,
  liveRunsEnabled,
}) {
  const [showConfig, setShowConfig] = useState(false);
  const selected = variants.find((v) => v.id === selectedId);

  function handleField(key, value) {
    onConfigChange({ ...config, [key]: value });
  }

  return (
    <div className="card">
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
        <Settings size={16} color="var(--color-primary)" />
        <h3 style={{ color: "var(--color-primary)" }}>Algorithm</h3>
      </div>

      {/* Variant selector */}
      <div className="field">
        <label>Select variant</label>
        <select
          value={selectedId}
          onChange={(e) => onSelectVariant(e.target.value)}
          disabled={disabled}
        >
          {variants.map((v) => (
            <option key={v.id} value={v.id}>
              {v.name}
            </option>
          ))}
        </select>
      </div>

      {selected && (
        <>
          <VariantBadges variant={selected} />
          <p
            style={{
              marginTop: 8,
              fontSize: "0.78rem",
              color: "var(--color-text-muted)",
              lineHeight: 1.6,
            }}
          >
            {selected.description}
          </p>
        </>
      )}

      <hr />

      {/* Config toggle */}
      <button
        className="btn-outline"
        style={{ width: "100%", display: "flex", justifyContent: "space-between", alignItems: "center" }}
        onClick={() => setShowConfig((s) => !s)}
      >
        <span>Edit Parameters</span>
        {showConfig ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
      </button>

      {showConfig && config && (
        <div style={{ marginTop: 12 }}>
          {CONFIG_GROUPS.map((group) => (
            <div key={group.label} style={{ marginBottom: 14 }}>
              <div
                style={{
                  fontSize: "0.72rem",
                  fontWeight: 700,
                  textTransform: "uppercase",
                  letterSpacing: "0.06em",
                  color: "var(--color-text-muted)",
                  marginBottom: 6,
                }}
              >
                {group.label}
              </div>
              {group.fields.map(({ key, label, min, max, step, integer }) => (
                <div className="field" key={key}>
                  <label>{label}</label>
                  <input
                    type="number"
                    min={min}
                    max={max}
                    step={step}
                    value={config[key] ?? ""}
                    onChange={(e) => {
                      const raw = e.target.value;
                      const val = integer ? parseInt(raw, 10) : parseFloat(raw);
                      if (!isNaN(val)) handleField(key, val);
                    }}
                    disabled={disabled}
                  />
                </div>
              ))}
            </div>
          ))}
        </div>
      )}

      <hr />

      {/* Cache status */}
      {isCached && !running && (
        <div style={{
          background: "#F0FDF4",
          border: "1px solid #BBF7D0",
          borderRadius: 7,
          padding: "0.5rem 0.75rem",
          fontSize: "0.78rem",
          color: "#15803D",
          fontWeight: 600,
          display: "flex",
          alignItems: "center",
          gap: 6,
          marginBottom: 4,
        }}>
          <span>&#9889;</span> Results cached — runs instantly
        </div>
      )}

      {/* Run button */}
      <button
        className="btn-primary"
        style={{ width: "100%", display: "flex", alignItems: "center", justifyContent: "center", gap: 8 }}
        onClick={onRun}
        disabled={disabled || running}
      >
        <Play size={15} />
        {running
          ? "Running simulation…"
          : isCached
          ? "Load from Cache"
          : "Run Simulation"}
      </button>

      {/* Re-run Fresh button — only shown when cache exists AND live runs enabled */}
      {isCached && !running && liveRunsEnabled && (
        <button
          className="btn-outline"
          style={{ width: "100%", marginTop: 6, display: "flex", alignItems: "center", justifyContent: "center", gap: 8 }}
          onClick={onClearAndRerun}
          disabled={disabled || running}
        >
          <Play size={13} />
          Re-run Fresh (calls OpenAI)
        </button>
      )}

      {/* Demo notice — shown when live runs are disabled */}
      {!liveRunsEnabled && (
        <div style={{
          marginTop: 8,
          background: "#FEF9C3",
          border: "1px solid #FDE68A",
          borderRadius: 7,
          padding: "0.5rem 0.75rem",
          fontSize: "0.75rem",
          color: "#92400E",
          lineHeight: 1.55,
        }}>
          <strong>Demo mode</strong> — pre-cached results load instantly. Live LLM runs require an API key.
        </div>
      )}
    </div>
  );
}
