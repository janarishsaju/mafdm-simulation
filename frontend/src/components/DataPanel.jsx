import React, { useState } from "react";
import { Database, ChevronDown, ChevronUp, RotateCcw } from "lucide-react";

export default function DataPanel({
  datasets,
  selectedId,
  onSelect,
  anchorOverrides,
  onAnchorChange,
  disabled,
  variantId,
}) {
  const [showAnchors, setShowAnchors] = useState(true);
  const selected = datasets.find((d) => d.id === selectedId);
  const anchorDays = selected?.anchor_days ?? {};
  const resolutionDay = selected?.resolution_anchor_day ?? null;
  const sortedDays = Object.keys(anchorDays)
    .map(Number)
    .sort((a, b) => a - b);

  const isRealLeaders = variantId === "real_leaders" || variantId === "real_leaders_networked";

  function handleAnchorEdit(day, value) {
    onAnchorChange({ ...anchorOverrides, [String(day)]: value });
  }

  function handleReset(day) {
    const next = { ...anchorOverrides };
    delete next[String(day)];
    onAnchorChange(next);
  }

  function dayLabel(day) {
    if (day === resolutionDay) {
      const sign = day > 0 ? `+${day}` : day === 0 ? "0" : `${day}`;
      return `Day ${sign} — Resolution Anchor`;
    }
    if (day === 0) return "Day 0 — Crisis Anchor";
    const sign = day > 0 ? `+${day}` : `${day}`;
    return `Day ${sign} — Anchor`;
  }

  function anchorPhaseNote(day) {
    if (day === resolutionDay) {
      const next = day + 1;
      return `ε switches to recovery floor (0.50) from Day +${next} onwards`;
    }
    return `ε crisis floor (0.30) active from this day`;
  }

  return (
    <div className="card">
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
        <Database size={16} color="var(--color-primary)" />
        <h3 style={{ color: "var(--color-primary)" }}>Dataset</h3>
      </div>

      {/* Dataset selector */}
      <div className="field">
        <label>Select dataset</label>
        <select
          value={selectedId}
          onChange={(e) => onSelect(e.target.value)}
          disabled={disabled}
        >
          {datasets.map((d) => (
            <option key={d.id} value={d.id}>
              {d.name}
            </option>
          ))}
        </select>
      </div>

      {/* Dataset summary */}
      {selected && (
        <div
          style={{
            background: "#F8FAFC",
            border: "1px solid var(--color-border)",
            borderRadius: 7,
            padding: "0.7rem 0.9rem",
            fontSize: "0.8rem",
            color: "var(--color-text-muted)",
            lineHeight: 1.7,
          }}
        >
          <div>
            <span style={{ fontWeight: 600, color: "var(--color-text)" }}>Event: </span>
            {selected.label}
          </div>
          <div>
            <span style={{ fontWeight: 600, color: "var(--color-text)" }}>Window: </span>
            Day {selected.sim_start} to Day +{selected.sim_end} (
            {selected.sim_end - selected.sim_start + 1} days)
          </div>
        </div>
      )}

      <hr />

      {/* Anchor inputs toggle */}
      <button
        className="btn-outline"
        style={{
          width: "100%",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
        onClick={() => setShowAnchors((s) => !s)}
      >
        <span>Anchor Days ({sortedDays.length})</span>
        {showAnchors ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
      </button>

      {showAnchors && sortedDays.length > 0 && (
        isRealLeaders ? (
          /* Real Leaders: read-only timing display, no blurb editing */
          <div style={{ marginTop: 10 }}>
            <div style={{
              background: "#FFF7ED",
              border: "1px solid #FED7AA",
              borderRadius: 7,
              padding: "0.6rem 0.8rem",
              fontSize: "0.76rem",
              color: "#92400E",
              lineHeight: 1.6,
              marginBottom: 10,
            }}>
              <strong>Blurb not used as input.</strong> Only anchor day numbers
              are used — to time the ε phase switch. No LLM reads these blurbs.
              {variantId === "real_leaders_networked" && (
                <span style={{ display: "block", marginTop: 4 }}>
                  <strong>Connections:</strong> subreddit co-participation (272) · content-theme (71) · theme-fallback (16) · thread-scraped (3).
                </span>
              )}
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {sortedDays.map((day) => (
                <div key={day} style={{
                  background: "#F8FAFC",
                  border: "1px solid var(--color-border)",
                  borderRadius: 7,
                  padding: "0.65rem 0.9rem",
                }}>
                  <div style={{
                    fontSize: "0.78rem",
                    fontWeight: 700,
                    color: day === resolutionDay ? "#217346" : "#C00000",
                    marginBottom: 3,
                  }}>
                    {dayLabel(day)}
                  </div>
                  <div style={{
                    fontSize: "0.74rem",
                    color: "var(--color-text-muted)",
                    fontStyle: "italic",
                  }}>
                    {anchorPhaseNote(day)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        ) : (
          /* All other variants: editable blurb textareas */
          <div style={{ marginTop: 10, display: "flex", flexDirection: "column", gap: 12 }}>
            {sortedDays.map((day) => {
              const defaultText = anchorDays[day] ?? "";
              const override    = anchorOverrides?.[String(day)];
              const current     = override !== undefined ? override : defaultText;
              const isEdited    = override !== undefined && override !== defaultText;

              return (
                <div key={day}>
                  <div style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    marginBottom: 4,
                  }}>
                    <label style={{
                      fontSize: "0.75rem",
                      fontWeight: 700,
                      color: isEdited ? "var(--color-primary)" : "var(--color-text-muted)",
                      textTransform: "uppercase",
                      letterSpacing: "0.05em",
                    }}>
                      {dayLabel(day)}{isEdited ? " ✎" : ""}
                    </label>
                    {isEdited && (
                      <button
                        title="Reset to default"
                        onClick={() => handleReset(day)}
                        disabled={disabled}
                        style={{
                          background: "none",
                          border: "none",
                          cursor: "pointer",
                          color: "var(--color-text-muted)",
                          padding: "2px 4px",
                          display: "flex",
                          alignItems: "center",
                        }}
                      >
                        <RotateCcw size={12} />
                      </button>
                    )}
                  </div>
                  <textarea
                    value={current}
                    onChange={(e) => handleAnchorEdit(day, e.target.value)}
                    disabled={disabled}
                    rows={4}
                    style={{
                      width: "100%",
                      resize: "vertical",
                      fontSize: "0.78rem",
                      lineHeight: 1.55,
                      padding: "0.5rem 0.6rem",
                      borderRadius: 6,
                      border: isEdited
                        ? "1.5px solid var(--color-primary)"
                        : "1px solid var(--color-border)",
                      background: disabled ? "#F8FAFC" : "white",
                      color: "var(--color-text)",
                      fontFamily: "inherit",
                      boxSizing: "border-box",
                    }}
                  />
                </div>
              );
            })}
          </div>
        )
      )}
    </div>
  );
}
