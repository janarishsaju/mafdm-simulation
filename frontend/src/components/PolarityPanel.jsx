import React from "react";
import { GitFork, CheckCircle, Loader } from "lucide-react";

const RUN_LABELS = {
  all:      "All leaders (full mix)",
  positive: "Positive leaders only",
  negative: "Negative leaders only",
};

const RUN_COLORS = {
  all:      "#C55A11",
  positive: "#16A34A",
  negative: "#DC2626",
};

function PhaseRow({ run, status }) {
  const done = status === "done";
  return (
    <div style={{
      display: "flex",
      alignItems: "center",
      gap: 10,
      padding: "0.45rem 0.7rem",
      background: done ? "#F0FDF4" : "#F8FAFC",
      border: `1px solid ${done ? "#BBF7D0" : "var(--color-border)"}`,
      borderRadius: 7,
      fontSize: "0.82rem",
    }}>
      <span style={{ color: RUN_COLORS[run], fontSize: 10 }}>&#9632;</span>
      <span style={{ flex: 1, fontWeight: 600, color: "var(--color-text)" }}>
        {RUN_LABELS[run] || run}
      </span>
      {done
        ? <CheckCircle size={14} color="#16A34A" />
        : <Loader size={14} color="#C55A11" style={{ animation: "spin 1s linear infinite" }} />
      }
      <span style={{ fontSize: "0.75rem", color: done ? "#15803D" : "#92400E" }}>
        {done ? "Done" : "Running…"}
      </span>
    </div>
  );
}

function MetricRow({ label, pearson, dtw, color }) {
  const pColor = pearson >= 0.9 ? "#16A34A" : pearson >= 0.75 ? "#D97706" : "#DC2626";
  const dColor = dtw <= 0.5 ? "#16A34A" : dtw <= 1.0 ? "#D97706" : "#DC2626";
  return (
    <tr>
      <td style={{ padding: "6px 10px", display: "flex", alignItems: "center", gap: 7 }}>
        <span style={{ width: 10, height: 10, borderRadius: 2, background: color, display: "inline-block" }} />
        <span style={{ fontSize: "0.82rem", fontWeight: 600, color: "var(--color-text)" }}>{label}</span>
      </td>
      <td style={{ padding: "6px 10px", textAlign: "center" }}>
        <span style={{ fontWeight: 700, color: pColor, fontSize: "0.88rem" }}>
          {pearson >= 0 ? `+${pearson.toFixed(3)}` : pearson.toFixed(3)}
        </span>
      </td>
      <td style={{ padding: "6px 10px", textAlign: "center" }}>
        <span style={{ fontWeight: 700, color: dColor, fontSize: "0.88rem" }}>
          {dtw.toFixed(3)}
        </span>
      </td>
    </tr>
  );
}

const REAL_LEADERS_VARIANTS = new Set(["real_leaders", "real_leaders_networked"]);

export default function PolarityPanel({ status, phases, result, errorMsg, variantName, variantId, datasetName }) {
  const isRealLeaders = REAL_LEADERS_VARIANTS.has(variantId);
  if (status === "idle") return null;

  return (
    <div className="card">
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
        <GitFork size={16} color="#7C3AED" />
        <h3 style={{ color: "#7C3AED" }}>Polarity Analysis</h3>
        {datasetName && (
          <span style={{ fontSize: "0.78rem", color: "var(--color-text-muted)", fontWeight: 500, paddingLeft: 4 }}>
            — {datasetName}
          </span>
        )}
      </div>

      {/* Error */}
      {errorMsg && (
        <div style={{
          background: "#FEF2F2", border: "1px solid #FECACA", borderRadius: 7,
          padding: "0.7rem 0.9rem", color: "#991B1B", fontSize: "0.84rem", marginBottom: 12,
        }}>
          <strong>Error:</strong> {errorMsg}
        </div>
      )}

      {/* Progress phases */}
      {(status === "running" || (status === "done" && !result)) && phases.length > 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: 6, marginBottom: 14 }}>
          {phases.map((p) => (
            <PhaseRow key={p.run} run={p.run} status={p.status} />
          ))}
        </div>
      )}

      {/* Running — show incomplete phases */}
      {status === "running" && (
        <div style={{
          background: "#FAF5FF", border: "1px solid #E9D5FF", borderRadius: 7,
          padding: "0.6rem 0.9rem", fontSize: "0.8rem", color: "#6D28D9",
          display: "flex", alignItems: "center", gap: 8,
        }}>
          <Loader size={13} style={{ animation: "spin 1s linear infinite" }} />
          Running 3 simulations (all / positive / negative leaders)…
        </div>
      )}

      {/* Results */}
      {result && (
        <>
          {/* Method note */}
          <div style={{
            background: "#EFF6FF", border: "1px solid #BFDBFE", borderRadius: 7,
            padding: "0.55rem 0.9rem", fontSize: "0.78rem", color: "#1E40AF",
            marginBottom: 10, lineHeight: 1.6,
          }}>
            {isRealLeaders ? (
              <>
                <strong>Leader signal: actual CSV scores</strong> — leader opinions are the real
                recorded attitude scores from the dataset. The 3 follower curves show how followers
                respond when driven by all leaders, positive leaders only, or negative leaders only.
              </>
            ) : (
              <>
                <strong>Leader signal: {variantName} simulation</strong> — leader opinions evolve
                through the selected algorithm (LLM / CA). Leaders are classified as positive or
                negative by their mean attitude score across the simulation window. The 3 follower
                curves show how followers respond under each leader group.
              </>
            )}
          </div>

          {/* Leader count summary */}
          <div style={{
            background: "#FAF5FF", border: "1px solid #E9D5FF", borderRadius: 7,
            padding: "0.55rem 0.9rem", fontSize: "0.8rem", color: "#5B21B6",
            marginBottom: 14, display: "flex", gap: 20, flexWrap: "wrap",
          }}>
            <span>
              <span style={{ color: "#16A34A", fontWeight: 700 }}>●</span>{" "}
              <span style={{ fontWeight: 700 }}>{result.n_pos}</span> positive leaders (avg score &gt; 0)
            </span>
            <span>
              <span style={{ color: "#DC2626", fontWeight: 700 }}>●</span>{" "}
              <span style={{ fontWeight: 700 }}>{result.n_neg}</span> negative leaders (avg score ≤ 0)
            </span>
            <span style={{ color: "var(--color-text-muted)" }}>
              classified by mean attitude across all simulation-window posts
            </span>
          </div>

          {/* Metrics table */}
          <div style={{ marginBottom: 16, overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.82rem" }}>
              <thead>
                <tr style={{ borderBottom: "2px solid var(--color-border)" }}>
                  <th style={{ padding: "6px 10px", textAlign: "left", color: "var(--color-text-muted)", fontWeight: 700, fontSize: "0.72rem", textTransform: "uppercase" }}>Group</th>
                  <th style={{ padding: "6px 10px", textAlign: "center", color: "var(--color-text-muted)", fontWeight: 700, fontSize: "0.72rem", textTransform: "uppercase" }}>Pearson</th>
                  <th style={{ padding: "6px 10px", textAlign: "center", color: "var(--color-text-muted)", fontWeight: 700, fontSize: "0.72rem", textTransform: "uppercase" }}>DTW</th>
                </tr>
              </thead>
              <tbody>
                <MetricRow label="All leaders (full mix)"   pearson={result.pearson_all} dtw={result.dtw_all} color={RUN_COLORS.all} />
                <MetricRow label={`Positive only (${result.n_pos})`} pearson={result.pearson_pos} dtw={result.dtw_pos} color={RUN_COLORS.positive} />
                <MetricRow label={`Negative only (${result.n_neg})`} pearson={result.pearson_neg} dtw={result.dtw_neg} color={RUN_COLORS.negative} />
              </tbody>
            </table>
          </div>

          {/* Polarity chart */}
          {result.chart_b64 && (
            <div>
              <div style={{ fontSize: "0.8rem", fontWeight: 600, color: "var(--color-text-muted)", marginBottom: 8 }}>
                Polarity comparison chart — {variantName}
              </div>
              <img
                src={`data:image/png;base64,${result.chart_b64}`}
                alt="Polarity comparison chart"
                style={{ width: "100%", borderRadius: 8, border: "1px solid var(--color-border)" }}
              />
            </div>
          )}
        </>
      )}

      <style>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
}
