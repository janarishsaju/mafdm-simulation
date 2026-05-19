import React, { useRef, useEffect } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ReferenceLine, ResponsiveContainer,
} from "recharts";
import { Activity } from "lucide-react";

function ProgressBar({ current, total }) {
  const pct = total > 0 ? Math.round((current / total) * 100) : 0;
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ fontSize: "0.78rem", color: "var(--color-text-muted)" }}>
          Day progress
        </span>
        <span style={{ fontSize: "0.78rem", fontWeight: 600 }}>
          {current} / {total} days ({pct}%)
        </span>
      </div>
      <div
        style={{
          height: 8,
          background: "#E2E8F0",
          borderRadius: 99,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            width: `${pct}%`,
            height: "100%",
            background: "var(--color-primary)",
            borderRadius: 99,
            transition: "width 0.3s ease",
          }}
        />
      </div>
    </div>
  );
}

function LLMStats({ results }) {
  if (results.length === 0) return null;
  const last = results[results.length - 1];
  const total = last.llm_pos + last.llm_neu + last.llm_neg || 1;
  const posP  = Math.round((last.llm_pos / total) * 100);
  const neuP  = Math.round((last.llm_neu / total) * 100);
  const negP  = 100 - posP - neuP;

  return (
    <div
      style={{
        display: "flex",
        gap: 12,
        fontSize: "0.78rem",
        marginTop: 8,
      }}
    >
      <span style={{ color: "#16A34A", fontWeight: 600 }}>
        LLM +1: {last.llm_pos} ({posP}%)
      </span>
      <span style={{ color: "#6B7280", fontWeight: 600 }}>
        LLM 0: {last.llm_neu} ({neuP}%)
      </span>
      <span style={{ color: "#DC2626", fontWeight: 600 }}>
        LLM −1: {last.llm_neg} ({negP}%)
      </span>
      <span style={{ marginLeft: "auto", color: "var(--color-text-muted)" }}>
        ε = {last.epsilon?.toFixed(3)}  vel = {last.velocity?.toFixed(3)}
      </span>
    </div>
  );
}

function DayLog({ results }) {
  const logRef = useRef(null);
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [results]);

  return (
    <div
      ref={logRef}
      className="scrollbar-thin"
      style={{
        maxHeight: 160,
        overflowY: "auto",
        fontFamily: "monospace",
        fontSize: "0.75rem",
        color: "var(--color-text-muted)",
        background: "#F8FAFC",
        border: "1px solid var(--color-border)",
        borderRadius: 6,
        padding: "0.5rem 0.7rem",
        lineHeight: 1.7,
      }}
    >
      {results.map((r) => (
        <div
          key={r.day}
          style={{ color: r.is_anchor ? "var(--color-accent)" : "inherit", fontWeight: r.is_anchor ? 600 : 400 }}
        >
          Day {r.day >= 0 ? `+${r.day}` : r.day}:{" "}
          leader={r.leader_avg?.toFixed(3)}  follower={r.follower_avg?.toFixed(3)}
          {"  "}ε={r.epsilon?.toFixed(3)}
          {r.is_anchor ? "  ◆ ANCHOR" : ""}
        </div>
      ))}
    </div>
  );
}

const CUSTOM_TOOLTIP = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div
      style={{
        background: "white",
        border: "1px solid #E2E8F0",
        borderRadius: 7,
        padding: "8px 12px",
        fontSize: "0.78rem",
        boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
      }}
    >
      <div style={{ fontWeight: 600, marginBottom: 4 }}>
        Day {label >= 0 ? `+${label}` : label}
      </div>
      {payload.map((p) => (
        <div key={p.name} style={{ color: p.color }}>
          {p.name}: {p.value?.toFixed(3)}
        </div>
      ))}
    </div>
  );
};

export default function SimulationPanel({ results, totalDays, status, source, variantName, datasetName }) {
  if (results.length === 0 && status === "idle") return null;

  const chartData = results.map((r) => ({
    day:    r.day,
    Leader: parseFloat(r.leader_avg?.toFixed(3)),
    Follower: parseFloat(r.follower_avg?.toFixed(3)),
  }));

  return (
    <div className="card" style={{ marginBottom: "1rem" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 14, flexWrap: "wrap" }}>
        <Activity size={16} color="var(--color-primary)" style={{ flexShrink: 0 }} />
        <div>
          <h3 style={{ color: "var(--color-primary)", lineHeight: 1.3 }}>
            {variantName || "Simulation Progress"}
          </h3>
          {datasetName && (
            <div style={{ fontSize: "0.75rem", color: "var(--color-text-muted)", fontWeight: 500 }}>
              {datasetName}
            </div>
          )}
        </div>
        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8 }}>
          {source === "cache" && (
            <span style={{
              background: "#F0FDF4",
              border: "1px solid #BBF7D0",
              color: "#15803D",
              fontSize: "0.72rem",
              fontWeight: 700,
              padding: "2px 8px",
              borderRadius: 99,
            }}>
              ⚡ From Cache
            </span>
          )}
          {source === "live" && status !== "done" && (
            <span style={{
              background: "#FEF3C7",
              border: "1px solid #FDE68A",
              color: "#92400E",
              fontSize: "0.72rem",
              fontWeight: 700,
              padding: "2px 8px",
              borderRadius: 99,
            }}>
              Live LLM
            </span>
          )}
          {status === "running" && (
            <span style={{ fontSize: "0.75rem", color: "var(--color-accent)", fontWeight: 600, animation: "pulse 1.5s ease-in-out infinite" }}>
              ● Running…
            </span>
          )}
          {status === "done" && (
            <span style={{ fontSize: "0.75rem", color: "#16A34A", fontWeight: 600 }}>
              ✓ Complete
            </span>
          )}
          {status === "error" && (
            <span style={{ fontSize: "0.75rem", color: "#DC2626", fontWeight: 600 }}>
              ✗ Error
            </span>
          )}
        </div>
      </div>

      <ProgressBar current={results.length} total={totalDays} />
      <LLMStats results={results} />

      {chartData.length > 0 && (
        <div style={{ marginTop: 14, height: 240 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 4, right: 10, left: -15, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
              <XAxis
                dataKey="day"
                tick={{ fontSize: 10, fill: "#64748B" }}
                tickFormatter={(v) => (v >= 0 ? `+${v}` : v)}
              />
              <YAxis
                domain={[-1.1, 1.1]}
                tick={{ fontSize: 10, fill: "#64748B" }}
                ticks={[-1, -0.5, 0, 0.5, 1]}
              />
              <Tooltip content={<CUSTOM_TOOLTIP />} />
              <Legend wrapperStyle={{ fontSize: "0.78rem" }} />
              <ReferenceLine y={0} stroke="#94A3B8" strokeDasharray="4 4" />
              <ReferenceLine x={0} stroke="#DC262644" strokeDasharray="4 4" />
              <Line
                type="monotone"
                dataKey="Leader"
                stroke="var(--color-primary)"
                strokeWidth={1.5}
                dot={false}
                strokeDasharray="4 2"
              />
              <Line
                type="monotone"
                dataKey="Follower"
                stroke="var(--color-accent)"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {results.length > 0 && (
        <div style={{ marginTop: 10 }}>
          <DayLog results={results} />
        </div>
      )}

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50%       { opacity: 0.4; }
        }
      `}</style>
    </div>
  );
}
