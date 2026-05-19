"""
backend/evaluator.py
====================
Post-simulation evaluation and chart generation.

Computes:
  - Pearson correlation between smoothed simulated follower curve and real curve
  - DTW distance (custom implementation matching phase_8 original)
  - Min simulated value (whether neutral crossing occurred)
  - A comparison chart returned as a base64-encoded PNG string

The chart and metrics are returned together in an EvaluationResult so the API
can send them to the frontend in a single response.
"""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr

from backend.simulation_runner import DayResult


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    pearson:         float
    p_value:         float
    dtw:             float
    min_simulated:   float
    crosses_neutral: bool
    eps_min:         float
    eps_max:         float
    chart_b64:       str        # base64-encoded PNG
    days:            List[int]
    follower_smoothed: List[float]
    real_smoothed:     List[float]

    def to_dict(self) -> dict:
        return {
            "pearson":          round(self.pearson,       4),
            "p_value":          round(self.p_value,       4),
            "dtw":              round(self.dtw,           4),
            "min_simulated":    round(self.min_simulated, 4),
            "crosses_neutral":  self.crosses_neutral,
            "eps_min":          round(self.eps_min,       4),
            "eps_max":          round(self.eps_max,       4),
            "chart_b64":        self.chart_b64,
            "days":             self.days,
            "follower_smoothed": [round(v, 4) for v in self.follower_smoothed],
            "real_smoothed":     [round(v, 4) for v in self.real_smoothed],
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate(
    results:        List[DayResult],
    real_curve:     Dict[int, float],
    variant_name:   str,
    event_label:    str,
    anchor_days:    Optional[List[int]] = None,
) -> EvaluationResult:
    """
    Compute metrics and build the comparison chart.

    Args:
        results:      list of DayResult from SimulationRunner.run()
        real_curve:   dict day → smoothed real follower average (from LoadedDataset)
        variant_name: human-readable variant label for the chart legend
        event_label:  e.g. "Day 0 = April 13, 2021"
        anchor_days:  list of anchor day integers to mark on the chart

    Returns:
        EvaluationResult with metrics + base64 chart PNG.
    """
    days = [r.day for r in results]
    days_arr = np.array(days, dtype=float)

    follower_raw = np.array([r.follower_avg for r in results])
    follower_smoothed = gaussian_filter1d(follower_raw, sigma=1.5)

    real_arr = np.array([real_curve.get(d, float("nan")) for d in days])

    # Pearson + DTW
    corr, p_val = pearsonr(follower_smoothed, real_arr)
    dtw_dist = _dtw_distance(follower_smoothed, real_arr)
    min_sim  = float(np.min(follower_smoothed))

    # ε trajectory
    eps_vals = [r.epsilon for r in results]
    eps_min_val = float(min(eps_vals))
    eps_max_val = float(max(eps_vals))

    # Chart
    chart_b64 = _build_chart(
        days_arr          = days_arr,
        follower_smoothed = follower_smoothed,
        real_smoothed     = real_arr,
        eps_arr           = np.array(eps_vals),
        results           = results,
        variant_name      = variant_name,
        event_label       = event_label,
        pearson           = float(corr),
        dtw               = dtw_dist,
        anchor_days       = anchor_days or [],
    )

    return EvaluationResult(
        pearson           = float(corr),
        p_value           = float(p_val),
        dtw               = dtw_dist,
        min_simulated     = min_sim,
        crosses_neutral   = min_sim < 0.0,
        eps_min           = eps_min_val,
        eps_max           = eps_max_val,
        chart_b64         = chart_b64,
        days              = days,
        follower_smoothed = follower_smoothed.tolist(),
        real_smoothed     = real_arr.tolist(),
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _dtw_distance(s: np.ndarray, t: np.ndarray) -> float:
    """DTW distance — exact implementation from phase_8/simulate_jj_e21_m3b.py."""
    n, m = len(s), len(t)
    d = np.full((n + 1, m + 1), np.inf)
    d[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = (s[i - 1] - t[j - 1]) ** 2 + min(
                d[i - 1, j], d[i, j - 1], d[i - 1, j - 1]
            )
    return math.sqrt(d[n, m])


def _build_chart(
    days_arr:          np.ndarray,
    follower_smoothed: np.ndarray,
    real_smoothed:     np.ndarray,
    eps_arr:           np.ndarray,
    results:           List[DayResult],
    variant_name:      str,
    event_label:       str,
    pearson:           float,
    dtw:               float,
    anchor_days:       List[int],
) -> str:
    """Build the 2-panel comparison chart and return it as a base64 PNG string."""
    sim_start = int(days_arr[0])
    sim_end   = int(days_arr[-1])

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(14, 9),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.patch.set_facecolor("#FFFFFF")
    for a in (ax, ax2):
        a.set_facecolor("#FFFFFF")

    # Phase bands — derived from anchor days
    _draw_phase_bands(ax, ax2, anchor_days, sim_start, sim_end)

    # Reference lines
    ax.axhline(0, color="#888888", linewidth=0.9, zorder=1)
    for d in anchor_days:
        ax.axvline(d, color="#C00000" if d == anchor_days[0] else "#217346",
                   linewidth=1.8, linestyle="--", alpha=0.9, zorder=5)
        ax2.axvline(d, color="#C00000" if d == anchor_days[0] else "#217346",
                    linewidth=1.2, linestyle="--", alpha=0.7, zorder=5)

    # Real curve (ground truth)
    ax.plot(days_arr, real_smoothed, color="#C00000", linewidth=2.5, zorder=4,
            label="Real follower curve (ground truth)")

    # Simulated follower curve
    ax.plot(days_arr, follower_smoothed, color="#C55A11", linewidth=2.8, zorder=5,
            label=f"{variant_name}  (Pearson={pearson:+.3f}, DTW={dtw:.3f})")
    ax.fill_between(days_arr, real_smoothed, follower_smoothed,
                    alpha=0.07, color="#C55A11", zorder=2)

    # Leader curve (dotted, lighter)
    leader_raw = np.array([r.leader_avg for r in results])
    ax.plot(days_arr, leader_raw, color="#1F3964", linewidth=1.2,
            linestyle=":", alpha=0.6, zorder=3, label="Leader avg")

    # Metrics box
    ax.text(
        0.98, 0.05,
        f"Pearson = {pearson:+.3f}\nDTW     = {dtw:.3f}\nMin sim = {float(np.min(follower_smoothed)):+.3f}",
        transform=ax.transAxes, fontsize=9, va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F0F0F0",
                  edgecolor="#CCCCCC", alpha=0.92),
        color="#1F3964", fontweight="bold",
    )

    ax.set_ylim(-1.15, 1.15)
    ax.set_xlim(sim_start, sim_end)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_yticklabels(
        ["−1.0\n(Very Neg)", "−0.5", "0.0\n(Neutral)", "+0.5", "+1.0\n(Very Pos)"],
        fontsize=8, color="#444",
    )
    ax.set_ylabel("Follower Attitude", fontsize=9, color="#333")
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax.tick_params(axis="x", labelsize=8.5, colors="#555")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.grid(axis="y", alpha=0.2, linewidth=0.6)
    ax.legend(fontsize=8.5, framealpha=0.9, edgecolor="#CCCCCC", loc="lower left")
    ax.set_title(
        f"MA-FDE-LLM Simulation — {variant_name}\n{event_label}",
        fontsize=9, fontweight="bold", color="#1F3964", pad=6, loc="left",
    )

    # Lower panel: ε(t)
    ax2.plot(days_arr, eps_arr, color="#C55A11", linewidth=2.0, zorder=4,
             marker="o", markersize=3.5, label="ε(t)")
    ax2.set_ylabel("ε(t)", fontsize=9, color="#333")
    ax2.set_xlabel(f"Days Relative to Crisis Event  ({event_label})",
                   fontsize=9, color="#333")
    ax2.set_xlim(sim_start, sim_end)
    ax2.set_ylim(0.0, float(np.max(eps_arr)) + 0.15)
    ax2.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax2.tick_params(axis="x", labelsize=8.5, colors="#555")
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)
    ax2.spines["left"].set_color("#CCCCCC")
    ax2.spines["bottom"].set_color("#CCCCCC")
    ax2.grid(axis="y", alpha=0.2, linewidth=0.6)
    ax2.legend(fontsize=7.5, framealpha=0.9, edgecolor="#CCCCCC", loc="upper right")

    plt.tight_layout(h_pad=0.5)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _draw_phase_bands(
    ax, ax2,
    anchor_days: List[int],
    sim_start: int,
    sim_end: int,
) -> None:
    """Draw coloured background bands based on anchor day boundaries."""
    if not anchor_days:
        return

    sorted_anchors = sorted(anchor_days)
    boundaries = [sim_start] + sorted_anchors + [sim_end]
    colors = ["#E8F4E8", "#FDECEA", "#E8F0FE", "#FFF8E1"]
    labels = ["Pre-Event"] + [f"Anchor Day {d}" for d in sorted_anchors] + [""]

    for i in range(len(boundaries) - 1):
        d0, d1 = boundaries[i], boundaries[i + 1]
        col = colors[i % len(colors)]
        for a in (ax, ax2):
            a.axvspan(d0, d1, color=col, alpha=0.45, zorder=0)
        if i < len(labels):
            ax.text(
                (d0 + d1) / 2, 0.97, labels[i],
                transform=ax.get_xaxis_transform(),
                fontsize=7.5, color="#444", ha="center", va="top",
                fontweight="bold", style="italic",
            )
