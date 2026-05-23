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

from backend.simulation_runner import DayResult


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PolarityResult:
    """Metrics and chart for a 3-run polarity comparison."""
    pearson_all:  float
    pearson_pos:  float
    pearson_neg:  float
    dtw_all:      float
    dtw_pos:      float
    dtw_neg:      float
    n_pos:        int     # number of positive-classified leaders
    n_neg:        int     # number of negative-classified leaders
    chart_b64:    str

    def to_dict(self) -> dict:
        return {
            "pearson_all": round(self.pearson_all, 4),
            "pearson_pos": round(self.pearson_pos, 4),
            "pearson_neg": round(self.pearson_neg, 4),
            "dtw_all":     round(self.dtw_all, 4),
            "dtw_pos":     round(self.dtw_pos, 4),
            "dtw_neg":     round(self.dtw_neg, 4),
            "n_pos":       self.n_pos,
            "n_neg":       self.n_neg,
            "chart_b64":   self.chart_b64,
        }


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
    variant_id:     str = "",
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
    follower_smoothed = _gaussian_filter1d(follower_raw, sigma=1.5)

    real_arr = np.array([real_curve.get(d, float("nan")) for d in days])

    # Pearson + DTW
    corr, p_val = _pearsonr(follower_smoothed, real_arr)
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
        real_leaders_mode = variant_id in ("real_leaders", "real_leaders_networked"),
        variant_id        = variant_id,
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


def evaluate_polarity(
    results_all:    List[DayResult],
    results_pos:    List[DayResult],
    results_neg:    List[DayResult],
    real_curve:     Dict[int, float],
    variant_name:   str,
    variant_id:     str,
    event_label:    str,
    anchor_days:    Optional[List[int]],
    polarity_counts: Dict[str, int],     # {"positive": N, "negative": M}
) -> PolarityResult:
    """Build a 4-curve polarity comparison chart and compute per-group metrics."""
    days     = [r.day for r in results_all]
    days_arr = np.array(days, dtype=float)
    real_arr = np.array([real_curve.get(d, float("nan")) for d in days])

    sm_all = _gaussian_filter1d(np.array([r.follower_avg for r in results_all]), sigma=1.5)
    sm_pos = _gaussian_filter1d(np.array([r.follower_avg for r in results_pos]), sigma=1.5)
    sm_neg = _gaussian_filter1d(np.array([r.follower_avg for r in results_neg]), sigma=1.5)

    r_all, _ = _pearsonr(sm_all, real_arr)
    r_pos, _ = _pearsonr(sm_pos, real_arr)
    r_neg, _ = _pearsonr(sm_neg, real_arr)

    d_all = _dtw_distance(sm_all, real_arr)
    d_pos = _dtw_distance(sm_pos, real_arr)
    d_neg = _dtw_distance(sm_neg, real_arr)

    chart_b64 = _build_polarity_chart(
        days_arr     = days_arr,
        sm_all       = sm_all,
        sm_pos       = sm_pos,
        sm_neg       = sm_neg,
        real_arr     = real_arr,
        r_all=float(r_all), r_pos=float(r_pos), r_neg=float(r_neg),
        d_all=d_all,        d_pos=d_pos,        d_neg=d_neg,
        variant_name = variant_name,
        variant_id   = variant_id,
        event_label  = event_label,
        anchor_days  = anchor_days or [],
        n_pos        = polarity_counts.get("positive", 0),
        n_neg        = polarity_counts.get("negative", 0),
    )

    return PolarityResult(
        pearson_all = float(r_all),
        pearson_pos = float(r_pos),
        pearson_neg = float(r_neg),
        dtw_all     = d_all,
        dtw_pos     = d_pos,
        dtw_neg     = d_neg,
        n_pos       = polarity_counts.get("positive", 0),
        n_neg       = polarity_counts.get("negative", 0),
        chart_b64   = chart_b64,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _gaussian_filter1d(x: np.ndarray, sigma: float) -> np.ndarray:
    """Pure numpy 1-D Gaussian smoothing (replaces scipy.ndimage.gaussian_filter1d)."""
    radius = int(4 * sigma + 0.5)
    k = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (k / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(x, radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _pearsonr(x: np.ndarray, y: np.ndarray):
    """Pearson r and approximate two-tailed p-value (replaces scipy.stats.pearsonr)."""
    xd = x - x.mean()
    yd = y - y.mean()
    denom = math.sqrt(float(np.dot(xd, xd) * np.dot(yd, yd)))
    r = float(np.dot(xd, yd) / denom) if denom > 0 else 0.0
    r = max(-1.0, min(1.0, r))
    n = len(x)
    if abs(r) >= 1.0 or n <= 2:
        return r, 0.0
    t_stat = r * math.sqrt((n - 2) / (1.0 - r ** 2))
    # Two-tailed p-value approximation via complementary error function
    p = math.erfc(abs(t_stat) / math.sqrt(2.0))
    return r, float(p)


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
    real_leaders_mode: bool = False,
    variant_id:        str  = "",
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

    # Leader curve — input signal for followers
    leader_raw = np.array([r.leader_avg for r in results])
    if real_leaders_mode:
        # Real Leaders: leader line is the actual data input — make it prominent
        ax.plot(days_arr, leader_raw, color="#1F3964", linewidth=2.2,
                linestyle="-", alpha=0.85, zorder=3, label="Real leader scores (CSV input)")
    else:
        # Other variants: leader is a simulated output — show lightly
        ax.plot(days_arr, leader_raw, color="#1F3964", linewidth=1.2,
                linestyle=":", alpha=0.55, zorder=3, label="Simulated leader avg")

    # Real curve (ground truth)
    ax.plot(days_arr, real_smoothed, color="#C00000", linewidth=2.5, zorder=4,
            label="Real follower curve (ground truth)")

    # Simulated follower curve
    follower_label = (
        "Simulated follower curve  "
        f"(Pearson={pearson:+.3f}, DTW={dtw:.3f})"
        if real_leaders_mode else
        f"{variant_name}  (Pearson={pearson:+.3f}, DTW={dtw:.3f})"
    )
    ax.plot(days_arr, follower_smoothed, color="#C55A11", linewidth=2.8, zorder=5,
            label=follower_label)
    ax.fill_between(days_arr, real_smoothed, follower_smoothed,
                    alpha=0.07, color="#C55A11", zorder=2)

    # Metrics box
    ax.text(
        0.98, 0.05,
        f"Pearson = {pearson:+.3f}\nDTW     = {dtw:.3f}\nMin sim = {float(np.min(follower_smoothed)):+.3f}",
        transform=ax.transAxes, fontsize=9, va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F0F0F0",
                  edgecolor="#CCCCCC", alpha=0.92),
        color="#1F3964", fontweight="bold",
    )

    # Help text — Real Leaders variants only
    if real_leaders_mode:
        if variant_id == "real_leaders_networked":
            help_text = (
                "How to read:\n"
                "Blue  = Real leader scores (group daily avg, interpolated from CSV)\n"
                "Orange = Simulated follower curve — CA+SIR with actual derived connections\n"
                "Red   = Observed real follower curve (ground truth)\n"
                "● Connections: subreddit co-participation (272) · content-theme (71)\n"
                "  theme-fallback (16) · Arctic Shift thread-scraped (3)\n"
                "● No LLM calls · No memory diary"
            )
        else:
            help_text = (
                "How to read:\n"
                "Blue  = Real leader scores (group daily avg, interpolated from CSV)\n"
                "Orange = Simulated follower curve — CA+SIR dynamics driven by real leader scores\n"
                "Red   = Observed real follower curve (ground truth)\n"
                "● Follower connections: random (3 leaders + 2 followers per agent)\n"
                "● No LLM calls · No memory diary"
            )
        ax.text(
            0.02, 0.97,
            help_text,
            transform=ax.transAxes,
            fontsize=7.2, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#EFF6FF",
                      edgecolor="#BFDBFE", alpha=0.93),
            color="#1E3A5F",
        )

        # Annotate the Days 16-18 leader dip (social cascade effect)
        day17_idx = int(np.argmin(np.abs(days_arr - 17)))
        if 0 <= day17_idx < len(leader_raw):
            y17 = float(leader_raw[day17_idx])
            ax.annotate(
                "Days 16–18: Social cascade\n"
                "Post-pause coverage by journalists &\n"
                "pharmacists sustains J&J hesitancy.\n"
                "Market share fell 20%→9% of daily doses.",
                xy=(float(days_arr[day17_idx]), y17),
                xytext=(11.0, -0.80),
                fontsize=6.8,
                color="#4A235A",
                bbox=dict(
                    boxstyle="round,pad=0.38",
                    facecolor="#FEF9FF",
                    edgecolor="#D7BDE2",
                    alpha=0.93,
                ),
                arrowprops=dict(
                    arrowstyle="->",
                    color="#8E44AD",
                    lw=1.3,
                    connectionstyle="arc3,rad=-0.2",
                ),
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


def _build_polarity_chart(
    days_arr:    np.ndarray,
    sm_all:      np.ndarray,
    sm_pos:      np.ndarray,
    sm_neg:      np.ndarray,
    real_arr:    np.ndarray,
    r_all: float, r_pos: float, r_neg: float,
    d_all: float, d_pos: float, d_neg: float,
    variant_name: str,
    variant_id:   str,
    event_label:  str,
    anchor_days:  List[int],
    n_pos: int,
    n_neg: int,
) -> str:
    """Two-panel polarity chart (follower curves + leader distribution bar) as base64 PNG."""
    _REAL_LEADERS_VARIANTS = {"real_leaders", "real_leaders_networked"}
    is_real_leaders = variant_id in _REAL_LEADERS_VARIANTS
    leader_signal_note = (
        "Leader signal: actual CSV scores (real recorded attitudes)"
        if is_real_leaders else
        f"Leader signal: {variant_name} simulation (LLM/CA driven)"
    )

    sim_start = int(days_arr[0])
    sim_end   = int(days_arr[-1])

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    if anchor_days:
        sorted_anchors = sorted(anchor_days)
        boundaries = [sim_start] + sorted_anchors + [sim_end]
        band_colors = ["#E8F4E8", "#FDECEA", "#E8F0FE", "#FFF8E1"]
        for i in range(len(boundaries) - 1):
            ax.axvspan(boundaries[i], boundaries[i + 1],
                       color=band_colors[i % len(band_colors)], alpha=0.35, zorder=0)
        for d in sorted_anchors:
            ax.axvline(d, color="#C00000" if d == sorted_anchors[0] else "#217346",
                       linewidth=1.5, linestyle="--", alpha=0.8, zorder=5)

    ax.axhline(0, color="#888888", linewidth=0.9, zorder=1)

    ax.plot(days_arr, real_arr, color="#1F3964", linewidth=2.5, zorder=6,
            label="Real observed follower curve (ground truth)")
    ax.plot(days_arr, sm_all, color="#C55A11", linewidth=2.2, linestyle="--", zorder=4,
            label=f"Simulated followers — all leaders (full mix)  r={r_all:+.3f}, DTW={d_all:.3f}")
    ax.plot(days_arr, sm_pos, color="#16A34A", linewidth=2.5, zorder=5,
            label=f"Simulated followers — positive leaders only ({n_pos})  r={r_pos:+.3f}, DTW={d_pos:.3f}")
    ax.plot(days_arr, sm_neg, color="#DC2626", linewidth=2.5, zorder=5,
            label=f"Simulated followers — negative leaders only ({n_neg})  r={r_neg:+.3f}, DTW={d_neg:.3f}")

    ax.set_ylim(-1.15, 1.15)
    ax.set_xlim(sim_start, sim_end)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_yticklabels(["−1.0\n(Very Neg)", "−0.5", "0.0\n(Neutral)", "+0.5", "+1.0\n(Very Pos)"],
                       fontsize=8, color="#444")
    ax.set_ylabel("Simulated Follower Attitude", fontsize=9, color="#333")
    ax.set_xlabel(f"Days Relative to Crisis Event  ({event_label})", fontsize=9, color="#333")
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax.tick_params(axis="x", labelsize=8.5, colors="#555")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.grid(axis="y", alpha=0.2, linewidth=0.6)
    ax.legend(fontsize=8.5, framealpha=0.9, edgecolor="#CCCCCC", loc="lower left")
    ax.set_title(
        f"Follower Polarity Analysis — {variant_name}\n"
        f"{event_label}   |   {leader_signal_note}",
        fontsize=9, fontweight="bold", color="#1F3964", pad=6, loc="left",
    )

    plt.tight_layout()
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
