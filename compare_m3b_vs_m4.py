"""
compare_m3b_vs_m4.py
====================
Compare MA-FDE-LLM M3b vs M4 on both J&J and AstraZeneca datasets.

Goal: Prove M4 improves AZ generalization while preserving J&J accuracy.

Usage:
    cd phase_9
    python compare_m3b_vs_m4.py

Produces a side-by-side table (Pearson, DTW) and saves JSON to
phase_9/m3b_vs_m4_comparison.json
"""

import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import time
import json
import math

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr

from backend.config import AlgorithmConfig
from backend.data_loader import load_dataset
from backend.simulation_runner import SimulationRunner
from backend.algorithm import get_algorithm


# ---------------------------------------------------------------------------
# Dataset configurations
# ---------------------------------------------------------------------------

DATASETS = {
    'jj_vaccine': dict(
        dataset_id='jj_vaccine',
        grid_rows=13,
        grid_cols=12,
        seed=42,
        n_leader_connections=3,
        n_follower_connections=2,
        sim_start=-5,
        sim_end=25,
        resolution_anchor_day=10,
    ),
    'az_vaccine': dict(
        dataset_id='az_vaccine',
        grid_rows=8,
        grid_cols=10,
        seed=42,
        n_leader_connections=3,
        n_follower_connections=2,
        sim_start=-5,
        sim_end=41,
        resolution_anchor_day=-999,
    ),
}

VARIANTS = ['mafdm_m3b', 'mafdm_m5']


def make_config(ds_cfg: dict) -> AlgorithmConfig:
    return AlgorithmConfig(
        R=0.99, W=0.30, ALPHA=0.40,
        GAMMA=0.90, LAM=0.50,
        LAM_FALL=0.30,
        LAM_RECOVER=0.70,
        EPS_VEL_ALPHA=0.30,
        EPS_INIT=0.90,
        EPS_MIN_CRISIS=0.30,
        EPS_MIN_RECOVERY=0.50,
        EPS_MAX=1.50,
        EPS_BETA=3.00,
        EPS_FIXED=0.90,
        MEMORY_THRESHOLD=0.08,
        MEMORY_ROLL_K=6,
        MEMORY_PROMPT_N=4,
        SIM_START=ds_cfg['sim_start'],
        SIM_END=ds_cfg['sim_end'],
        GRID_ROWS=ds_cfg['grid_rows'],
        GRID_COLS=ds_cfg['grid_cols'],
        RESOLUTION_ANCHOR_DAY=ds_cfg['resolution_anchor_day'],
        SEED=ds_cfg['seed'],
        LLM_MODEL='gpt-4o',
        LLM_TEMPERATURE=0.0,
        LLM_MAX_TOKENS=5,
        LLM_MAX_WORKERS=15,
    )


def dtw_distance(s: np.ndarray, t: np.ndarray) -> float:
    n, m = len(s), len(t)
    d = np.full((n + 1, m + 1), np.inf)
    d[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = (s[i-1] - t[j-1])**2 + min(d[i-1,j], d[i,j-1], d[i-1,j-1])
    return math.sqrt(d[n, m])


def run_variant(variant_id: str, ds_cfg: dict) -> dict:
    config = make_config(ds_cfg)
    ds = load_dataset(
        dataset_id=ds_cfg['dataset_id'],
        grid_rows=ds_cfg['grid_rows'],
        grid_cols=ds_cfg['grid_cols'],
        seed=ds_cfg['seed'],
        n_leader_connections=ds_cfg['n_leader_connections'],
        n_follower_connections=ds_cfg['n_follower_connections'],
    )
    algo   = get_algorithm(variant_id, config)
    runner = SimulationRunner()

    t0 = time.time()
    day_results = runner.run(dataset=ds, algorithm=algo, config=config)
    elapsed = time.time() - t0

    days_list    = [r.day for r in day_results]
    fol_raw      = np.array([r.follower_avg for r in day_results])
    fol_smoothed = gaussian_filter1d(fol_raw, sigma=1.5)
    real_arr     = np.array([ds.real_curve.get(d, float('nan')) for d in days_list])

    pearson, pval = pearsonr(fol_smoothed, real_arr)
    dtw           = dtw_distance(fol_smoothed, real_arr)

    return {
        'pearson':   round(float(pearson), 4),
        'p_value':   round(float(pval),    4),
        'dtw':       round(dtw,            4),
        'elapsed_s': round(elapsed,        1),
    }


def main():
    print('=' * 70)
    print('  MA-FDE-LLM  M3b vs M4  --  J&J and AstraZeneca')
    print('=' * 70)

    results = {}

    for ds_id, ds_cfg in DATASETS.items():
        print(f'\n  Dataset: {ds_id}')
        results[ds_id] = {}
        for variant_id in VARIANTS:
            print(f'  Running {variant_id}...', flush=True)
            r = run_variant(variant_id, ds_cfg)
            results[ds_id][variant_id] = r
            print(f'    Pearson {r["pearson"]:+.4f}  DTW {r["dtw"]:.4f}  ({r["elapsed_s"]:.0f}s)')

    # Summary table
    print(f'\n{"=" * 70}')
    print('  COMPARISON SUMMARY')
    print(f'  {"Dataset":<15} {"Variant":<35} {"Pearson":>8} {"DTW":>8}')
    print(f'  {"-"*15} {"-"*35} {"-"*8} {"-"*8}')
    for ds_id in DATASETS:
        for variant_id in VARIANTS:
            r = results[ds_id][variant_id]
            print(f'  {ds_id:<15} {variant_id:<35} {r["pearson"]:>+8.4f} {r["dtw"]:>8.4f}')
    print(f'{"=" * 70}')

    # Delta analysis
    print('\n  M4 vs M3b delta (positive = M4 better):')
    for ds_id in DATASETS:
        m3 = results[ds_id]['mafdm_m3b']
        m4 = results[ds_id]['mafdm_m4']
        d_pearson = m4['pearson'] - m3['pearson']
        d_dtw     = m3['dtw']    - m4['dtw']       # positive = M4 lower = better
        print(f'  {ds_id:<15}  Pearson delta: {d_pearson:+.4f}   DTW delta: {d_dtw:+.4f}')

    # Save
    out_path = os.path.join(_HERE, 'm3b_vs_m4_comparison.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved: {out_path}')


if __name__ == '__main__':
    main()
