"""
plot_results.py
===============
Generate publication-quality plots for the NeuroProof benchmark results.
Requires: matplotlib, numpy, pandas (installed in experiments/ venv).
"""

from __future__ import annotations
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for servers
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D


# ── Style settings ────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         11,
    'axes.labelsize':    12,
    'axes.titlesize':    13,
    'legend.fontsize':   10,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'figure.dpi':        150,
    'savefig.bbox':      'tight',
    'savefig.dpi':       300,
    'text.usetex':       False,   # set True if LaTeX installed
})

COLORS = {
    'NeuroProof': '#1f77b4',
    'DPLL-Baseline': '#ff7f0e',
    'NeuroProof+ATSS': '#2ca02c',
}
MARKERS = {
    'NeuroProof': 'o',
    'DPLL-Baseline': 's',
    'NeuroProof+ATSS': '^',
}


def load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


# ── Figure 1: Phase Transition ────────────────────────────────────────────────

def plot_phase_transition(df: pd.DataFrame, out_dir: str) -> None:
    """
    Fig 1: Fraction of SAT instances vs clause-to-variable ratio
    for NeuroProof and DPLL-Baseline.
    """
    data = df[df['name'].str.startswith('rand3cnf')].copy()
    if data.empty:
        return

    # Extract ratio from name
    data['ratio'] = data['name'].str.extract(r'_r([\d.]+)').astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # --- Left: SAT fraction ---
    ax = axes[0]
    for solver, grp in data.groupby('solver'):
        grouped = grp.groupby('ratio')['status']
        ratios = sorted(grouped.groups.keys())
        fracs = [
            (grouped.get_group(r) == 'SAT').mean()
            for r in ratios
        ]
        ax.plot(ratios, fracs, marker=MARKERS.get(solver, 'o'),
                color=COLORS.get(solver, 'gray'),
                label=solver, linewidth=1.8, markersize=5)

    ax.axvline(4.267, color='gray', linestyle='--', alpha=0.6, label='Phase transition (4.27)')
    ax.set_xlabel('Clause-to-variable ratio $\\alpha$')
    ax.set_ylabel('Fraction SAT')
    ax.set_title('(a) Phase Transition (n=30)')
    ax.legend(loc='lower left', fontsize=9)
    ax.set_xlim(2.0, 6.0)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

    # --- Right: Median solve time ---
    ax2 = axes[1]
    for solver, grp in data.groupby('solver'):
        grouped = grp.groupby('ratio')['time_sec']
        ratios = sorted(grouped.groups.keys())
        medians = [grouped.get_group(r).median() * 1000 for r in ratios]
        ax2.semilogy(ratios, medians, marker=MARKERS.get(solver, 'o'),
                     color=COLORS.get(solver, 'gray'),
                     label=solver, linewidth=1.8, markersize=5)

    ax2.axvline(4.267, color='gray', linestyle='--', alpha=0.6)
    ax2.set_xlabel('Clause-to-variable ratio $\\alpha$')
    ax2.set_ylabel('Median solve time (ms)')
    ax2.set_title('(b) Solve Time vs Ratio')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.set_xlim(2.0, 6.0)
    ax2.grid(alpha=0.3, which='both')

    fig.tight_layout()
    out = os.path.join(out_dir, 'fig1_phase_transition.pdf')
    fig.savefig(out)
    print(f"Saved: {out}")
    plt.close(fig)


# ── Figure 2: Pigeonhole Principle ───────────────────────────────────────────

def plot_pigeonhole(df: pd.DataFrame, out_dir: str) -> None:
    """Fig 2: Proof size and solve time for PHP_n."""
    data = df[df['name'].str.startswith('PHP_')].copy()
    if data.empty:
        return

    data['n'] = data['name'].str.extract(r'PHP_(\d+)').astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # --- Left: Solve time ---
    ax = axes[0]
    for solver, grp in data.groupby('solver'):
        ns = sorted(grp['n'].unique())
        times = [grp[grp['n'] == n]['time_sec'].values[0] * 1000 if
                 len(grp[grp['n'] == n]) > 0 else float('nan')
                 for n in ns]
        ax.semilogy(ns, times, marker=MARKERS.get(solver, 'o'),
                    color=COLORS.get(solver, 'gray'),
                    label=solver, linewidth=1.8, markersize=6)

    ax.set_xlabel('Number of holes $n$')
    ax.set_ylabel('Solve time (ms, log scale)')
    ax.set_title('(a) Pigeonhole: Solve Time')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which='both')

    # --- Right: Proof size (NeuroProof only, UNSAT proofs) ---
    ax2 = axes[1]
    np_data = data[data['solver'] == 'NeuroProof']
    if not np_data.empty:
        ns = sorted(np_data['n'].unique())
        sizes = [np_data[np_data['n'] == n]['proof_size'].values[0]
                 if len(np_data[np_data['n'] == n]) > 0 else 0 for n in ns]
        ax2.bar(ns, sizes, color=COLORS['NeuroProof'], alpha=0.8)
        # Fit exponential trendline
        valid = [(n, s) for n, s in zip(ns, sizes) if s > 0]
        if len(valid) >= 3:
            xs = np.array([v[0] for v in valid])
            ys = np.array([v[1] for v in valid])
            coeffs = np.polyfit(xs, np.log(ys + 1), 1)
            trend_y = np.exp(np.polyval(coeffs, xs))
            ax2.plot(xs, trend_y, 'r--', label='Exp. fit', linewidth=1.5)

    ax2.set_xlabel('Number of holes $n$')
    ax2.set_ylabel('Proof size (# steps)')
    ax2.set_title('(b) Pigeonhole: Proof Size')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(out_dir, 'fig2_pigeonhole.pdf')
    fig.savefig(out)
    print(f"Saved: {out}")
    plt.close(fig)


# ── Figure 3: Proof Quality Table ─────────────────────────────────────────────

def plot_proof_quality(df: pd.DataFrame, out_dir: str) -> None:
    """Fig 3: Proof size/depth for classical tautologies."""
    data = df[df['name'].str.startswith('tauto(')].copy()
    if data.empty:
        return

    data['formula'] = data['name'].str.extract(r'tauto\((.+)\)')
    data = data[data['status'] == 'PROVED']

    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')

    col_labels = ['Formula', 'Proof Size', 'Proof Depth', 'Time (ms)']
    rows = []
    for _, row in data.iterrows():
        rows.append([
            row['formula'][:40],
            str(row['proof_size']),
            str(row['proof_depth']),
            f"{row['time_sec'] * 1000:.2f}"
        ])

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        colWidths=[0.5, 0.15, 0.15, 0.15]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Style header
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor('#2c5f8a')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # Alternating row colors
    for i in range(1, len(rows) + 1):
        for j in range(len(col_labels)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#e8f0f8')

    ax.set_title('Proof Quality on Classical Tautologies (NeuroProof+ATSS)',
                 fontsize=12, y=0.98)

    out = os.path.join(out_dir, 'fig3_proof_quality.pdf')
    fig.savefig(out)
    print(f"Saved: {out}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    experiments_dir = os.path.dirname(__file__)
    csv_path = os.path.join(experiments_dir, 'results.csv')

    if not os.path.exists(csv_path):
        print(f"No results found at {csv_path}")
        print("Run benchmark_suite.py first.")
        sys.exit(1)

    df = load_results(csv_path)
    print(f"Loaded {len(df)} results from {csv_path}")

    out_dir = os.path.join(experiments_dir, 'figures')
    os.makedirs(out_dir, exist_ok=True)

    plot_phase_transition(df, out_dir)
    plot_pigeonhole(df, out_dir)
    plot_proof_quality(df, out_dir)

    print("All figures generated.")
