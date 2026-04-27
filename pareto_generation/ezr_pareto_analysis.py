#!/usr/bin/env python3
"""
EZR Pareto Analysis — 2D and 3D
Runs EZR N times, then shows two Pareto frontiers across three metrics:

  Accuracy        hold-out accuracy (%)              higher is better
  Stability       avg feature frequency across runs  higher is better
  Tree Complexity normalized composite score 0–1     lower is better

  2D frontier: Accuracy × Stability
  3D frontier: Accuracy × Stability × Tree Complexity

Tree Complexity formula (Souza et al. NeurIPS 2022 / Dessì et al. ECAI 2023):
  Score = (depth_norm + k × attrs_norm) / (1 + k)
  depth_norm = tree_depth / max(tree_depth across all runs)
  attrs_norm = num_features / max(num_features across all runs)
  k = 1.0 by default (equal weight; justified by cognitive load research)

Usage:
  python3 ezr_pareto_analysis.py --dataset data.csv
  python3 ezr_pareto_analysis.py --dataset data.csv --runs 30 --output-dir my_output
  python3 ezr_pareto_analysis.py --dataset data.csv --k 2.0
"""

import subprocess, re, argparse, sys, json
import pandas as pd
from pathlib import Path
from collections import defaultdict


# --- EZR Runner ---------------------------------------------------------------

def run_ezr(dataset, n_runs, base_seed, k=1.0):
    """Run EZR n_runs times. Returns (trees, feature_counts).

    Each tree dict exposes only: run, accuracy, stability, tree_complexity,
    features, raw_output. Internal fields (tree_depth, num_features) are used
    during computation then discarded from outputs.
    """
    trees = []
    feature_counts = defaultdict(int)

    for i in range(1, n_runs + 1):
        seed = base_seed + i
        print(f"[{i}/{n_runs}] Running EZR (seed={seed})...")
        try:
            result = subprocess.run(
                ["ezr", "-f", str(dataset), "-s", str(seed)],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode != 0:
                print(f"  Warning: run failed"); continue

            tree = _parse(result.stdout, i)
            if tree:
                trees.append(tree)
                for f in tree['features']:
                    feature_counts[f] += 1
                print(f"  accuracy={tree['accuracy']}"
                      f"  depth={tree['_depth']}  attrs={tree['_num_features']}"
                      f"  features={', '.join(tree['features'])}")
        except Exception as e:
            print(f"  Warning: {e}")

    if not trees:
        return trees, dict(feature_counts)

    n = len(trees)

    # Stability: avg frequency of a tree's own features across all runs (%)
    for tree in trees:
        if tree['features']:
            avg = sum(feature_counts[f] for f in tree['features']) / len(tree['features'])
            tree['stability'] = round((avg / n) * 100, 1)
        else:
            tree['stability'] = 0.0

    # Tree Complexity: normalized composite score (0–1, lower = simpler)
    max_depth = max(t['_depth'] for t in trees)
    max_attrs = max(t['_num_features'] for t in trees)
    for tree in trees:
        depth_norm = tree['_depth'] / max_depth if max_depth else 0
        attrs_norm = tree['_num_features'] / max_attrs if max_attrs else 0
        tree['tree_complexity'] = round((depth_norm + k * attrs_norm) / (1 + k), 4)

    return trees, dict(feature_counts)


def _parse(output, run_num):
    """Parse one EZR run output. Returns tree dict with internal fields prefixed _."""
    lines = output.strip().split('\n')
    if len(lines) < 2:
        return None

    m = re.search(r'hold.out[:\s]+(\d+)', lines[-1])
    if not m:
        return None
    accuracy = int(m.group(1))

    features = []
    used_line = lines[-2].strip()
    if used_line.startswith('Used:'):
        features = used_line[len('Used:'):].split()
    if not features:
        for line in lines:
            m2 = re.search(r'if\s+(\w+)\s+', line)
            if m2 and m2.group(1) not in features:
                features.append(m2.group(1))

    rule_lines = [l for l in lines if re.search(r'if\s+\w+', l)]
    depth = max((l.count('|') for l in rule_lines), default=0) + 1

    return {
        'run':            run_num,
        'accuracy':       accuracy,
        'stability':      0.0,         # filled after all runs
        'tree_complexity': 0.0,        # filled after all runs
        'features':       features,
        'raw_output':     output,
        '_depth':         depth,       # internal — used for tree_complexity calc
        '_num_features':  len(features),
    }


# --- Pareto Logic -------------------------------------------------------------

# Dimension metadata: key → (axis label, maximize)
DIM_META = {
    'accuracy':        ('Accuracy (%)',         True),
    'stability':       ('Stability (%)',         True),
    'tree_complexity': ('Tree Complexity (0–1)', False),
}

# All unique 2D pairs across the three metrics
PAIRS_2D = [
    ('accuracy',  'stability'),
    ('accuracy',  'tree_complexity'),
    ('stability', 'tree_complexity'),
]

DIMS_3D = [('accuracy', True), ('stability', True), ('tree_complexity', False)]


def pareto_nd(trees, dims):
    def dominates(b, a):
        at_least = all(b[k] >= a[k] if mx else b[k] <= a[k] for k, mx in dims)
        strictly  = any(b[k] >  a[k] if mx else b[k] <  a[k] for k, mx in dims)
        return at_least and strictly
    return [a for a in trees if not any(dominates(b, a) for b in trees if b is not a)]


def knee(frontier, dims):
    vals = {k: [t[k] for t in frontier] for k, _ in dims}
    def score(t):
        scores = []
        for k, maximize in dims:
            lo, hi = min(vals[k]), max(vals[k])
            norm = (t[k] - lo) / max(hi - lo, 1e-9)
            scores.append(norm if maximize else 1 - norm)
        return sum(scores) / len(scores)
    return max(frontier, key=score)


# --- Output ------------------------------------------------------------------

def _pub(tree, rank=None):
    """Public-facing tree dict — strips internal _ fields."""
    d = {
        'run':             tree['run'],
        'accuracy':        tree['accuracy'],
        'stability':       tree['stability'],
        'tree_complexity': tree['tree_complexity'],
        'features':        tree['features'],
        'raw_output':      tree['raw_output'],
    }
    if rank is not None:
        d['rank'] = rank
    return d


def save_csv(trees, frontiers_2d, f3d, feature_counts, out):
    f3d_ids = set(id(t) for t in f3d)
    frontier_cols = {
        f'on_{dx}_vs_{dy}_frontier': set(id(t) for t in f)
        for (dx, dy), f in frontiers_2d.items()
    }
    pd.DataFrame([{
        'run':             t['run'],
        'accuracy':        t['accuracy'],
        'stability':       t['stability'],
        'tree_complexity': t['tree_complexity'],
        'features':        ', '.join(t['features']),
        'on_3d_frontier':  id(t) in f3d_ids,
        **{col: id(t) in ids for col, ids in frontier_cols.items()},
    } for t in trees]).to_csv(out / 'all_trees.csv', index=False)

    n = len(trees)
    pd.DataFrame([{
        'feature':     f,
        'occurrences': c,
        'frequency_%': round(c / n * 100, 1),
    } for f, c in sorted(feature_counts.items(), key=lambda x: -x[1])
    ]).to_csv(out / 'feature_frequency.csv', index=False)


def save_json(trees, frontiers_2d, knees_2d, f3d, k3d, out):
    with open(out / 'all_trees.json', 'w') as f:
        json.dump({'total': len(trees), 'trees': [_pub(t) for t in trees]}, f, indent=2)

    for (dim_x, dim_y), frontier in frontiers_2d.items():
        k_pt = knees_2d[(dim_x, dim_y)]
        sorted_f = sorted(frontier, key=lambda t: t['accuracy'], reverse=True)
        fname = f'pareto_2d_{dim_x}_vs_{dim_y}.json'
        with open(out / fname, 'w') as f:
            json.dump({
                'frontier_size': len(sorted_f),
                'dimensions':    f'{dim_x} × {dim_y}',
                'knee_run':      k_pt['run'],
                'trees': [_pub(t, rank=i) for i, t in enumerate(sorted_f, 1)],
            }, f, indent=2)

    sorted_3d = sorted(f3d, key=lambda t: t['accuracy'], reverse=True)
    with open(out / 'pareto_3d.json', 'w') as f:
        json.dump({
            'frontier_size': len(sorted_3d),
            'dimensions':    'accuracy × stability × tree_complexity',
            'knee_run':      k3d['run'],
            'trees': [_pub(t, rank=i) for i, t in enumerate(sorted_3d, 1)],
        }, f, indent=2)


def plot_2d(trees, frontier, k_pt, dim_x, dim_y, out):
    """Generic 2D Pareto scatter for any two of the three metrics."""
    import matplotlib.pyplot as plt

    x_label, _ = DIM_META[dim_x]
    y_label, _ = DIM_META[dim_y]
    off = [t for t in trees if t not in frontier]

    fig, ax = plt.subplots(figsize=(10, 7))

    if off:
        ax.scatter([t[dim_x] for t in off], [t[dim_y] for t in off],
                   s=80, alpha=0.4, color='#cccccc', edgecolors='#888', linewidth=0.5,
                   label='Off-frontier')
        for t in off:
            ax.annotate(f"R{t['run']}", (t[dim_x], t[dim_y]),
                        xytext=(4, 4), textcoords='offset points',
                        fontsize=7, color='#aaaaaa', alpha=0.8)

    ax.scatter([t[dim_x] for t in frontier], [t[dim_y] for t in frontier],
               s=150, alpha=0.85, color='#2ecc71', edgecolors='#27ae60', linewidth=1.5,
               label='Pareto frontier')

    sorted_f = sorted(frontier, key=lambda t: t[dim_x])
    ax.plot([t[dim_x] for t in sorted_f], [t[dim_y] for t in sorted_f],
            'g--', alpha=0.3, linewidth=1)

    for t in frontier:
        ax.annotate(f"Run {t['run']}", (t[dim_x], t[dim_y]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, fontweight='bold', color='#27ae60')

    ax.scatter([k_pt[dim_x]], [k_pt[dim_y]], s=350, color='#e74c3c',
               marker='*', edgecolors='#c0392b', linewidth=2, zorder=5, label='Knee')

    ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
    ax.set_title(f'2D Pareto Front: {x_label} vs. {y_label}', fontsize=14, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98, f'Runs: {len(trees)}  |  Frontier: {len(frontier)}',
            transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    fname = f'pareto_2d_{dim_x}_vs_{dim_y}.png'
    plt.savefig(out / fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  {fname}")


def plot_3d(trees, frontier, k_pt, out):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from matplotlib.colors import Normalize

    off = [t for t in trees if t not in frontier]
    fig = plt.figure(figsize=(14, 10), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    if off:
        ax.scatter([t['stability'] for t in off],
                   [t['tree_complexity'] for t in off],
                   [t['accuracy'] for t in off],
                   c='#999', s=50, alpha=0.4, edgecolors='#666', linewidth=0.3,
                   label='Off-frontier')

    acc_vals = [t['accuracy'] for t in frontier]
    norm = Normalize(vmin=min(acc_vals), vmax=max(acc_vals))

    unique_xy = set((t['stability'], t['tree_complexity']) for t in frontier)
    if len(unique_xy) >= 3:
        ax.plot_trisurf(
            [t['stability'] for t in frontier],
            [t['tree_complexity'] for t in frontier],
            [t['accuracy'] for t in frontier],
            cmap='RdYlGn', norm=norm, alpha=0.6,
            edgecolor='#444', linewidth=0.5, shade=True
        )

    sc = ax.scatter(
        [t['stability'] for t in frontier],
        [t['tree_complexity'] for t in frontier],
        [t['accuracy'] for t in frontier],
        c=acc_vals, cmap='RdYlGn', norm=norm, s=180, alpha=0.95,
        edgecolors='#1a1a1a', linewidth=1.5, zorder=5
    )
    for i, t in enumerate(sorted(frontier, key=lambda x: x['accuracy'], reverse=True), 1):
        ax.text(t['stability'], t['tree_complexity'], t['accuracy'], str(i),
                fontsize=10, fontweight='bold', ha='center', va='center', zorder=20)

    ax.scatter([k_pt['stability']], [k_pt['tree_complexity']], [k_pt['accuracy']],
               c='#e74c3c', s=300, marker='*', edgecolors='#c0392b',
               linewidth=2, zorder=10, label='Knee (optimal)')

    plt.colorbar(sc, ax=ax, pad=0.1, shrink=0.7, label='Accuracy (%)')
    ax.set_xlabel('\nStability (%)', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_ylabel('\nTree Complexity (0–1)', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_zlabel('\nAccuracy (%)', fontsize=12, fontweight='bold', labelpad=8)
    ax.zaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))
    ax.set_title('3D Pareto Front: Accuracy × Stability × Tree Complexity',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(); ax.view_init(elev=25, azim=45)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False

    plt.tight_layout()
    plt.savefig(out / 'pareto_3d.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  pareto_3d.png")


def print_summary(trees, frontiers_2d, knees_2d, f3d, k3d, feature_counts, k=1.0):
    n = len(trees)
    print(f"\n{'='*60}")
    print(f"SUMMARY  ({n} runs)")
    print(f"{'='*60}")

    for (dim_x, dim_y), frontier in frontiers_2d.items():
        k_pt = knees_2d[(dim_x, dim_y)]
        print(f"\n2D Frontier ({dim_x} × {dim_y}): {len(frontier)} trees")
        print(f"  Knee: Run {k_pt['run']}"
              f"  {dim_x}={k_pt[dim_x]}  {dim_y}={k_pt[dim_y]}")

    print(f"\n3D Frontier (accuracy × stability × tree_complexity): {len(f3d)} trees")
    print(f"  Complexity formula: (depth_norm + {k} × attrs_norm) / {1 + k}")
    print(f"  Knee: Run {k3d['run']}"
          f"  accuracy={k3d['accuracy']}%  stability={k3d['stability']}%"
          f"  tree_complexity={k3d['tree_complexity']}")

    print(f"\nTop Features (by frequency across all runs):")
    for feat, cnt in sorted(feature_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {feat}: {cnt}/{n} ({cnt/n*100:.0f}%)")
    print(f"{'='*60}\n")


# --- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='EZR Pareto Analysis — Accuracy × Stability × Tree Complexity',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Metrics:
  accuracy        hold-out accuracy (%)
  stability       avg feature frequency across runs (%)
  tree_complexity normalized composite: (depth_norm + k*attrs_norm) / (1+k)
        """
    )
    parser.add_argument('--dataset', required=True, help='Path to dataset CSV')
    parser.add_argument('--runs', type=int, default=50, help='Number of EZR runs (default: 50)')
    parser.add_argument('--seed', type=int, default=1234567891,
                        help='Base random seed; run i uses seed+i (default: 1234567891)')
    parser.add_argument('--output-dir', default='ezr_output',
                        help='Output directory (default: ezr_output)')
    parser.add_argument('--k', type=float, default=1.0,
                        help='Complexity weight: k=1 equal, k>1 penalises features more (default: 1.0)')
    args = parser.parse_args()

    dataset = Path(args.dataset)
    if not dataset.exists():
        print(f"Error: {args.dataset} not found"); sys.exit(1)

    out = Path(args.output_dir)
    out.mkdir(exist_ok=True)

    print(f"Complexity weighting: k={args.k}")
    trees, feature_counts = run_ezr(dataset, args.runs, args.seed, k=args.k)
    if not trees:
        print("No trees collected."); sys.exit(1)

    # 2D frontiers for all three pairs
    frontiers_2d = {}
    knees_2d = {}
    for dim_x, dim_y in PAIRS_2D:
        dims = [(dim_x, DIM_META[dim_x][1]), (dim_y, DIM_META[dim_y][1])]
        f = pareto_nd(trees, dims)
        frontiers_2d[(dim_x, dim_y)] = f
        knees_2d[(dim_x, dim_y)]     = knee(f, dims)

    # 3D frontier
    f3d = pareto_nd(trees, DIMS_3D)
    k3d = knee(f3d, DIMS_3D)

    print("\nSaving outputs...")
    save_csv(trees, frontiers_2d, f3d, feature_counts, out)
    print("  all_trees.csv  feature_frequency.csv")
    save_json(trees, frontiers_2d, knees_2d, f3d, k3d, out)
    print("  all_trees.json  pareto_2d_*.json  pareto_3d.json")

    for (dim_x, dim_y), f in frontiers_2d.items():
        plot_2d(trees, f, knees_2d[(dim_x, dim_y)], dim_x, dim_y, out)
    plot_3d(trees, f3d, k3d, out)

    print_summary(trees, frontiers_2d, knees_2d, f3d, k3d, feature_counts, k=args.k)
    print(f"Results in: {out}/")


if __name__ == '__main__':
    main()
