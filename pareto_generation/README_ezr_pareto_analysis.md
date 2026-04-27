# EZR Pareto Analysis

`ezr_pareto_analysis.py` runs EZR N times and produces both **2D** and **3D** Pareto frontier analyses in one command. EZR is run once — results are shared across both analyses.

---

## Dimensions

**Accuracy** is always the primary axis. Pick 1 or 2 extra dimensions via `--dims`:

| Dimension | Meaning | Better when |
|---|---|---|
| `accuracy` | EZR hold-out win% — how well the tree predicts | higher *(always included)* |
| `complexity` | Number of features used | lower |
| `stability` | Avg feature frequency across runs (%) | higher |
| `overfitting_gap` | Train accuracy − hold-out accuracy (%) | lower |
| `tree_depth` | Decision levels in the tree | lower |
| `coverage` | Rows reaching the root split | higher |

- **1 dim** → 2D plot only (Accuracy × dim1)
- **2 dims** → 2D plot + 3D plot (Accuracy × dim1 × dim2)

---

## Usage

```bash
python3 ezr_pareto_analysis.py --dataset BankChurners.csv
```

| Argument | Default | Description |
|---|---|---|
| `--dataset PATH` | *(required)* | Path to dataset CSV |
| `--runs N` | `50` | Number of EZR runs |
| `--seed N` | `1234567891` | Base random seed — each run uses `seed+i` for diversity while staying reproducible |
| `--output-dir DIR` | `ezr_output` | Output directory |
| `--dims DIM [DIM]` | `complexity stability` | 1 or 2 extra dimensions |

**Examples:**
```bash
# Default: 2D (accuracy × complexity) + 3D (accuracy × complexity × stability)
python3 ezr_pareto_analysis.py --dataset data.csv

# 2D + 3D swapping stability for overfitting gap
python3 ezr_pareto_analysis.py --dataset data.csv --dims complexity overfitting_gap

# 2D only: accuracy vs tree depth
python3 ezr_pareto_analysis.py --dataset data.csv --dims tree_depth
```

---

## Outputs

Output filenames reflect the dimensions chosen:

```
ezr_output/
  all_trees.csv                  Every run: all metrics + frontier membership flags
  feature_frequency.csv          Feature appearance frequency across all runs
  all_trees.json                 Full tree data for every run, including raw decision tree output
  pareto_2d.json                 2D frontier trees with rank, metrics, and raw output
  pareto_3d.json                 3D frontier trees with rank, metrics, and raw output (if 2 dims)
  pareto_2d_<dim>.png            2D scatter — all runs labeled, frontier highlighted
  pareto_3d_<dim1>_<dim2>.png    3D scatter with % accuracy axis labels (if 2 dims)
```

---

## Understanding the Frontiers

**2D** — A tree is on the frontier if no other tree has both higher accuracy and lower complexity. The **knee** is the simplest tree that doesn't sacrifice accuracy.

**3D** — A tree is on the frontier if no other tree is strictly better on all three dimensions. The **knee** (red star) is the tree with the best normalized balance across all three.

### Choosing a frontier tree

- **Want simple?** Pick the frontier tree with the fewest features
- **Want accurate?** Pick the frontier tree with the highest win score
- **Want stable?** Pick the frontier tree with the highest stability %
- **Want balanced?** Use the **knee point** (red star) — it optimizes all three

This supports *Convergent Divergence*: teams converge on the fact that multiple valid solutions exist, then diverge on which trade-off fits their context.

### Stability thresholds

| Range | Reliability |
|---|---|
| 70%+ | Very reliable — uses core features |
| 50–70% | Reasonable — mixed reliability |
| <50% | Risky — uses exploratory features |

---

## Dependencies

Install everything at once:

```bash
pip install -r requirements.txt
```

This installs EZR directly from the GitHub repo (includes the latest bug fixes not yet on PyPI) plus `pandas`, `numpy`, and `matplotlib`.

If you previously installed EZR via pip, uninstall it first to avoid conflicts:

```bash
pip uninstall ezr
pip install -r requirements.txt
```

EZR must be available as `ezr` in PATH after install.
