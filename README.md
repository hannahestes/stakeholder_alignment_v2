# Stakeholder Alignment via Pareto-Optimal Decision Trees

A two-stage pipeline that (1) finds the best decision trees from a dataset using
multi-objective Pareto optimization, then (2) tests how different stakeholder
personas evaluate and choose between those trees — first from raw rules, then
from LLM-generated plain-language descriptions.

The core research question: **do persona-tailored descriptions help stakeholders
converge on a balanced optimal tree, or do they reveal divergent priorities?**

---

## Pipeline Overview

```
Dataset CSV
    │
    ▼
[pareto_generation/]
    ezr_pareto_analysis.py
    │  Runs EZR 50× with varying seeds
    │  Computes: accuracy, stability, tree_complexity
    │  Finds Pareto frontiers (2D × 3 pairs + 3D)
    │  Outputs: plots, CSVs, JSONs
    │
    ▼
Pareto frontier JSON
    │
    ▼
[tree_to_persona/]
    evaluate.py
    │  Phase 1: 9 personas see raw tree attributes + decision rules
    │            → choose a tree or say "not sure"
    │  Phase 2: Ollama generates descriptions from executable statement
    │            → personas must choose (no "not sure")
    │
    ▼
preferences.csv + descriptions.json
```

---

## Metrics

Every tree is measured on exactly three attributes:

| Metric | Description | Better when |
|--------|-------------|-------------|
| **Accuracy** | Hold-out prediction accuracy (%) | Higher |
| **Stability** | Avg frequency of a tree's features across all runs (%) | Higher |
| **Tree Complexity** | Normalized composite of depth + # features (0–1) | Lower |

Tree Complexity formula (Souza et al. NeurIPS 2022):
```
Score = (depth_norm + k × attrs_norm) / (1 + k)   [default k=1.0]
```
Equal weighting (k=1) is backed by cognitive load research showing depth and
feature count hurt understandability roughly equally.

---

## Personas

9 GenderMag-inspired personas: **3 roles × 3 cognitive styles**

| Role | Tech Fluency | Cares most about |
|------|-------------|-----------------|
| Project Manager (PjM) | 1/5 | Simplicity, stability |
| Product Manager (PdM) | 3/5 | Accuracy, customer impact |
| Software Engineer (SWE) | 5/5 | Accuracy, reliability |

| Style | Self-Efficacy | Info Processing |
|-------|-------------|----------------|
| Abi | Low | Comprehensive, process-oriented |
| Pat | Medium | Comprehensive, reflective |
| Tim | High | Selective, tinkerer |

Low-tech personas (PjM-Abi, PjM-Pat, PdM-Abi) may say "not sure" in Phase 1
if the frontier trees exceed their complexity threshold. After descriptions,
everyone must choose.

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Then install and start Ollama (for persona evaluation):
```bash
# https://ollama.ai/
ollama pull neural-chat
ollama serve   # keep running in a separate terminal
```

### 2. Run Pareto generation

```bash
python3 pareto_generation/ezr_pareto_analysis.py \
    --dataset A2C_Acrobot.csv \
    --output-dir my_output
```

### 3. Run persona evaluation

```bash
python3 tree_to_persona/evaluate.py \
    --pareto my_output/pareto_2d_accuracy_vs_stability.json \
    --out my_output/preferences.csv \
    --desc-out my_output/descriptions.json
```

### Dry run (no EZR or Ollama needed)

```bash
# Persona prompts only — prints without calling Ollama
python3 tree_to_persona/evaluate.py \
    --pareto pareto_generation/example_output/pareto_2d.json \
    --dry-run
```

---

## Outputs

### Pareto generation (`my_output/`)

| File | Description |
|------|-------------|
| `all_trees.csv` | All runs: accuracy, stability, tree_complexity, frontier flags |
| `all_trees.json` | Same, with raw EZR decision rules |
| `feature_frequency.csv` | Feature appearance count across all runs |
| `pareto_2d_accuracy_vs_stability.json` | 2D frontier + knee |
| `pareto_2d_accuracy_vs_tree_complexity.json` | 2D frontier + knee |
| `pareto_2d_stability_vs_tree_complexity.json` | 2D frontier + knee |
| `pareto_3d.json` | 3D frontier + knee (all three metrics) |
| `pareto_2d_accuracy_vs_stability.png` | Scatter: all runs + frontier highlighted |
| `pareto_2d_accuracy_vs_tree_complexity.png` | Scatter |
| `pareto_2d_stability_vs_tree_complexity.png` | Scatter |
| `pareto_3d.png` | 3D surface: accuracy × stability × tree_complexity |

### Persona evaluation (`my_output/`)

| File | Description |
|------|-------------|
| `preferences.csv` | One row per persona: Phase 1 choice, Phase 2 choice, changed? |
| `descriptions.json` | All Ollama-generated descriptions, keyed by persona × run |

---

## Dataset

**A2C_Acrobot.csv** — 223 rows, 14 columns. Reinforcement learning
hyperparameter optimization data from [timm/moot](https://github.com/timm/moot).
Each row is a training run of the A2C algorithm on the Acrobot-v1 environment;
columns are hyperparameters (learning rate, entropy coefficient, etc.) and the
target is whether the agent successfully learned the task.

Preferred over BankChurners for this study because it has no client ID column
that leaks into the model as a memorization artifact.

---

## Directory Structure

```
stakeholder_alignment/
├── README.md                      
├── requirements.txt
├── A2C_Acrobot.csv
│
├── pareto_generation/
│   ├── ezr_pareto_analysis.py     ← Pareto analysis script
│   ├── README_ezr_pareto_analysis.md
│   └── example_output/            ← Pre-computed BankChurners results
│
└── tree_to_persona/
    ├── evaluate.py                ← Main persona study script
    ├── README.md
    ├── tree_dsl.py                ← EZR output parser / DSL
    ├── phase1_generator.py        ← Batch Phase 1 prompt generator
    ├── ollama_simulator.py        ← Alternative batch Ollama runner
    └── example_output/            ← Pre-computed BankChurners persona results
```

---

## Research Context

This pipeline operationalizes **Convergent Divergence** — the idea that when
stakeholders are shown a Pareto frontier of optimal solutions:

- They **converge** on the fact that multiple valid solutions exist (no one
  solution dominates)
- They **diverge** on *which* trade-off best fits their role and context

The **knee point** (balanced optimal, red star in plots) serves as a neutral
starting point for negotiation. The hypothesis is that persona-tailored
descriptions in Phase 2 reduce "not sure" responses and may shift preferences
toward or away from the knee — revealing whether alignment is possible and where
disagreements lie.

See subdirectory READMEs for implementation details:
- [`pareto_generation/README_ezr_pareto_analysis.md`](pareto_generation/README_ezr_pareto_analysis.md)
- [`tree_to_persona/README.md`](tree_to_persona/README.md)
