# tree_to_persona

Persona preference study for Pareto frontier decision trees.

Tests whether LLM-generated, persona-tailored descriptions help stakeholders
select trees from the Pareto frontier — and whether they converge toward the
knee point (the balanced optimal).

---

## Research Design

**9 GenderMag-inspired personas**: 3 roles × 3 cognitive styles

| Persona | Role | Style | Tech Fluency | Can parse raw tree? |
|---------|------|-------|-------------|---------------------|
| PjM-Abi | Project Manager | Abi (low SE, process-oriented) | 1/5 | No (threshold 2) |
| PjM-Pat | Project Manager | Pat (medium SE, reflective) | 1/5 | Partially (threshold 3) |
| PjM-Tim | Project Manager | Tim (high SE, selective) | 1/5 | Mostly (threshold 4) |
| PdM-Abi | Product Manager | Abi | 3/5 | No (threshold 2) |
| PdM-Pat | Product Manager | Pat | 3/5 | Partially (threshold 3) |
| PdM-Tim | Product Manager | Tim | 3/5 | Yes (threshold 5) |
| SWE-Abi | Software Engineer | Abi | 5/5 | Always |
| SWE-Pat | Software Engineer | Pat | 5/5 | Always |
| SWE-Tim | Software Engineer | Tim | 5/5 | Always |

**Phase 1 — raw tree**
Each persona sees the frontier attribute table + the raw EZR decision rules.
Low-tech personas may say "not sure" (they can't reliably read the rules).

**Phase 2 — with description**
Ollama generates a plain-language description of each tree, tailored to the
persona's role and cognitive style. Then the persona must choose — "not sure"
is not allowed.

**Key hypotheses**
- H1: Low-tech personas (PjM-Abi, PdM-Abi) say "not sure" more in Phase 1
- H2: Descriptions reduce "not sure" responses in Phase 2
- H3: After descriptions, personas shift toward the Pareto knee (balanced tree)
- H6 (Convergent Divergence): Descriptions cause convergence on facts but
  divergence on which tree fits each role's context

---

## Files

| File | Purpose |
|------|---------|
| `evaluate.py` | **Main script** — runs Phase 1 and Phase 2 for all 9 personas via Ollama |
| `tree_dsl.py` | DSL for parsing EZR raw output into structured tree objects |
| `phase1_generator.py` | Generates Phase 1 evaluation prompts in batch (JSON) |
| `phase2_customizer.py` | Adds role/style-specific sections to base explanations |
| `ollama_simulator.py` | Batch persona simulator using Ollama (alternative runner) |

---

## Quickstart

### Prerequisites

1. Install [Ollama](https://ollama.ai/)
2. Pull the model:
   ```bash
   ollama pull neural-chat
   ```
3. Start the Ollama server in a separate terminal:
   ```bash
   ollama serve
   ```
4. Install Python dependencies:
   ```bash
   pip install -r ../requirements.txt
   ```

### Step 1: Generate a Pareto frontier

```bash
cd ../pareto_generation
python3 ezr_pareto_analysis.py --dataset ../A2C_Acrobot.csv --runs 50 --out acrobot_output
```

This produces `acrobot_output/pareto_2d.json` (and a 3D variant if `--dims` has two values).

### Step 2: Run the persona study

```bash
cd ../tree_to_persona

# Dry run — prints prompts, no Ollama calls needed
python3 evaluate.py --pareto ../pareto_generation/acrobot_output/pareto_2d.json --dry-run

# Full run (requires ollama serve + neural-chat pulled)
python3 evaluate.py --pareto ../pareto_generation/acrobot_output/pareto_2d.json \
    --out preferences.csv \
    --desc-out descriptions.json
```

### Step 3: Review results

`preferences.csv` — one row per persona:
```
persona, role, style, can_parse_raw,
phase1_choice, phase1_reason,
phase2_choice, phase2_reason, changed
```

`descriptions.json` — all Ollama-generated descriptions, keyed by persona then run number.

---

## Using example output (BankChurners)

The `../pareto_generation/example_output/` directory has pre-computed results
from the BankChurners dataset. To run the study against those:

```bash
python3 evaluate.py \
    --pareto ../pareto_generation/example_output/pareto_2d.json \
    --out bank_preferences.csv
```

---

## Dataset: A2C_Acrobot

Located at `../A2C_Acrobot.csv`. Reinforcement learning hyperparameter
optimisation dataset (223 rows, 14 columns) from the timm/moot repository.
Columns describe algorithm hyperparameters; the target is whether the
agent learned to swing up the Acrobot arm successfully.

This dataset is a cleaner alternative to BankChurners because it does not
contain a client ID column (CLIENTNUM) that leaks into the model as a
memorisation artifact.
