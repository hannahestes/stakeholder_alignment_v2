#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — SE4AI Full Experiment Pipeline
#
# USAGE:
#   chmod +x run_pipeline.sh
#   ./run_pipeline.sh
#
# WHAT THIS RUNS (in order):
#   Step 1  — EZR calibration loop    (N=25…1000 × 5 seeds, reads local files)
#   Step 2  — Persona experiment      (Gemini via Vertex AI, ~15 min)
#   Step 3a — Claude judge            (Anthropic API, ~12 min)
#   Step 3b — Ollama judge            (local Llama 3.1 70B, ~2-4 hrs)
#   Step 4  — Agreement metrics       (offline, <1 min)
#   Step 5  — Calibration stats       (offline, <1 min)
#
# PREREQUISITES:
#   pip install scipy numpy krippendorff google-genai anthropic
#   gcloud auth application-default login
#   ollama pull llama3.1:70b
#
# =============================================================================
#
# ── SET YOUR API KEYS AND CONFIG HERE ────────────────────────────────────────
#
export ANTHROPIC_API_KEY="your_anthropic_api_key_here"   # ← replace this
export VERTEX_PROJECT="sectesting-492815"                 # ← your GCP project
export VERTEX_LOCATION="us-central1"
export OLLAMA_MODEL="llama3.1:70b"
export OLLAMA_BASE_URL="http://localhost:11434"
#
# ── SKIP FLAGS (set to 1 to skip that step) ───────────────────────────────────
#
# SKIP_CLAUDE=1   — skip Claude judge (use if no API key or reusing old results)
# SKIP_OLLAMA=1   — skip Ollama judge (use if Ollama not running or want Claude only)
# SKIP_CALIB=1    — skip calibration loop (use if pareto files already exist)
#
SKIP_CLAUDE="${SKIP_CLAUDE:-0}"
SKIP_OLLAMA="${SKIP_OLLAMA:-0}"
SKIP_CALIB="${SKIP_CALIB:-0}"
#
# =============================================================================

set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

step()  { echo -e "\n${BOLD}${BLUE}══════════════════════════════════════${NC}"; \
          echo -e "${BOLD}${BLUE}  $1${NC}"; \
          echo -e "${BOLD}${BLUE}══════════════════════════════════════${NC}"; }
ok()    { echo -e "${GREEN}  ✓ $1${NC}"; }
warn()  { echo -e "${YELLOW}  ⚠ $1${NC}"; }
info()  { echo -e "${CYAN}  → $1${NC}"; }
fail()  { echo -e "${RED}  ✗ ERROR: $1${NC}"; exit 1; }

RESULTS_DIR="results"
JUDGE_DIR="judge_results"

# ── Banner ────────────────────────────────────────────────────────────────────
echo -e "\n${BOLD}SE4AI — Full Experiment Pipeline${NC}"
echo -e "$(date '+%Y-%m-%d %H:%M:%S')\n"
echo -e "  Vertex project : ${VERTEX_PROJECT}"
echo -e "  Vertex location: ${VERTEX_LOCATION}"
echo -e "  Ollama model   : ${OLLAMA_MODEL}"
echo -e "  Skip Claude    : ${SKIP_CLAUDE}"
echo -e "  Skip Ollama    : ${SKIP_OLLAMA}"
echo -e "  Skip Calibration: ${SKIP_CALIB}\n"

# ── Preflight checks ──────────────────────────────────────────────────────────
step "PREFLIGHT CHECKS"

# API key sanity check
[[ "${ANTHROPIC_API_KEY}" == "your_anthropic_api_key_here" ]] && \
    [[ "${SKIP_CLAUDE}" == "0" ]] && \
    fail "Replace ANTHROPIC_API_KEY in run_pipeline.sh with your real key, or set SKIP_CLAUDE=1"

# Python
python3 --version &>/dev/null || fail "python3 not found"
ok "python3 found"

# Venv
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
    ok "Activated .venv"
else
    warn "No .venv found — using system python"
fi

# Required python packages
for pkg in scipy numpy krippendorff google.genai anthropic; do
    python3 -c "import ${pkg%%.*}" 2>/dev/null || \
        fail "Python package '${pkg}' not installed. Run: pip install scipy numpy krippendorff google-genai anthropic"
done
ok "All Python packages present"

# Scripts
[[ -f "persona_experiment_v3.py" ]] || fail "persona_experiment_v3.py not found"
[[ -f "llm_judge_final.py" ]]       || fail "llm_judge_final.py not found"
[[ -f "compute_agreement.py" ]]     || fail "compute_agreement.py not found"
[[ -f "calibration_stats.py" ]]     || fail "calibration_stats.py not found"
ok "All scripts found"

# Anthropic API key
if [[ "${SKIP_CLAUDE}" == "0" ]]; then
    [[ -n "${ANTHROPIC_API_KEY:-}" ]] || fail "ANTHROPIC_API_KEY not set"
    ok "ANTHROPIC_API_KEY set"
fi

# Ollama
if [[ "${SKIP_OLLAMA}" == "0" ]]; then
    curl -sf "${OLLAMA_BASE_URL}/api/tags" &>/dev/null || \
        fail "Ollama not reachable at ${OLLAMA_BASE_URL}. Start it or set SKIP_OLLAMA=1"
    ok "Ollama reachable"
fi

# GCP auth
gcloud auth application-default print-access-token &>/dev/null || \
    fail "GCP auth not configured. Run: gcloud auth application-default login"
ok "GCP auth configured"

ok "All preflight checks passed — starting pipeline"

# ── STEP 1: EZR Calibration Loop ─────────────────────────────────────────────
step "STEP 1 / 5 — EZR CALIBRATION (N=25…1000 × 5 seeds)"

if [[ "${SKIP_CALIB}" == "1" ]]; then
    warn "Skipping calibration loop (SKIP_CALIB=1)"
    warn "Assuming Pareto files already exist under Outputs-*/"
else
    info "Reading Pareto files for all 10 run counts × 5 seeds = 50 runs"
    info "Output: ${RESULTS_DIR}/calibration/calibration_raw.json"

    START=$(date +%s)
    python3 persona_experiment_v3.py --calibration-only
    END=$(date +%s)

    [[ -f "${RESULTS_DIR}/calibration/calibration_raw.json" ]] || \
        fail "Calibration output not written — check that Pareto files exist"
    ok "Calibration complete in $(( END - START ))s"
fi

# ── STEP 2: Persona Experiment ────────────────────────────────────────────────
step "STEP 2 / 5 — PERSONA EXPERIMENT (Gemini via Vertex AI)"
info "Running 9 personas × 5 trees × 2 rounds = 90 evaluations + 9 summaries"
info "Using optimal N=400 Pareto frontier"
info "Estimated time: ~15 minutes"

START=$(date +%s)
python3 persona_experiment_v3.py --skip-calibration
END=$(date +%s)

[[ -f "${RESULTS_DIR}/persona_evaluations_round_a.json" ]] || fail "Round A results missing"
[[ -f "${RESULTS_DIR}/persona_evaluations_round_b.json" ]] || fail "Round B results missing"
[[ -f "${RESULTS_DIR}/persona_summaries.json" ]]           || fail "Summaries missing"

N_EVALS=$(python3 -c "
import json
a = json.load(open('${RESULTS_DIR}/persona_evaluations_round_a.json'))
b = json.load(open('${RESULTS_DIR}/persona_evaluations_round_b.json'))
print(len(a)+len(b))")
N_SUM=$(python3 -c "import json; print(len(json.load(open('${RESULTS_DIR}/persona_summaries.json'))))")

ok "Persona experiment complete in $(( END - START ))s"
ok "${N_EVALS} evaluations + ${N_SUM} summaries → ${RESULTS_DIR}/"

# ── STEP 3a: Claude Judge ─────────────────────────────────────────────────────
step "STEP 3a / 5 — CLAUDE JUDGE (Anthropic API)"

if [[ "${SKIP_CLAUDE}" == "1" ]]; then
    warn "Skipping Claude judge (SKIP_CLAUDE=1)"
else
    info "180 API calls (90 evals × 2 tasks). Estimated: ~12 minutes"

    if [[ -f "${JUDGE_DIR}/ollama_rubric_scores.json" ]]; then
        cp "${JUDGE_DIR}/ollama_rubric_scores.json"   "${JUDGE_DIR}/ollama_rubric_scores.backup.json"
        cp "${JUDGE_DIR}/ollama_classifications.json" "${JUDGE_DIR}/ollama_classifications.backup.json"
        info "Backed up existing Ollama files"
    fi

    START=$(date +%s)
    USE_CLAUDE=true USE_OLLAMA=false python3 llm_judge_final.py
    END=$(date +%s)

    [[ -f "${JUDGE_DIR}/claude_rubric_scores.json" ]]   || fail "Claude rubric scores missing"
    [[ -f "${JUDGE_DIR}/claude_classifications.json" ]] || fail "Claude classifications missing"
    ok "Claude judge complete in $(( END - START ))s"

    if [[ -f "${JUDGE_DIR}/ollama_rubric_scores.backup.json" ]]; then
        cp "${JUDGE_DIR}/ollama_rubric_scores.backup.json"   "${JUDGE_DIR}/ollama_rubric_scores.json"
        cp "${JUDGE_DIR}/ollama_classifications.backup.json" "${JUDGE_DIR}/ollama_classifications.json"
        ok "Ollama backup restored"
    fi
fi

# ── STEP 3b: Ollama Judge ─────────────────────────────────────────────────────
step "STEP 3b / 5 — OLLAMA JUDGE (${OLLAMA_MODEL})"

if [[ "${SKIP_OLLAMA}" == "1" ]]; then
    warn "Skipping Ollama judge (SKIP_OLLAMA=1)"
else
    info "180 calls to local model. Estimated: 2–4 hours for 70B"

    if [[ -f "${JUDGE_DIR}/claude_rubric_scores.json" ]]; then
        cp "${JUDGE_DIR}/claude_rubric_scores.json"   "${JUDGE_DIR}/claude_rubric_scores.backup.json"
        cp "${JUDGE_DIR}/claude_classifications.json" "${JUDGE_DIR}/claude_classifications.backup.json"
        info "Backed up Claude files before Ollama run"
    fi

    START=$(date +%s)
    USE_CLAUDE=false USE_OLLAMA=true OLLAMA_MODEL="${OLLAMA_MODEL}" python3 llm_judge_final.py
    END=$(date +%s)

    [[ -f "${JUDGE_DIR}/ollama_rubric_scores.json" ]]   || fail "Ollama rubric scores missing"
    [[ -f "${JUDGE_DIR}/ollama_classifications.json" ]] || fail "Ollama classifications missing"
    ok "Ollama judge complete in $(( END - START ))s"

    if [[ -f "${JUDGE_DIR}/claude_rubric_scores.backup.json" ]]; then
        cp "${JUDGE_DIR}/claude_rubric_scores.backup.json"   "${JUDGE_DIR}/claude_rubric_scores.json"
        cp "${JUDGE_DIR}/claude_classifications.backup.json" "${JUDGE_DIR}/claude_classifications.json"
        ok "Claude backup restored"
    fi
fi

# ── STEP 4: Agreement Metrics ─────────────────────────────────────────────────
step "STEP 4 / 5 — INTER-JUDGE AGREEMENT METRICS"
info "Loading Claude and Ollama results, computing Krippendorff α, Spearman ρ, Cohen κ"

[[ -f "${JUDGE_DIR}/claude_rubric_scores.json" ]]     || fail "Claude rubric file missing — did Step 3a complete?"
[[ -f "${JUDGE_DIR}/claude_classifications.json" ]]   || fail "Claude classifications file missing"
[[ -f "${JUDGE_DIR}/ollama_rubric_scores.json" ]]     || fail "Ollama rubric file missing — did Step 3b complete?"
[[ -f "${JUDGE_DIR}/ollama_classifications.json" ]]   || fail "Ollama classifications file missing"

START=$(date +%s)
CLAUDE_RUBRIC="${JUDGE_DIR}/claude_rubric_scores.json" \
CLAUDE_CLASS="${JUDGE_DIR}/claude_classifications.json" \
OLLAMA_RUBRIC="${JUDGE_DIR}/ollama_rubric_scores.json" \
OLLAMA_CLASS="${JUDGE_DIR}/ollama_classifications.json" \
OUTPUT_JSON="${JUDGE_DIR}/agreement_metrics.json" \
OUTPUT_TXT="${JUDGE_DIR}/agreement_report.txt" \
JUDGE_DIR="${JUDGE_DIR}" \
python3 compute_agreement.py
END=$(date +%s)

[[ -f "${JUDGE_DIR}/agreement_metrics.json" ]] || fail "Agreement metrics not written"
[[ -f "${JUDGE_DIR}/agreement_report.txt" ]]   || fail "Agreement report not written"
ok "Agreement metrics complete in $(( END - START ))s"
ok "Report: ${JUDGE_DIR}/agreement_report.txt"

# ── STEP 5: Calibration Statistics ───────────────────────────────────────────
step "STEP 5 / 5 — CALIBRATION SIGNIFICANCE TESTS"
info "Wilcoxon rank-sum sensitivity analysis: N=400 vs all alternatives"

START=$(date +%s)
python3 calibration_stats.py
END=$(date +%s)

[[ -f "calibration_stats.json" ]] || fail "Calibration stats JSON not written"
ok "Calibration stats complete in $(( END - START ))s"

# ── Summary ───────────────────────────────────────────────────────────────────
step "PIPELINE COMPLETE"

echo -e "${BOLD}All outputs:${NC}"
echo ""
echo -e "  ${CYAN}Calibration${NC}"
echo -e "    ${RESULTS_DIR}/calibration/calibration_raw.json         — N=25…1000 × 5 seeds raw metrics"
echo -e "    calibration_stats.json                                   — Wilcoxon p-values & effect sizes"
echo ""
echo -e "  ${CYAN}Experiment results${NC}"
echo -e "    ${RESULTS_DIR}/persona_evaluations_round_a.json          — Raw stimuli responses (45 evals)"
echo -e "    ${RESULTS_DIR}/persona_evaluations_round_b.json          — Explained stimuli responses (45 evals)"
echo -e "    ${RESULTS_DIR}/persona_summaries.json                    — End-of-session summaries (9)"
echo -e "    ${RESULTS_DIR}/run_metadata.json                         — Reproducibility record"
echo ""
echo -e "  ${CYAN}Judge results${NC}"
echo -e "    ${JUDGE_DIR}/claude_rubric_scores.json                   — Claude facet scores (90 evals)"
echo -e "    ${JUDGE_DIR}/claude_classifications.json                 — Claude blind classifications"
echo -e "    ${JUDGE_DIR}/ollama_rubric_scores.json                   — Ollama facet scores (90 evals)"
echo -e "    ${JUDGE_DIR}/ollama_classifications.json                 — Ollama blind classifications"
echo ""
echo -e "  ${CYAN}Agreement & statistics${NC}"
echo -e "    ${JUDGE_DIR}/agreement_metrics.json                      — Krippendorff α, Spearman ρ, Cohen κ"
echo -e "    ${JUDGE_DIR}/agreement_report.txt                        — Human-readable report for paper"
echo ""

if [[ -f "${JUDGE_DIR}/agreement_report.txt" ]]; then
    echo -e "${BOLD}Agreement summary:${NC}"
    grep -E "Krippendorff|Spearman|Cohen|Role accuracy|Full persona" \
        "${JUDGE_DIR}/agreement_report.txt" | head -6 | \
        while IFS= read -r line; do echo "    ${line}"; done
fi

echo ""
echo -e "${GREEN}${BOLD}Done. $(date '+%Y-%m-%d %H:%M:%S')${NC}\n"