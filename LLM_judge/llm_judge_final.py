"""
llm_judge.py — SE4AI LLM-as-Judge Validation (v2 — two-round aware)

Reads the persona evaluation outputs produced by persona_experiment_v2.py and
runs two independent judges (Claude via Anthropic API + a local Ollama model)
to validate that each persona response is behaviorally consistent with its
defined GenderMag facets and role.

Loads both round files:
  results/persona_evaluations_round_a.json  — raw tree responses
  results/persona_evaluations_round_b.json  — explained tree responses

Judges ALL responses (both rounds) so you can compare behavioral consistency
between raw and explained conditions — e.g. does Tim's risk-tolerance show up
more clearly when he has an explanation to work with?

Two validation tasks per response:
  1. RUBRIC SCORING  — score each response on 5 GenderMag facets + role fit (1–5 each)
  2. BLIND CLASSIFICATION — given only the response, guess which of 9 personas wrote it

Outputs:
    judge_results/claude_rubric_scores.json       — Claude rubric scores (both rounds)
    judge_results/claude_classifications.json     — Claude blind classifications
    judge_results/ollama_rubric_scores.json       — Ollama rubric scores (both rounds)
    judge_results/ollama_classifications.json     — Ollama blind classifications
    judge_results/agreement_summary.json          — Krippendorff α, Spearman ρ, Cohen κ,
                                                    per-round descriptive stats
    judge_results/run_metadata.json               — full reproducibility record

Agreement metrics grounded in:
  - Krippendorff (2019) for ordinal facet scores
  - Cohen (1960) for blind classification (nominal)
  - Spearman rank correlation for preference ordering
  - Hoffman et al. (2023) for the rubric dimensions themselves

Requirements:
    pip install anthropic requests numpy scipy krippendorff --break-system-packages
"""

import json
import os
import time
import datetime
import statistics
from pathlib import Path

import anthropic
import requests
import numpy as np
from scipy import stats
from scipy.stats import spearmanr

try:
    import krippendorff
    HAS_KRIPPENDORFF = True
except ImportError:
    HAS_KRIPPENDORFF = False
    print("Warning: krippendorff package not found. Install with: pip install krippendorff")

# ── Config ────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL       = "claude-sonnet-4-5"

OLLAMA_BASE_URL    = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL       = os.environ.get("OLLAMA_MODEL", "llama3.1:70b")  # adjust to your model

RESULTS_DIR        = Path("results")
JUDGE_DIR          = Path("judge_results")
CALL_DELAY_S       = 2   # between judge API calls

# ── All 9 persona IDs for classification task ─────────────────────────────────
ALL_PERSONAS = [
    "PjM-Abi", "PjM-Pat", "PjM-Tim",
    "PdM-Abi", "PdM-Pat", "PdM-Tim",
    "SWE-Abi", "SWE-Pat", "SWE-Tim",
]

# ── Persona ground-truth definitions for rubric ───────────────────────────────
PERSONA_DEFINITIONS = {
    "PjM-Abi": {
        "role": "Project Manager", "tech_level": 1, "cognitive_profile": "Abi",
        "motivations": "task-oriented", "info_processing": "comprehensive",
        "self_efficacy": "low", "risk_attitude": "risk-averse",
        "learning_style": "process-oriented"
    },
    "PjM-Pat": {
        "role": "Project Manager", "tech_level": 1, "cognitive_profile": "Pat",
        "motivations": "task-oriented", "info_processing": "comprehensive",
        "self_efficacy": "medium", "risk_attitude": "risk-averse",
        "learning_style": "reflective-tinkerer"
    },
    "PjM-Tim": {
        "role": "Project Manager", "tech_level": 1, "cognitive_profile": "Tim",
        "motivations": "technology-oriented", "info_processing": "selective",
        "self_efficacy": "high", "risk_attitude": "risk-tolerant",
        "learning_style": "tinkerer"
    },
    "PdM-Abi": {
        "role": "Product Manager", "tech_level": 3, "cognitive_profile": "Abi",
        "motivations": "task-oriented", "info_processing": "comprehensive",
        "self_efficacy": "low", "risk_attitude": "risk-averse",
        "learning_style": "process-oriented"
    },
    "PdM-Pat": {
        "role": "Product Manager", "tech_level": 3, "cognitive_profile": "Pat",
        "motivations": "task-oriented", "info_processing": "comprehensive",
        "self_efficacy": "medium", "risk_attitude": "risk-averse",
        "learning_style": "reflective-tinkerer"
    },
    "PdM-Tim": {
        "role": "Product Manager", "tech_level": 3, "cognitive_profile": "Tim",
        "motivations": "technology-oriented", "info_processing": "selective",
        "self_efficacy": "high", "risk_attitude": "risk-tolerant",
        "learning_style": "tinkerer"
    },
    "SWE-Abi": {
        "role": "Software Engineer", "tech_level": 5, "cognitive_profile": "Abi",
        "motivations": "task-oriented", "info_processing": "comprehensive",
        "self_efficacy": "low", "risk_attitude": "risk-averse",
        "learning_style": "process-oriented"
    },
    "SWE-Pat": {
        "role": "Software Engineer", "tech_level": 5, "cognitive_profile": "Pat",
        "motivations": "task-oriented", "info_processing": "comprehensive",
        "self_efficacy": "medium", "risk_attitude": "risk-averse",
        "learning_style": "reflective-tinkerer"
    },
    "SWE-Tim": {
        "role": "Software Engineer", "tech_level": 5, "cognitive_profile": "Tim",
        "motivations": "technology-oriented", "info_processing": "selective",
        "self_efficacy": "high", "risk_attitude": "risk-tolerant",
        "learning_style": "tinkerer"
    },
}

# ── Judge system prompt ────────────────────────────────────────────────────────
JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for a software engineering research study.
You will be shown responses from LLM-simulated stakeholder personas evaluating
Pareto-frontier decision trees. Your job is to assess how behaviorally consistent
each response is with a specific GenderMag cognitive profile and software team role.

You must be objective and critical. A response that sounds professional but doesn't
reflect the specific cognitive style should receive low scores on the relevant facets.

Always respond ONLY with valid JSON, no extra text."""


def build_rubric_prompt(response_text: str, persona_id: str) -> str:
    """Prompt for rubric scoring task — judge knows the persona identity."""
    p = PERSONA_DEFINITIONS[persona_id]
    return f"""You are evaluating whether the following response is behaviorally consistent
with the persona described below.

=== PERSONA DEFINITION ===
Role: {p['role']} (technical level: {p['tech_level']}/5)
Cognitive Profile: {p['cognitive_profile']}
- Motivations: {p['motivations']} (task-focused vs technology-curious)
- Information Processing: {p['info_processing']} (reads everything first vs dives in)
- Computer Self-Efficacy: {p['self_efficacy']} (low=blames self, high=blames vendor)
- Risk Attitude: {p['risk_attitude']} (avoids unknown tech vs tries new things)
- Learning Style: {p['learning_style']} (follows steps vs tinkers freely)

=== RESPONSE TO EVALUATE ===
{response_text}

=== SCORING TASK ===
Score each dimension 1–5 where:
  1 = strongly inconsistent with persona definition
  3 = neutral / could be anyone
  5 = strongly consistent with persona definition

Also note: behavioral signals to look for:
- Role fit: Does technical language level match tech_level? Do priorities match the role?
- Info processing: Comprehensive = asks many broad questions before deciding. Selective = commits to first promising option, asks narrow follow-ups.
- Risk attitude: Risk-averse = explicitly favors stability/predictability. Risk-tolerant = comfortable with lower stability for higher accuracy.
- Self-efficacy: Low = hedging language ("I'm not sure", "I'd need help"). High = confident assertions, blames the tool not themselves.
- Learning style: Process-oriented = asks for step-by-step. Tinkerer = wants to explore freely.

Respond ONLY with this JSON:
{{
  "persona_id": "{persona_id}",
  "role_fit": <1-5>,
  "info_processing_fit": <1-5>,
  "risk_attitude_fit": <1-5>,
  "self_efficacy_fit": <1-5>,
  "learning_style_fit": <1-5>,
  "overall_consistency": <1-5>,
  "rationale": "<2-3 sentences explaining your scores, citing specific phrases from the response>"
}}"""


def build_classification_prompt(response_text: str) -> str:
    """Prompt for blind classification task — judge does NOT know the persona."""
    personas_list = "\n".join([
        f"  - {pid}: {PERSONA_DEFINITIONS[pid]['role']} / {PERSONA_DEFINITIONS[pid]['cognitive_profile']} profile (tech level {PERSONA_DEFINITIONS[pid]['tech_level']}/5)"
        for pid in ALL_PERSONAS
    ])
    return f"""You are given a response from ONE of 9 possible stakeholder personas.
Based ONLY on the response content — the language used, priorities expressed,
questions asked, and level of technical engagement — identify which persona wrote it.

=== THE 9 POSSIBLE PERSONAS ===
{personas_list}

Key distinguishing signals:
- Technical language level → narrows down the role (PjM=1, PdM=3, SWE=5)
- Risk language (stability preference vs accuracy preference) → Abi/Pat vs Tim
- Hedging/confidence language → Abi (low efficacy) vs Pat (medium) vs Tim (high)
- Breadth of questions → Abi/Pat (comprehensive) vs Tim (selective/focused)
- Curiosity about the technology itself → Tim only

=== RESPONSE TO CLASSIFY ===
{response_text}

Respond ONLY with this JSON:
{{
  "predicted_persona": "<one of the 9 persona IDs>",
  "predicted_role": "<Project Manager | Product Manager | Software Engineer>",
  "predicted_cognitive_profile": "<Abi | Pat | Tim>",
  "confidence": <1-5, where 1=guessing, 5=very confident>,
  "classification_rationale": "<cite 2-3 specific phrases that drove your prediction>"
}}"""


# ── API clients ────────────────────────────────────────────────────────────────

class ClaudeJudge:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.call_count = 0

    def call(self, user_prompt: str) -> str:
        if self.call_count > 0:
            time.sleep(CALL_DELAY_S)
        self.call_count += 1
        message = self.client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1000,
            system=JUDGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}]
        )
        return message.content[0].text.strip()


class OllamaJudge:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model    = model
        self.call_count = 0

    def call(self, user_prompt: str) -> str:
        if self.call_count > 0:
            time.sleep(CALL_DELAY_S)
        self.call_count += 1
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.0}  # deterministic for judge
        }
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=120
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()


def safe_parse_json(raw: str) -> dict:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = [l for l in cleaned.split("\n") if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        return {"parse_error": str(e), "raw": raw}


# ── Agreement metrics ──────────────────────────────────────────────────────────

def compute_agreement(claude_scores: list[dict], ollama_scores: list[dict]) -> dict:
    """
    Compute inter-judge reliability metrics between Claude and Ollama.

    Metrics:
      - Krippendorff's alpha (ordinal) for each facet score dimension
        [Krippendorff 2019; grounded in Artstein & Poesio 2008]
      - Spearman rho for overall_consistency ranking
        [Chen et al. 2025 MAJ-Eval uses this as primary metric]
      - Cohen's kappa for blind persona classification (nominal)
        [Cohen 1960; Judge's Verdict 2025]
      - Per-persona acceptance rate and mean clarity/sufficiency
        [Hoffman et al. 2023 Explanation Satisfaction Scale]
    """
    results = {}

    # ── 1. Krippendorff's alpha per facet (ordinal scale 1–5) ────────────────
    facet_dimensions = [
        "role_fit", "info_processing_fit", "risk_attitude_fit",
        "self_efficacy_fit", "learning_style_fit", "overall_consistency"
    ]

    kripp_results = {}
    for dim in facet_dimensions:
        c_vals = [s.get(dim) for s in claude_scores if dim in s and s[dim] is not None]
        o_vals = [s.get(dim) for s in ollama_scores if dim in s and s[dim] is not None]
        if len(c_vals) == len(o_vals) and len(c_vals) > 1:
            if HAS_KRIPPENDORFF:
                data = np.array([c_vals, o_vals], dtype=float)
                alpha = krippendorff.alpha(
                    reliability_data=data,
                    level_of_measurement="ordinal"
                )
                kripp_results[dim] = round(float(alpha), 4)
            else:
                # Fallback: Pearson r as approximation
                r, p = stats.pearsonr(c_vals, o_vals)
                kripp_results[dim] = {"pearson_r": round(r, 4), "p": round(p, 4)}
    results["krippendorff_alpha"] = kripp_results

    # ── 2. Spearman rho on overall_consistency ────────────────────────────────
    c_overall = [s.get("overall_consistency") for s in claude_scores
                 if s.get("overall_consistency") is not None]
    o_overall = [s.get("overall_consistency") for s in ollama_scores
                 if s.get("overall_consistency") is not None]
    if len(c_overall) == len(o_overall) and len(c_overall) > 1:
        rho, p_val = spearmanr(c_overall, o_overall)
        results["spearman_rho_overall"] = {"rho": round(rho, 4), "p": round(p_val, 4)}

    # ── 3. Cohen's kappa for blind classification ─────────────────────────────
    c_preds = [s.get("predicted_persona") for s in claude_scores
               if "predicted_persona" in s]
    o_preds = [s.get("predicted_persona") for s in ollama_scores
               if "predicted_persona" in s]
    if c_preds and o_preds and len(c_preds) == len(o_preds):
        results["classification_agreement"] = _cohen_kappa(c_preds, o_preds)

    return results


def _cohen_kappa(rater1: list, rater2: list) -> dict:
    """Compute Cohen's kappa for two lists of nominal labels."""
    labels = sorted(set(rater1 + rater2))
    label_index = {l: i for i, l in enumerate(labels)}
    n = len(rater1)
    k = len(labels)
    matrix = np.zeros((k, k), dtype=float)
    for a, b in zip(rater1, rater2):
        if a in label_index and b in label_index:
            matrix[label_index[a]][label_index[b]] += 1
    po = np.trace(matrix) / n
    row_sums = matrix.sum(axis=1) / n
    col_sums = matrix.sum(axis=0) / n
    pe = float(np.dot(row_sums, col_sums))
    kappa = (po - pe) / (1 - pe) if (1 - pe) != 0 else 0.0
    exact_match = sum(a == b for a, b in zip(rater1, rater2)) / n
    return {
        "cohen_kappa": round(kappa, 4),
        "exact_match_rate": round(exact_match, 4),
        "n_items": n,
        "interpretation": _kappa_label(kappa)
    }


def _kappa_label(k: float) -> str:
    if k >= 0.81: return "almost perfect"
    if k >= 0.61: return "substantial"
    if k >= 0.41: return "moderate"
    if k >= 0.21: return "fair"
    return "slight or poor"


def compute_descriptive_stats(evaluations: list[dict]) -> dict:
    """
    Compute per-persona and per-tree descriptive statistics from the
    persona_evaluations.json output.

    Measures follow Hoffman et al. (2023):
      - Acceptance rate  → human-AI performance dimension
      - Clarity mean     → Explanation Satisfaction Scale dimension 1
      - Sufficiency mean → Explanation Satisfaction Scale dimension 2
    """
    stats_out = {"by_persona": {}, "by_tree": {}, "by_role": {}, "by_cognitive_profile": {}}

    # Group by persona
    by_persona = {}
    for ev in evaluations:
        pid = ev["persona_id"]
        if pid not in by_persona:
            by_persona[pid] = []
        by_persona[pid].append(ev)

    for pid, evs in by_persona.items():
        parsed = [e["parsed_response"] for e in evs if "parse_error" not in e.get("parsed_response", {})]
        accepts   = [p["accepts_recommendation"] for p in parsed if "accepts_recommendation" in p]
        clarity   = [p["clarity_score"]     for p in parsed if "clarity_score" in p]
        suff      = [p["sufficiency_score"] for p in parsed if "sufficiency_score" in p]

        stats_out["by_persona"][pid] = {
            "acceptance_rate":      round(sum(accepts) / len(accepts), 3) if accepts else None,
            "clarity_mean":         round(statistics.mean(clarity), 3) if clarity else None,
            "clarity_sd":           round(statistics.stdev(clarity), 3) if len(clarity) > 1 else None,
            "sufficiency_mean":     round(statistics.mean(suff), 3) if suff else None,
            "sufficiency_sd":       round(statistics.stdev(suff), 3) if len(suff) > 1 else None,
            "n_evaluations":        len(evs),
        }

    # Group by tree
    by_tree = {}
    for ev in evaluations:
        tid = ev["tree_id"]
        if tid not in by_tree:
            by_tree[tid] = {"evaluations": [], "metrics": ev["tree_metrics"]}
        by_tree[tid]["evaluations"].append(ev)

    for tid, data in by_tree.items():
        parsed  = [e["parsed_response"] for e in data["evaluations"]
                   if "parse_error" not in e.get("parsed_response", {})]
        accepts = [p["accepts_recommendation"] for p in parsed if "accepts_recommendation" in p]
        clarity = [p["clarity_score"]     for p in parsed if "clarity_score" in p]
        suff    = [p["sufficiency_score"] for p in parsed if "sufficiency_score" in p]

        stats_out["by_tree"][tid] = {
            "tree_metrics":     data["metrics"],
            "acceptance_rate":  round(sum(accepts) / len(accepts), 3) if accepts else None,
            "clarity_mean":     round(statistics.mean(clarity), 3) if clarity else None,
            "sufficiency_mean": round(statistics.mean(suff), 3) if suff else None,
            "n_personas":       len(data["evaluations"]),
        }

    # Group by role and cognitive profile
    for group_key in ["role", "cognitive_profile"]:
        groups = {}
        for ev in evaluations:
            g = ev[group_key]
            if g not in groups:
                groups[g] = []
            groups[g].append(ev)
        for g, evs in groups.items():
            parsed  = [e["parsed_response"] for e in evs
                       if "parse_error" not in e.get("parsed_response", {})]
            accepts = [p["accepts_recommendation"] for p in parsed if "accepts_recommendation" in p]
            clarity = [p["clarity_score"]     for p in parsed if "clarity_score" in p]
            suff    = [p["sufficiency_score"] for p in parsed if "sufficiency_score" in p]
            target  = "by_role" if group_key == "role" else "by_cognitive_profile"
            stats_out[target][g] = {
                "acceptance_rate":  round(sum(accepts) / len(accepts), 3) if accepts else None,
                "clarity_mean":     round(statistics.mean(clarity), 3) if clarity else None,
                "sufficiency_mean": round(statistics.mean(suff), 3) if suff else None,
                "n_evaluations":    len(evs),
            }

    # Group by round (raw vs explained) — key comparison for RQ1
    for round_name in ["raw", "explained"]:
        round_evs = [ev for ev in evaluations if ev.get("round") == round_name]
        if not round_evs:
            continue
        parsed  = [e["parsed_response"] for e in round_evs
                   if "parse_error" not in e.get("parsed_response", {})]
        accepts = [p["accepts_recommendation"] for p in parsed if "accepts_recommendation" in p]
        clarity = [p["clarity_score"]     for p in parsed if "clarity_score" in p]
        suff    = [p["sufficiency_score"] for p in parsed if "sufficiency_score" in p]
        stats_out.setdefault("by_round", {})[round_name] = {
            "acceptance_rate":  round(sum(accepts) / len(accepts), 3) if accepts else None,
            "clarity_mean":     round(statistics.mean(clarity), 3) if clarity else None,
            "clarity_sd":       round(statistics.stdev(clarity), 3) if len(clarity) > 1 else None,
            "sufficiency_mean": round(statistics.mean(suff), 3) if suff else None,
            "sufficiency_sd":   round(statistics.stdev(suff), 3) if len(suff) > 1 else None,
            "n_evaluations":    len(round_evs),
        }

    return stats_out


# ── Main judge runner ──────────────────────────────────────────────────────────

def run_judge(use_ollama: bool = True, use_claude: bool = True):
    JUDGE_DIR.mkdir(exist_ok=True)

    # Load both round files from persona_experiment_v2.py
    # Falls back to single-file v1 output if v2 files not found
    path_a  = RESULTS_DIR / "persona_evaluations_round_a.json"
    path_b  = RESULTS_DIR / "persona_evaluations_round_b.json"
    path_v1 = RESULTS_DIR / "persona_evaluations.json"

    if path_a.exists() and path_b.exists():
        with open(path_a) as f:
            evals_a = json.load(f)
        with open(path_b) as f:
            evals_b = json.load(f)
        for ev in evals_a:
            ev.setdefault("round", "raw")
        for ev in evals_b:
            ev.setdefault("round", "explained")
        evaluations = evals_a + evals_b
        print(f"Loaded two-round outputs: {len(evals_a)} raw + {len(evals_b)} explained = {len(evaluations)} total")
    elif path_v1.exists():
        with open(path_v1) as f:
            evaluations = json.load(f)
        for ev in evaluations:
            ev.setdefault("round", "explained")
        print(f"Loaded single-round output (v1 fallback): {len(evaluations)} evaluations")
    else:
        print(f"ERROR: No evaluation files found in {RESULTS_DIR}/")
        print(f"  Expected: persona_evaluations_round_a.json + persona_evaluations_round_b.json")
        return

    print(f"Loaded {len(evaluations)} persona evaluations total")

    # Init judges
    if not ANTHROPIC_API_KEY:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        return

    claude = ClaudeJudge(ANTHROPIC_API_KEY) if use_claude else None
    ollama = OllamaJudge(OLLAMA_BASE_URL, OLLAMA_MODEL) if use_ollama else None

    if not use_claude and not use_ollama:
        print("ERROR: At least one judge must be enabled (USE_CLAUDE or USE_OLLAMA)")
        return
    if use_claude:
        print("Judge 1: Claude")
    if use_ollama:
        print(f"Judge 2: Ollama ({OLLAMA_MODEL})")

    claude_rubric_scores  = []
    claude_classifications = []
    ollama_rubric_scores  = []
    ollama_classifications = []

    total = len(evaluations)
    for i, ev in enumerate(evaluations):
        persona_id    = ev["persona_id"]
        response_text = ev.get("raw_response", "")

        if not response_text or len(response_text) < 20:
            print(f"  Skipping {persona_id} tree {ev['tree_id']} — empty response")
            continue

        print(f"  [{i+1}/{total}] Judging {persona_id} / {ev['tree_id']}...")

        # ── Claude: rubric scoring + classification ──────────────────────────
        if use_claude:
            rubric_prompt = build_rubric_prompt(response_text, persona_id)
            raw_rubric    = claude.call(rubric_prompt)
            parsed_rubric = safe_parse_json(raw_rubric)
            parsed_rubric["eval_id"]   = f"{persona_id}__{ev['tree_id']}__{ ev.get('round','unknown')}"
            parsed_rubric["tree_id"]   = ev["tree_id"]
            parsed_rubric["judge"]     = "claude"
            parsed_rubric["timestamp"] = datetime.datetime.utcnow().isoformat()
            claude_rubric_scores.append(parsed_rubric)

            class_prompt    = build_classification_prompt(response_text)
            raw_class       = claude.call(class_prompt)
            parsed_class    = safe_parse_json(raw_class)
            parsed_class["eval_id"]      = f"{persona_id}__{ev['tree_id']}__{ ev.get('round','unknown')}"
            parsed_class["true_persona"] = persona_id
            parsed_class["correct"]      = parsed_class.get("predicted_persona") == persona_id
            parsed_class["judge"]        = "claude"
            parsed_class["timestamp"]    = datetime.datetime.utcnow().isoformat()
            claude_classifications.append(parsed_class)

        # ── Ollama: same two tasks (builds its own prompts independently) ─────
        if ollama:
            try:
                o_rubric_prompt = build_rubric_prompt(response_text, persona_id)
                o_class_prompt  = build_classification_prompt(response_text)
                eval_id         = f"{persona_id}__{ev['tree_id']}__{ev.get('round','unknown')}"

                raw_o_rubric    = ollama.call(o_rubric_prompt)
                parsed_o_rubric = safe_parse_json(raw_o_rubric)
                parsed_o_rubric["eval_id"]   = eval_id
                parsed_o_rubric["tree_id"]   = ev["tree_id"]
                parsed_o_rubric["judge"]     = "ollama"
                parsed_o_rubric["timestamp"] = datetime.datetime.utcnow().isoformat()
                ollama_rubric_scores.append(parsed_o_rubric)

                raw_o_class    = ollama.call(o_class_prompt)
                parsed_o_class = safe_parse_json(raw_o_class)
                parsed_o_class["eval_id"]      = eval_id
                parsed_o_class["true_persona"] = persona_id
                parsed_o_class["correct"]      = parsed_o_class.get("predicted_persona") == persona_id
                parsed_o_class["judge"]        = "ollama"
                parsed_o_class["timestamp"]    = datetime.datetime.utcnow().isoformat()
                ollama_classifications.append(parsed_o_class)

            except Exception as e:
                print(f"    Ollama error: {e} — skipping Ollama for this item")

    # ── Save raw judge outputs ────────────────────────────────────────────────
    def save(data, path):
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved → {path}")

    save(claude_rubric_scores,   JUDGE_DIR / "claude_rubric_scores.json")
    save(claude_classifications, JUDGE_DIR / "claude_classifications.json")
    if ollama:
        save(ollama_rubric_scores,   JUDGE_DIR / "ollama_rubric_scores.json")
        save(ollama_classifications, JUDGE_DIR / "ollama_classifications.json")

    # ── Compute agreement metrics ─────────────────────────────────────────────
    agreement = {}
    if ollama and ollama_rubric_scores:
        # Filter to matching eval_ids
        claude_ids = {s["eval_id"]: s for s in claude_rubric_scores if "eval_id" in s}
        ollama_ids = {s["eval_id"]: s for s in ollama_rubric_scores  if "eval_id" in s}
        shared_ids = sorted(set(claude_ids) & set(ollama_ids))

        c_matched = [claude_ids[i] for i in shared_ids]
        o_matched = [ollama_ids[i] for i in shared_ids]

        c_class_ids = {s["eval_id"]: s for s in claude_classifications if "eval_id" in s}
        o_class_ids = {s["eval_id"]: s for s in ollama_classifications if "eval_id" in s}
        shared_class = sorted(set(c_class_ids) & set(o_class_ids))
        c_class_matched = [c_class_ids[i] for i in shared_class]
        o_class_matched = [o_class_ids[i] for i in shared_class]

        agreement = compute_agreement(c_matched, o_matched)

        # Add classification kappa separately
        if c_class_matched and o_class_matched:
            c_preds = [s.get("predicted_persona") for s in c_class_matched]
            o_preds = [s.get("predicted_persona") for s in o_class_matched]
            agreement["classification_kappa"] = _cohen_kappa(c_preds, o_preds)

    # ── Descriptive stats from persona experiment outputs ─────────────────────
    desc_stats = compute_descriptive_stats(evaluations)

    # ── Per-judge classification accuracy ─────────────────────────────────────
    def classification_accuracy(classifications: list[dict]) -> dict:
        if not classifications:
            return {}
        total_n   = len(classifications)
        correct_n = sum(1 for c in classifications if c.get("correct"))
        role_correct = sum(
            1 for c in classifications
            if c.get("predicted_role") == PERSONA_DEFINITIONS.get(
                c.get("true_persona", ""), {}
            ).get("role")
        )
        profile_correct = sum(
            1 for c in classifications
            if c.get("predicted_cognitive_profile") == PERSONA_DEFINITIONS.get(
                c.get("true_persona", ""), {}
            ).get("cognitive_profile")
        )
        return {
            "full_persona_accuracy":    round(correct_n / total_n, 3),
            "role_accuracy":            round(role_correct / total_n, 3),
            "cognitive_profile_accuracy": round(profile_correct / total_n, 3),
            "n": total_n,
            "chance_baseline_full":     round(1 / 9, 3),
            "chance_baseline_role":     round(1 / 3, 3),
            "chance_baseline_profile":  round(1 / 3, 3),
        }

    summary = {
        "inter_judge_agreement":     agreement,
        "descriptive_stats":         desc_stats,
        "claude_classification_accuracy": classification_accuracy(claude_classifications),
        "ollama_classification_accuracy": classification_accuracy(ollama_classifications) if ollama else {},
        "total_evaluations_judged":  len(claude_rubric_scores),
        "methodology_note": (
            "Inter-judge reliability measured via Krippendorff's alpha (ordinal facet scores, "
            "Krippendorff 2019), Spearman rho (ranking correlation, Chen et al. 2025), and "
            "Cohen's kappa (blind persona classification, Cohen 1960). "
            "Descriptive statistics follow Hoffman et al. (2023) XAI evaluation framework: "
            "acceptance rate (human-AI performance), clarity score and sufficiency score "
            "(Explanation Satisfaction Scale)."
        )
    }

    save(summary, JUDGE_DIR / "agreement_summary.json")

    metadata = {
        "claude_judge_model":  CLAUDE_MODEL if use_claude else "not used",
        "ollama_judge_model":  OLLAMA_MODEL if use_ollama else "not used",
        "ollama_url":          OLLAMA_BASE_URL,
        "judge_temperature":   0.0,
        "input_evaluations":   len(evaluations),
        "judged_evaluations":  max(len(claude_rubric_scores), len(ollama_rubric_scores)),
        "total_claude_calls":  claude.call_count if claude else 0,
        "total_ollama_calls":  ollama.call_count if ollama else 0,
        "run_date":            datetime.datetime.utcnow().isoformat(),
        "citations": {
            "rubric_dimensions":     "Hoffman et al. (2023). Measures for explainable AI. Frontiers in Computer Science, 5, 1096257.",
            "krippendorff_alpha":    "Krippendorff, K. (2019). Content Analysis (4th Ed.). SAGE.",
            "spearman_rho":          "Chen et al. (2025). Multi-Agent-as-Judge. arXiv:2507.21028.",
            "cohen_kappa":           "Cohen, J. (1960). Educational and Psychological Measurement, 20, 37-46.",
            "judge_paradigm":        "Zheng et al. (2023). Judging LLM-as-a-Judge. NeurIPS 36.",
            "persona_validation":    "Schuller et al. (2024). Generating personas using LLMs. CHI EA."
        }
    }
    save(metadata, JUDGE_DIR / "run_metadata.json")

    print(f"\n✅ Judge run complete!")
    if claude:
        print(f"   Claude calls: {claude.call_count}")
    if ollama:
        print(f"   Ollama calls: {ollama.call_count}")
    print(f"   Results in: {JUDGE_DIR}/")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    use_ollama = os.environ.get("USE_OLLAMA", "true").lower() == "true"
    use_claude = os.environ.get("USE_CLAUDE", "true").lower() == "true"

    if use_claude and not ANTHROPIC_API_KEY:
        print("ERROR: USE_CLAUDE=true but ANTHROPIC_API_KEY is not set")
        print("  export ANTHROPIC_API_KEY=your_key_here")
        print("  Or run Ollama-only: export USE_CLAUDE=false")
        exit(1)

    print(f"Judge config: Claude={use_claude}, Ollama={use_ollama}")
    run_judge(use_ollama=use_ollama, use_claude=use_claude)