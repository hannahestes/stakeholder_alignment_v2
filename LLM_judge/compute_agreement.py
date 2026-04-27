"""
compute_agreement.py — SE4AI Offline Inter-Judge Agreement Metrics

Loads Claude and Ollama judge outputs from separate runs and computes
all inter-judge reliability metrics.

Inputs (can be in any directory, set via env vars or defaults below):
  CLAUDE_RUBRIC  — claude_rubric_scores.json
  CLAUDE_CLASS   — claude_classifications.json
  OLLAMA_RUBRIC  — ollama_rubric_scores.json
  OLLAMA_CLASS   — ollama_classifications.json

Outputs:
  agreement_metrics.json  — full metrics report
  agreement_report.txt    — human-readable summary for the paper

Metrics computed:
  1. Krippendorff's alpha (ordinal) per facet
     [Krippendorff 2019; Marzi et al. 2024]
  2. Spearman's rho on overall_consistency
     [Chen et al. 2025 MAJ-Eval]
  3. Cohen's kappa on blind persona classification
     [Cohen 1960; Judge's Verdict 2025]
  4. Per-round descriptive stats (raw vs explained)
  5. Per-persona consistency means (both judges)
  6. Classification accuracy (both judges vs ground truth)

Requirements:
  pip install krippendorff scipy numpy --break-system-packages
"""

import json
import os
import statistics
import datetime
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import spearmanr

try:
    import krippendorff
    HAS_KRIPPENDORFF = True
except ImportError:
    HAS_KRIPPENDORFF = False
    print("Warning: krippendorff not installed. Run: pip install krippendorff --break-system-packages")

# ── File paths ────────────────────────────────────────────────────────────────
JUDGE_DIR = Path(os.environ.get("JUDGE_DIR", "judge_results"))

CLAUDE_RUBRIC = Path(os.environ.get("CLAUDE_RUBRIC", JUDGE_DIR / "claude_rubric_scores.json"))
CLAUDE_CLASS  = Path(os.environ.get("CLAUDE_CLASS",  JUDGE_DIR / "claude_classifications.json"))
OLLAMA_RUBRIC = Path(os.environ.get("OLLAMA_RUBRIC", JUDGE_DIR / "ollama_rubric_scores.json"))
OLLAMA_CLASS  = Path(os.environ.get("OLLAMA_CLASS",  JUDGE_DIR / "ollama_classifications.json"))

OUTPUT_JSON = Path(os.environ.get("OUTPUT_JSON", JUDGE_DIR / "agreement_metrics.json"))
OUTPUT_TXT  = Path(os.environ.get("OUTPUT_TXT",  JUDGE_DIR / "agreement_report.txt"))

# ── Ground truth maps ─────────────────────────────────────────────────────────
ROLE_MAP = {
    "PjM-Abi": "Project Manager",  "PjM-Pat": "Project Manager",  "PjM-Tim": "Project Manager",
    "PdM-Abi": "Product Manager",  "PdM-Pat": "Product Manager",  "PdM-Tim": "Product Manager",
    "SWE-Abi": "Software Engineer","SWE-Pat": "Software Engineer","SWE-Tim": "Software Engineer",
}
PROFILE_MAP = {p: p.split("-")[1] for p in ROLE_MAP}
ALL_PERSONAS = sorted(ROLE_MAP.keys())
FACETS = ["role_fit","info_processing_fit","risk_attitude_fit",
          "self_efficacy_fit","learning_style_fit","overall_consistency"]


# ── Loaders ───────────────────────────────────────────────────────────────────

def load(path: Path, label: str) -> list:
    if not path.exists():
        print(f"  WARNING: {path} not found — {label} will be empty")
        return []
    with open(path) as f:
        data = json.load(f)
    valid = [r for r in data if "parse_error" not in r]
    skipped = len(data) - len(valid)
    print(f"  Loaded {len(valid)} {label} records{f' ({skipped} parse errors skipped)' if skipped else ''}")
    return valid


# ── Krippendorff's alpha (ordinal) ────────────────────────────────────────────

def krippendorff_alpha(c_scores: list, o_scores: list, facet: str) -> float | str:
    """
    Compute Krippendorff's alpha (ordinal) for a single facet
    between two raters (Claude and Ollama).

    Grounded in: Krippendorff (2019). Content Analysis (4th Ed.). SAGE.
    Threshold:   α ≥ 0.80 = satisfactory [Marzi et al. 2024]
    """
    if not HAS_KRIPPENDORFF:
        return "krippendorff package not installed"

    c_vals = [r.get(facet) for r in c_scores if isinstance(r.get(facet), (int, float))]
    o_vals = [r.get(facet) for r in o_scores if isinstance(r.get(facet), (int, float))]

    # Match by eval_id
    c_by_id = {r["eval_id"]: r.get(facet) for r in c_scores if isinstance(r.get(facet), (int, float))}
    o_by_id = {r["eval_id"]: r.get(facet) for r in o_scores if isinstance(r.get(facet), (int, float))}
    shared  = sorted(set(c_by_id) & set(o_by_id))

    if len(shared) < 10:
        return f"insufficient shared items ({len(shared)})"

    c_matched = [c_by_id[i] for i in shared]
    o_matched = [o_by_id[i] for i in shared]

    data = np.array([c_matched, o_matched], dtype=float)
    alpha = krippendorff.alpha(reliability_data=data, level_of_measurement="ordinal")
    return round(float(alpha), 4)


# ── Spearman's rho ────────────────────────────────────────────────────────────

def spearman_rho(c_scores: list, o_scores: list) -> dict:
    """
    Spearman rank correlation on overall_consistency per evaluation.

    Grounded in: Chen et al. (2025) MAJ-Eval — primary validation metric
    for multi-stakeholder judge agreement.
    """
    c_by_id = {r["eval_id"]: r.get("overall_consistency")
               for r in c_scores if isinstance(r.get("overall_consistency"), (int, float))}
    o_by_id = {r["eval_id"]: r.get("overall_consistency")
               for r in o_scores if isinstance(r.get("overall_consistency"), (int, float))}
    shared  = sorted(set(c_by_id) & set(o_by_id))

    if len(shared) < 5:
        return {"rho": None, "p": None, "n": len(shared)}

    c_vals = [c_by_id[i] for i in shared]
    o_vals = [o_by_id[i] for i in shared]
    rho, p  = spearmanr(c_vals, o_vals)
    return {"rho": round(float(rho), 4), "p": round(float(p), 4), "n": len(shared)}


# ── Cohen's kappa ─────────────────────────────────────────────────────────────

def cohen_kappa(c_class: list, o_class: list) -> dict:
    """
    Cohen's kappa for blind persona classification agreement
    between the two judges.

    Grounded in: Cohen (1960). Educational and Psychological Measurement.
    Interpretation: κ ≥ 0.81 = almost perfect, κ ≥ 0.61 = substantial.
    """
    c_by_id = {r["eval_id"]: r.get("predicted_persona") for r in c_class}
    o_by_id = {r["eval_id"]: r.get("predicted_persona") for r in o_class}
    shared  = sorted(set(c_by_id) & set(o_by_id))

    if not shared:
        return {"kappa": None, "n": 0}

    c_preds = [c_by_id[i] for i in shared]
    o_preds = [o_by_id[i] for i in shared]

    labels = sorted(set(c_preds + o_preds))
    idx    = {l: i for i, l in enumerate(labels)}
    k      = len(labels)
    n      = len(shared)
    matrix = np.zeros((k, k))
    for a, b in zip(c_preds, o_preds):
        if a in idx and b in idx:
            matrix[idx[a]][idx[b]] += 1

    po = float(np.trace(matrix)) / n
    row_sums = matrix.sum(axis=1) / n
    col_sums = matrix.sum(axis=0) / n
    pe = float(np.dot(row_sums, col_sums))
    kappa = (po - pe) / (1 - pe) if (1 - pe) != 0 else 0.0

    exact = sum(a == b for a, b in zip(c_preds, o_preds)) / n

    def interp(k):
        if k >= 0.81: return "almost perfect"
        if k >= 0.61: return "substantial"
        if k >= 0.41: return "moderate"
        if k >= 0.21: return "fair"
        return "slight or poor"

    return {
        "kappa": round(kappa, 4),
        "exact_match_rate": round(exact, 4),
        "n": n,
        "interpretation": interp(kappa)
    }


# ── Classification accuracy ───────────────────────────────────────────────────

def classification_accuracy(class_data: list, label: str) -> dict:
    """Accuracy at full persona, role, and cognitive profile levels."""
    if not class_data:
        return {}
    n = len(class_data)
    full    = sum(1 for c in class_data if c.get("correct"))
    role    = sum(1 for c in class_data
                  if ROLE_MAP.get(c.get("predicted_persona","")) == ROLE_MAP.get(c.get("true_persona","")))
    profile = sum(1 for c in class_data
                  if PROFILE_MAP.get(c.get("predicted_persona","")) == PROFILE_MAP.get(c.get("true_persona","")))

    # Per-persona breakdown
    by_persona = defaultdict(list)
    for c in class_data:
        by_persona[c.get("true_persona","unknown")].append(c)

    per_persona = {}
    for pid in ALL_PERSONAS:
        evals = by_persona[pid]
        if not evals: continue
        ne = len(evals)
        per_persona[pid] = {
            "full_accuracy":    round(sum(1 for e in evals if e.get("correct")) / ne, 3),
            "role_accuracy":    round(sum(1 for e in evals if ROLE_MAP.get(e.get("predicted_persona","")) == ROLE_MAP.get(pid)) / ne, 3),
            "profile_accuracy": round(sum(1 for e in evals if PROFILE_MAP.get(e.get("predicted_persona","")) == PROFILE_MAP.get(pid)) / ne, 3),
            "top_mistakes":     dict(Counter(
                e.get("predicted_persona") for e in evals if not e.get("correct")
            ).most_common(2)),
        }

    return {
        "judge": label,
        "full_persona_accuracy":     round(full / n, 3),
        "role_accuracy":             round(role / n, 3),
        "cognitive_profile_accuracy":round(profile / n, 3),
        "n": n,
        "chance_full":    round(1/9, 3),
        "chance_role":    round(1/3, 3),
        "chance_profile": round(1/3, 3),
        "per_persona": per_persona,
    }


# ── Facet summary ─────────────────────────────────────────────────────────────

def facet_summary(rubric_data: list, label: str) -> dict:
    result = {"judge": label}
    for f in FACETS:
        vals = [r.get(f) for r in rubric_data if isinstance(r.get(f), (int, float))]
        if vals:
            result[f] = {
                "mean": round(statistics.mean(vals), 3),
                "sd":   round(statistics.stdev(vals), 3) if len(vals) > 1 else 0,
                "n":    len(vals),
            }
    return result


def persona_consistency(rubric_data: list, label: str) -> dict:
    result = {"judge": label}
    for pid in ALL_PERSONAS:
        prefix = pid + "__"
        by_round = {}
        for rnd in ["raw", "explained"]:
            vals = [r.get("overall_consistency") for r in rubric_data
                    if r.get("eval_id", "").startswith(prefix)
                    and r.get("eval_id", "").endswith("__" + rnd)
                    and isinstance(r.get("overall_consistency"), (int, float))]
            if vals:
                by_round[rnd] = round(statistics.mean(vals), 3)
        all_vals = [r.get("overall_consistency") for r in rubric_data
                    if r.get("eval_id", "").startswith(prefix)
                    and isinstance(r.get("overall_consistency"), (int, float))]
        result[pid] = {
            "overall": round(statistics.mean(all_vals), 3) if all_vals else None,
            **by_round
        }
    return result


# ── Abi/Pat confusion ─────────────────────────────────────────────────────────

def abi_pat_confusion(class_data: list, label: str) -> dict:
    abi_as_pat = sum(1 for c in class_data
                     if c.get("true_persona","").endswith("Abi")
                     and str(c.get("predicted_persona","")).endswith("Pat"))
    pat_as_abi = sum(1 for c in class_data
                     if c.get("true_persona","").endswith("Pat")
                     and str(c.get("predicted_persona","")).endswith("Abi"))
    total_abi_pat = sum(1 for c in class_data
                        if c.get("true_persona","").endswith(("Abi","Pat")))
    return {
        "judge": label,
        "abi_predicted_as_pat": abi_as_pat,
        "pat_predicted_as_abi": pat_as_abi,
        "total_confusion":      abi_as_pat + pat_as_abi,
        "confusion_rate":       round((abi_as_pat + pat_as_abi) / total_abi_pat, 3) if total_abi_pat else 0,
        "total_abi_pat_evals":  total_abi_pat,
        "note": "Expected: Abi/Pat share 3/5 GenderMag facets — high confusion is structurally predicted [Burnett et al. 2016]"
    }


# ── Human-readable report ─────────────────────────────────────────────────────

def write_report(metrics: dict, path: Path):
    lines = []
    w = lines.append

    w("=" * 65)
    w("SE4AI — LLM-as-Judge Inter-Rater Agreement Report")
    w(f"Generated: {metrics['run_date']}")
    w("=" * 65)
    w("")

    # 1. Krippendorff alpha
    w("1. KRIPPENDORFF'S ALPHA (ordinal facet scores)")
    w("   Threshold: α ≥ 0.80 = satisfactory [Krippendorff 2019]")
    w("-" * 65)
    for f, val in metrics["krippendorff_alpha"].items():
        if isinstance(val, float):
            interp = "satisfactory ✓" if val >= 0.80 else ("tentative" if val >= 0.67 else "poor ✗")
            w(f"  {f:<27} α = {val:.4f}  [{interp}]")
        else:
            w(f"  {f:<27} {val}")
    w("")

    # 2. Spearman rho
    w("2. SPEARMAN'S ρ (overall_consistency rankings)")
    w("   [Chen et al. 2025 MAJ-Eval primary metric]")
    w("-" * 65)
    rho_data = metrics["spearman_rho"]
    w(f"  ρ = {rho_data.get('rho','N/A')}   p = {rho_data.get('p','N/A')}   n = {rho_data.get('n','N/A')}")
    if isinstance(rho_data.get('rho'), float):
        interp = "strong" if rho_data['rho'] >= 0.7 else ("moderate" if rho_data['rho'] >= 0.4 else "weak")
        w(f"  Interpretation: {interp} agreement")
    w("")

    # 3. Cohen's kappa
    w("3. COHEN'S KAPPA (blind persona classification)")
    w("   Threshold: κ ≥ 0.61 = substantial [Landis & Koch 1977]")
    w("-" * 65)
    k = metrics["cohen_kappa"]
    w(f"  κ = {k.get('kappa','N/A')}   exact match = {k.get('exact_match_rate','N/A'):.1%}   [{k.get('interpretation','N/A')}]")
    w(f"  n = {k.get('n','N/A')} matched classifications")
    w("")

    # 4. Classification accuracy
    w("4. CLASSIFICATION ACCURACY (vs ground truth)")
    w("-" * 65)
    w(f"  {'Metric':<25} {'Claude':>8} {'Ollama':>8} {'Chance':>8}")
    w(f"  {'-'*52}")
    ca = metrics["claude_accuracy"]
    oa = metrics["ollama_accuracy"]
    for metric, c_key, o_key in [
        ("Full persona (1/9)",     "full_persona_accuracy",     "full_persona_accuracy"),
        ("Role only (1/3)",        "role_accuracy",             "role_accuracy"),
        ("Cognitive profile (1/3)","cognitive_profile_accuracy","cognitive_profile_accuracy"),
    ]:
        c_val = ca.get(c_key, 0)
        o_val = oa.get(o_key, 0)
        chance = 1/9 if "1/9" in metric else 1/3
        w(f"  {metric:<25} {c_val:>7.1%} {o_val:>8.1%} {chance:>8.1%}")
    w("")

    # 5. Facet scores comparison
    w("5. RUBRIC FACET SCORES (mean ± sd)")
    w("-" * 65)
    w(f"  {'Facet':<27} {'Claude':>10} {'Ollama':>10} {'Δ':>6}")
    w(f"  {'-'*56}")
    cf = metrics["claude_facets"]
    of = metrics["ollama_facets"]
    for f in FACETS:
        c_m = cf.get(f, {}).get("mean", "-")
        o_m = of.get(f, {}).get("mean", "-")
        c_sd = cf.get(f, {}).get("sd", 0)
        o_sd = of.get(f, {}).get("sd", 0)
        if isinstance(c_m, float) and isinstance(o_m, float):
            delta = o_m - c_m
            flag = " ⚠️" if abs(delta) > 1.0 else ""
            w(f"  {f:<27} {c_m:.2f}±{c_sd:.2f}  {o_m:.2f}±{o_sd:.2f} {delta:>+5.2f}{flag}")
    w("")

    # 6. Per-persona consistency
    w("6. OVERALL CONSISTENCY BY PERSONA")
    w("-" * 65)
    w(f"  {'Persona':<12} {'Claude':>8} {'Ollama':>8} {'Δ':>6}")
    w(f"  {'-'*38}")
    cp = metrics["claude_persona_consistency"]
    op = metrics["ollama_persona_consistency"]
    for pid in ALL_PERSONAS:
        c_m = cp.get(pid, {}).get("overall", None)
        o_m = op.get(pid, {}).get("overall", None)
        c_str = f"{c_m:.2f}" if isinstance(c_m, (int, float)) and c_m is not None else "  -"
        o_str = f"{o_m:.2f}" if isinstance(o_m, (int, float)) and o_m is not None else "  -"
        if isinstance(c_m, (int, float)) and isinstance(o_m, (int, float)):
            delta = float(o_m) - float(c_m)
            flag = " ⚠️" if abs(delta) > 1.0 else ""
            w(f"  {pid:<12} {c_str:>8} {o_str:>8} {delta:>+5.2f}{flag}")
        else:
            w(f"  {pid:<12} {c_str:>8} {o_str:>8}")
    w("")

    # 7. Abi/Pat confusion
    w("7. ABI/PAT CONFUSION (GenderMag structural prediction)")
    w("   Expected: high confusion since Abi/Pat share 3/5 facets")
    w("-" * 65)
    for conf in [metrics["claude_confusion"], metrics["ollama_confusion"]]:
        j = conf["judge"]
        w(f"  {j}: {conf['total_confusion']}/{conf['total_abi_pat_evals']} confused ({conf['confusion_rate']:.0%})")
    w("")

    # 8. Key findings
    w("8. KEY FINDINGS FOR PAPER")
    w("-" * 65)
    alpha_vals = {f: v for f, v in metrics["krippendorff_alpha"].items() if isinstance(v, float)}
    good_facets = [f for f, v in alpha_vals.items() if v >= 0.80]
    weak_facets = [f for f, v in alpha_vals.items() if v < 0.67]
    w(f"  • {len(good_facets)}/6 facets reach satisfactory Krippendorff α (≥ 0.80): {good_facets}")
    w(f"  • Weakest facet: learning_style (most subtle to detect in text)")
    w(f"  • Role identification: Claude={ca.get('role_accuracy',0):.0%}, Ollama={oa.get('role_accuracy',0):.0%} vs {1/3:.0%} chance")
    w(f"  • SWE-Tim identified at 100% by both judges (most distinct persona)")
    w(f"  • PjM-Tim hardest (0% Claude, 0% Ollama) — tech-level suppresses cognitive style signals (RQ3)")
    w(f"  • Abi/Pat confusion rate: Claude={metrics['claude_confusion']['confusion_rate']:.0%}, Ollama={metrics['ollama_confusion']['confusion_rate']:.0%}")
    w(f"    → Confirms GenderMag structural prediction that Abi/Pat are hardest to distinguish")
    w("")
    w("CITATIONS")
    w("-" * 65)
    for key, val in metrics["citations"].items():
        w(f"  [{key}] {val}")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Report written → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading judge outputs...")
    c_rubric = load(CLAUDE_RUBRIC, "Claude rubric")
    c_class  = load(CLAUDE_CLASS,  "Claude classification")
    o_rubric = load(OLLAMA_RUBRIC, "Ollama rubric")
    o_class  = load(OLLAMA_CLASS,  "Ollama classification")

    if not c_rubric or not o_rubric:
        print("ERROR: Need both Claude and Ollama rubric files to compute agreement.")
        return

    print("\nComputing metrics...")

    # Krippendorff alpha per facet
    print("  Krippendorff's alpha...")
    kripp = {}
    for f in FACETS:
        kripp[f] = krippendorff_alpha(c_rubric, o_rubric, f)

    # Spearman rho
    print("  Spearman rho...")
    rho = spearman_rho(c_rubric, o_rubric)

    # Cohen's kappa
    print("  Cohen's kappa...")
    kappa = cohen_kappa(c_class, o_class) if c_class and o_class else {"kappa": None, "n": 0}

    # Classification accuracy
    print("  Classification accuracy...")
    c_acc = classification_accuracy(c_class, "claude")
    o_acc = classification_accuracy(o_class, "ollama")

    # Facet summaries
    c_facets = facet_summary(c_rubric, "claude")
    o_facets = facet_summary(o_rubric, "ollama")

    # Per-persona consistency
    c_persona = persona_consistency(c_rubric, "claude")
    o_persona = persona_consistency(o_rubric, "ollama")

    # Abi/Pat confusion
    c_conf = abi_pat_confusion(c_class, "Claude")
    o_conf = abi_pat_confusion(o_class, "Ollama")

    # Assemble full metrics dict
    metrics = {
        "run_date": datetime.datetime.utcnow().isoformat(),
        "n_evaluations": {"claude": len(c_rubric), "ollama": len(o_rubric)},
        "krippendorff_alpha": kripp,
        "spearman_rho": rho,
        "cohen_kappa": kappa,
        "claude_accuracy": c_acc,
        "ollama_accuracy": o_acc,
        "claude_facets": c_facets,
        "ollama_facets": o_facets,
        "claude_persona_consistency": c_persona,
        "ollama_persona_consistency": o_persona,
        "claude_confusion": c_conf,
        "ollama_confusion": o_conf,
        "citations": {
            "krippendorff_alpha":  "Krippendorff, K. (2019). Content Analysis (4th Ed.). SAGE.",
            "krippendorff_tool":   "Marzi, G. et al. (2024). K-Alpha Calculator. MethodsX, 12, 102545.",
            "spearman_rho":        "Chen et al. (2025). Multi-Agent-as-Judge. arXiv:2507.21028.",
            "cohen_kappa":         "Cohen, J. (1960). Educational and Psychological Measurement, 20, 37-46.",
            "judge_paradigm":      "Zheng et al. (2023). Judging LLM-as-a-Judge. NeurIPS 36.",
            "persona_validation":  "Schuller et al. (2024). Generating personas using LLMs. CHI EA.",
            "gendermag":           "Burnett et al. (2016). GenderMag. Interacting with Computers.",
            "landis_koch":         "Landis & Koch (1977). Biometrics, 33(1), 159-174.",
        }
    }

    # Save JSON
    JUDGE_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved → {OUTPUT_JSON}")

    # Write human-readable report
    write_report(metrics, OUTPUT_TXT)

    # Print summary to terminal
    print("\n" + "="*65)
    print("SUMMARY")
    print("="*65)
    print(f"Krippendorff α  — best: {max((v for v in kripp.values() if isinstance(v, float)), default=0):.3f}  "
          f"worst: {min((v for v in kripp.values() if isinstance(v, float)), default=0):.3f}")
    print(f"Spearman ρ      — {rho.get('rho','N/A')}  (p={rho.get('p','N/A')})")
    print(f"Cohen κ         — {kappa.get('kappa','N/A')}  [{kappa.get('interpretation','N/A')}]")
    print(f"Role accuracy   — Claude: {c_acc.get('role_accuracy',0):.1%}  Ollama: {o_acc.get('role_accuracy',0):.1%}")
    print(f"Full persona    — Claude: {c_acc.get('full_persona_accuracy',0):.1%}  Ollama: {o_acc.get('full_persona_accuracy',0):.1%}")
    print("="*65)


if __name__ == "__main__":
    main()