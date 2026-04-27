"""
calibration_stats.py — Statistical significance testing for EZR run-count calibration

Tests whether N=400 is significantly better than all other configurations
on the composite score, using:

  - Kruskal-Wallis H test (overall difference across all 10 configurations)
  - Pairwise Wilcoxon rank-sum tests (N=400 vs each alternative)
  - Bonferroni correction (9 comparisons, adjusted alpha = 0.05/9 ≈ 0.006)
  - Rank-biserial correlation r as effect size (non-parametric, bounded [-1, 1])
  - Cohen's d as supplementary effect size (parametric, for reference)

Rank-biserial r interpretation [Cohen 1988]:
  |r| >= 0.10 = small, |r| >= 0.30 = medium, |r| >= 0.50 = large

INPUT: Per-replication composite scores for each run count.
  Edit the `SCORES` dict below with your actual replication data.
  Each list must have exactly 5 values (one per seed replication).

OUTPUT:
  - Console report with all statistics
  - calibration_stats.json (machine-readable)
  - calibration_stats.tex  (LaTeX table ready to drop into paper)

Dependencies: scipy, numpy
  pip install scipy numpy --break-system-packages
"""

import json
import math
import numpy as np
from scipy.stats import kruskal, mannwhitneyu, ttest_ind

# ── INPUT: Replace with your actual per-replication scores ───────────────────
# Each key is N_runs; each value is a list of 5 composite scores
# (one per deterministic seed replication).
# The composite score formula is:
#   Score(N) = (1/5) * (A_hat + S_hat + (1-C_hat) + (1-V_hat) + F_hat)
# where each term is min-max normalized across all 50 runs.
#
# If you only have the aggregated means from Table 2, set HAVE_RAW = False
# and the script will run a sensitivity analysis instead.

HAVE_RAW = False   # ← set True and fill SCORES if you have per-replication data

SCORES = {
    25:   [0.241, 0.258, 0.271, 0.263, 0.282],   # replace with real values
    50:   [0.312, 0.328, 0.341, 0.355, 0.354],
    75:   [0.471, 0.489, 0.503, 0.512, 0.510],
    100:  [0.701, 0.718, 0.731, 0.745, 0.750],
    250:  [0.531, 0.548, 0.561, 0.554, 0.571],
    400:  [0.911, 0.928, 0.935, 0.941, 0.945],   # ← this is your champion
    500:  [0.671, 0.684, 0.692, 0.701, 0.702],
    650:  [0.748, 0.759, 0.771, 0.775, 0.782],
    750:  [0.771, 0.778, 0.781, 0.787, 0.793],
    1000: [0.751, 0.758, 0.768, 0.771, 0.772],
}

# Aggregated means from Table 2 (used when HAVE_RAW = False)
MEANS = {
    25:   0.263,
    50:   0.338,
    75:   0.497,
    100:  0.729,
    250:  0.553,
    400:  0.932,
    500:  0.690,
    650:  0.767,
    750:  0.782,
    1000: 0.764,
}

N_REPS    = 5        # replications per configuration
ALPHA     = 0.05
N_CONFIGS = 9        # comparisons (N=400 vs each other)
ALPHA_ADJ = ALPHA / N_CONFIGS   # Bonferroni-corrected threshold


# ── Statistical functions ─────────────────────────────────────────────────────

def rank_biserial_r(x, y):
    """
    Rank-biserial correlation as effect size for Wilcoxon rank-sum test.

    r = (U / (n1 * n2)) * 2 - 1
    Bounded [-1, 1]. Positive r means x tends to be larger than y.

    Reference: Kerby (2014). The Simple Difference Formula: An Approach to
    Teaching Nonparametric Correlation. Comprehensive Psychology.
    """
    n1, n2 = len(x), len(y)
    u_stat, _ = mannwhitneyu(x, y, alternative='two-sided')
    r = (2 * u_stat) / (n1 * n2) - 1
    return round(r, 4)


def cohens_d(x, y):
    """
    Cohen's d (parametric effect size, for reference alongside r).
    Uses pooled standard deviation.
    """
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return None
    sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)
    pooled_sd = math.sqrt(((nx - 1) * sx**2 + (ny - 1) * sy**2) / (nx + ny - 2))
    if pooled_sd == 0:
        return None
    return round((np.mean(x) - np.mean(y)) / pooled_sd, 4)


def effect_size_label(r):
    ar = abs(r)
    if ar >= 0.50: return "large"
    if ar >= 0.30: return "medium"
    if ar >= 0.10: return "small"
    return "negligible"


def sensitivity_analysis(means):
    """
    When only aggregated means are available, simulate plausible per-replication
    distributions using small synthetic variance and test robustness.
    Reports results under three variance assumptions.
    """
    print("="*65)
    print("SENSITIVITY ANALYSIS (no raw replication data available)")
    print("Simulating per-replication distributions under three")
    print("variance assumptions (SD = 0.5%, 1%, 2% of Score range).")
    print("="*65)

    np.random.seed(42)
    results = {}

    for sd_frac, label in [(0.005, "SD=0.5%"), (0.010, "SD=1%"), (0.020, "SD=2%")]:
        simulated = {n: np.random.normal(loc=m, scale=sd_frac, size=N_REPS).tolist()
                     for n, m in means.items()}
        s400 = simulated[400]
        comparisons = {}

        for n, scores in simulated.items():
            if n == 400:
                continue
            _, p = mannwhitneyu(s400, scores, alternative='greater')
            r = rank_biserial_r(s400, scores)
            d = cohens_d(s400, scores)
            sig = p < ALPHA_ADJ
            comparisons[n] = {
                "p_raw": round(p, 6),
                "p_bonf": round(min(p * N_CONFIGS, 1.0), 6),
                "significant": sig,
                "r": r,
                "d": d,
                "effect": effect_size_label(r),
            }

        results[label] = comparisons
        n_sig = sum(1 for v in comparisons.values() if v["significant"])
        print(f"\n  [{label}] N=400 significantly beats {n_sig}/9 alternatives "
              f"(Bonferroni α={ALPHA_ADJ:.4f})")
        for n in sorted(comparisons):
            c = comparisons[n]
            sig_str = "✓" if c["significant"] else "✗"
            print(f"    vs N={n:<5} p_bonf={c['p_bonf']:.4f}  r={c['r']:+.3f} "
                  f"[{c['effect']}]  d={c['d']:+.3f}  {sig_str}")

    print()
    print("NOTE: Replace placeholder SCORES with actual replication data")
    print("      for definitive results. Set HAVE_RAW = True above.")
    return results


# ── Main analysis ─────────────────────────────────────────────────────────────

def run_analysis():
    if not HAVE_RAW:
        sens = sensitivity_analysis(MEANS)
        report = {"mode": "sensitivity_analysis", "results": sens,
                  "note": "Set HAVE_RAW=True and fill SCORES with actual data."}
        with open("calibration_stats.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        print("\nSaved → calibration_stats.json")
        print("\nTo get definitive results, add your 5 per-replication")
        print("scores for each N to the SCORES dict at the top of this script.")
        return

    # ── Full analysis with raw replication data ───────────────────────────────
    all_scores = [SCORES[n] for n in sorted(SCORES)]
    s400 = SCORES[400]

    # 1. Kruskal-Wallis overall test
    H, p_kw = kruskal(*all_scores)
    print("="*65)
    print("CALIBRATION SIGNIFICANCE REPORT")
    print("="*65)
    print(f"\n1. KRUSKAL-WALLIS (overall difference across all 10 configs)")
    print(f"   H = {H:.4f}   p = {p_kw:.6f}   "
          f"{'SIGNIFICANT' if p_kw < ALPHA else 'NOT significant'} (α={ALPHA})")

    # 2. Pairwise: N=400 vs each alternative
    print(f"\n2. PAIRWISE WILCOXON (N=400 vs each alternative)")
    print(f"   Bonferroni-corrected α = {ALPHA}/{N_CONFIGS} = {ALPHA_ADJ:.5f}")
    print(f"   Effect size: rank-biserial r  (|r|≥0.50=large, ≥0.30=medium, ≥0.10=small)")
    print(f"   Supplementary: Cohen's d")
    print()
    print(f"  {'N_runs':<8} {'p (raw)':>10} {'p (Bonf)':>10} {'Sig':>5} "
          f"{'r':>7} {'Effect':>10} {'d':>8}")
    print("  " + "-"*62)

    comparisons = {}
    for n in sorted(SCORES):
        if n == 400:
            continue
        scores_n = SCORES[n]
        _, p_raw = mannwhitneyu(s400, scores_n, alternative='greater')
        p_bonf   = min(p_raw * N_CONFIGS, 1.0)
        r        = rank_biserial_r(s400, scores_n)
        d        = cohens_d(s400, scores_n)
        sig      = p_bonf < ALPHA
        eff      = effect_size_label(r)

        comparisons[n] = {
            "mean_400":   round(np.mean(s400), 4),
            "mean_n":     round(np.mean(scores_n), 4),
            "p_raw":      round(p_raw, 6),
            "p_bonferroni": round(p_bonf, 6),
            "significant":  sig,
            "r":            r,
            "d":            d if d else "N/A",
            "effect_size":  eff,
        }

        sig_str = "✓" if sig else "✗"
        d_str   = f"{d:+.3f}" if d else "N/A"
        print(f"  {n:<8} {p_raw:>10.5f} {p_bonf:>10.5f} {sig_str:>5} "
              f"{r:>+7.3f} {eff:>10} {d_str:>8}")

    n_sig = sum(1 for v in comparisons.values() if v["significant"])
    print()
    print(f"  N=400 significantly outperforms {n_sig}/9 alternatives "
          f"(Bonferroni α={ALPHA_ADJ:.5f})")

    # 3. Write LaTeX table
    latex = _build_latex_table(comparisons, H, p_kw)
    with open("calibration_stats.tex", "w") as f:
        f.write(latex)
    print("\nSaved → calibration_stats.tex")

    # 4. Save JSON
    report = {
        "kruskal_wallis": {"H": round(H, 4), "p": round(p_kw, 6)},
        "alpha": ALPHA,
        "alpha_bonferroni": round(ALPHA_ADJ, 6),
        "n_reps_per_config": N_REPS,
        "comparisons_vs_400": comparisons,
        "n_significant": n_sig,
        "citations": {
            "kruskal_wallis":        "Kruskal & Wallis (1952). JASA, 47(260), 583-621.",
            "wilcoxon_ranksum":      "Mann & Whitney (1947). Ann. Math. Stat., 18(1), 50-60.",
            "bonferroni":            "Dunn (1961). JASA, 56(293), 52-64.",
            "rank_biserial_r":       "Kerby (2014). Comprehensive Psychology, 3, 11-IT.",
            "cohens_d":              "Cohen (1988). Statistical Power Analysis (2nd ed.). LEA.",
            "effect_size_thresholds":"Cohen (1988). |r|>=0.50 large, >=0.30 medium, >=0.10 small.",
        }
    }
    with open("calibration_stats.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Saved → calibration_stats.json")


def _build_latex_table(comparisons, H, p_kw):
    """Generate a LaTeX table of pairwise results for the paper."""
    rows = []
    for n in sorted(comparisons):
        c = comparisons[n]
        sig = "$^{*}$" if c["significant"] else ""
        d   = f"{c['d']:+.3f}" if isinstance(c["d"], float) else "---"
        rows.append(
            f"  {n:<6} & {c['mean_n']:.3f} & {c['p_raw']:.5f} & "
            f"{c['p_bonferroni']:.5f}{sig} & {c['r']:+.3f} & "
            f"{c['effect_size']:<10} & {d} \\\\"
        )

    rows_str = "\n".join(rows)
    return rf"""% ── Statistical Significance: N=400 vs Alternatives ──────────────
% Kruskal-Wallis H={H:.3f}, p={p_kw:.5f} (overall)
% Bonferroni-corrected \alpha = {ALPHA}/{N_CONFIGS} \approx {ALPHA_ADJ:.4f}
% $^{{*}}$ = significant after Bonferroni correction
%
\begin{{table}}[ht]
  \caption{{Pairwise comparison of $N=400$ against all other run counts
           on the composite calibration score (5 replications per
           configuration). $p$ values from one-sided Wilcoxon rank-sum
           test ($N=400 > N_{{alt}}$); Bonferroni-corrected for 9
           comparisons ($\alpha_{{adj}} \approx {ALPHA_ADJ:.4f}$).
           $r$: rank-biserial correlation (effect size;
           $|r| \geq 0.50$ = large, $\geq 0.30$ = medium,
           $\geq 0.10$ = small~\cite{{kerby2014simple}}).
           $d$: Cohen's $d$ (supplementary).
           Overall Kruskal-Wallis: $H={H:.3f}$, $p={p_kw:.5f}$.}}
  \label{{tab:calibration-stats}}
  \small
  \begin{{tabular}}{{rrrrrrl}}
    \toprule
    $N_{{\text{{runs}}}}$ & \textbf{{Score mean}} & $p$ (raw) &
    $p$ (Bonf.) & $r$ & Effect & $d$ \\
    \midrule
    \addlinespace
{rows_str}
    \bottomrule
    \multicolumn{{7}}{{l}}{{$^{{*}}$significant after Bonferroni correction ($\alpha_{{adj}} \approx {ALPHA_ADJ:.4f}$).}}
  \end{{tabular}}
\end{{table}}
"""


if __name__ == "__main__":
    run_analysis()