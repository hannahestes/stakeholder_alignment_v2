#!/usr/bin/env python3
"""
evaluate.py — Persona Preference Study for Pareto Frontier Trees

Two-phase study using GenderMag-inspired personas. Uses a local Ollama model
for LLM inference (free, no API key required).

Trees are evaluated on three metrics only:
  accuracy        hold-out accuracy (%)           higher is better
  stability       avg feature frequency (%)        higher is better
  tree_complexity normalized composite score 0–1   lower is simpler

  Phase 1: Each persona sees the raw tree (attribute table + decision rules).
           Low-tech personas may say "not sure" — tree is too complex to parse.

  Phase 2: Each persona sees an Ollama-generated, persona-tailored description.
           Everyone must choose — "not sure" is not an option.

Prerequisites:
  1. Install Ollama: https://ollama.ai/
  2. Pull the model: ollama pull neural-chat
  3. Start server:   ollama serve  (separate terminal)

Usage:
  python3 evaluate.py --pareto ../pareto_generation/example_output/pareto_2d.json
  python3 evaluate.py --pareto pareto_2d.json --out preferences.csv
  python3 evaluate.py --pareto pareto_2d.json --dry-run

Output:
  preferences.csv   — one row per persona (phase1/phase2 choice, changed?)
  descriptions.json — Ollama-generated descriptions per persona × tree
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import requests


# ── Ollama ────────────────────────────────────────────────────────────────────

OLLAMA_BASE   = 'http://localhost:11434'
DEFAULT_MODEL = 'neural-chat'


def ollama_generate(prompt: str, model: str) -> str:
    try:
        resp = requests.post(
            f'{OLLAMA_BASE}/api/generate',
            json={'model': model, 'prompt': prompt, 'stream': False},
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json().get('response', '').strip()
    except requests.exceptions.ConnectionError:
        print('\nERROR: Cannot connect to Ollama. Start it with: ollama serve')
        sys.exit(1)
    except requests.exceptions.Timeout:
        print('\nERROR: Ollama timed out.')
        sys.exit(1)


def check_ollama(model: str):
    try:
        resp = requests.get(f'{OLLAMA_BASE}/api/tags', timeout=3)
        resp.raise_for_status()
        available = [m['name'].split(':')[0] for m in resp.json().get('models', [])]
        if model not in available:
            print(f'ERROR: Model "{model}" not found.')
            print(f'  Install: ollama pull {model}')
            print(f'  Available: {", ".join(available) or "none"}')
            sys.exit(1)
        print(f'Ollama ready — model: {model}')
    except requests.exceptions.ConnectionError:
        print('ERROR: Ollama is not running. Start it with: ollama serve')
        sys.exit(1)


# ── Persona Definitions ──────────────────────────────────────────────────────
#
# 3 roles × 3 cognitive styles = 9 personas
#
# complexity_threshold (0–1): if any frontier tree's tree_complexity score
#   exceeds this, the persona may say "not sure" in Phase 1 (the rules are
#   too complex for them to interpret unaided).
#
# priorities: which of the three metrics matter most to this persona,
#   used to frame Phase 2 descriptions.

PERSONAS = {
    # ── Project Manager ───────────────────────────────────────────────────────
    'PjM-Abi': {
        'role': 'Project Manager',
        'style': 'Abi',
        'role_desc': (
            'Non-technical project manager (tech fluency 1/5). '
            'Focuses on schedule, team coordination, and explaining decisions to stakeholders. '
            'Values predictability and operational simplicity above all.'
        ),
        'style_desc': (
            'Low self-efficacy, comprehensive info-processor, risk-averse, process-oriented. '
            'Prefers step-by-step guidance. Technical jargon is intimidating.'
        ),
        'priorities': ['tree_complexity', 'stability'],
        'complexity_threshold': 0.3,
    },
    'PjM-Pat': {
        'role': 'Project Manager',
        'style': 'Pat',
        'role_desc': (
            'Non-technical project manager (tech fluency 1/5). '
            'Focuses on schedule, team coordination, and explaining decisions to stakeholders. '
            'Values predictability and operational simplicity above all.'
        ),
        'style_desc': (
            'Medium self-efficacy, comprehensive info-processor, risk-averse, reflective tinkerer. '
            'Learns by trying; appreciates concrete examples and reassurance.'
        ),
        'priorities': ['tree_complexity', 'stability'],
        'complexity_threshold': 0.4,
    },
    'PjM-Tim': {
        'role': 'Project Manager',
        'style': 'Tim',
        'role_desc': (
            'Non-technical project manager (tech fluency 1/5). '
            'Focuses on schedule, team coordination, and explaining decisions to stakeholders. '
            'Values predictability and operational simplicity above all.'
        ),
        'style_desc': (
            'High self-efficacy, selective info-processor, risk-tolerant, tinkerer. '
            'Skips hand-holding; wants key trade-offs stated up front.'
        ),
        'priorities': ['tree_complexity', 'stability'],
        'complexity_threshold': 0.5,
    },

    # ── Product Manager ───────────────────────────────────────────────────────
    'PdM-Abi': {
        'role': 'Product Manager',
        'style': 'Abi',
        'role_desc': (
            'Moderately technical product manager (tech fluency 3/5). '
            'Owns feature direction and customer outcome metrics. '
            'Cares about user impact, business value, and what the model means for customers.'
        ),
        'style_desc': (
            'Low self-efficacy, comprehensive info-processor, risk-averse, process-oriented. '
            'Needs plain language; worried about making the wrong call.'
        ),
        'priorities': ['accuracy', 'tree_complexity'],
        'complexity_threshold': 0.35,
    },
    'PdM-Pat': {
        'role': 'Product Manager',
        'style': 'Pat',
        'role_desc': (
            'Moderately technical product manager (tech fluency 3/5). '
            'Owns feature direction and customer outcome metrics. '
            'Cares about user impact, business value, and what the model means for customers.'
        ),
        'style_desc': (
            'Medium self-efficacy, comprehensive info-processor, risk-averse, reflective tinkerer. '
            'Wants examples of what the tree means in practice.'
        ),
        'priorities': ['accuracy', 'tree_complexity', 'stability'],
        'complexity_threshold': 0.5,
    },
    'PdM-Tim': {
        'role': 'Product Manager',
        'style': 'Tim',
        'role_desc': (
            'Moderately technical product manager (tech fluency 3/5). '
            'Owns feature direction and customer outcome metrics. '
            'Cares about user impact, business value, and what the model means for customers.'
        ),
        'style_desc': (
            'High self-efficacy, selective info-processor, risk-tolerant, tinkerer. '
            'Confident with data; wants outcome-focused summary, not basics.'
        ),
        'priorities': ['accuracy', 'stability', 'tree_complexity'],
        'complexity_threshold': 0.7,
    },

    # ── Software Engineer ─────────────────────────────────────────────────────
    'SWE-Abi': {
        'role': 'Software Engineer',
        'style': 'Abi',
        'role_desc': (
            'Highly technical software engineer (tech fluency 5/5). '
            'Responsible for implementation and deployment. '
            'Cares about accuracy, maintainability, and reliability.'
        ),
        'style_desc': (
            'Low self-efficacy, comprehensive info-processor, risk-averse, process-oriented. '
            'Despite technical background, prefers clear step-by-step explanations.'
        ),
        'priorities': ['accuracy', 'tree_complexity', 'stability'],
        'complexity_threshold': 1.0,   # engineers can always read raw trees
    },
    'SWE-Pat': {
        'role': 'Software Engineer',
        'style': 'Pat',
        'role_desc': (
            'Highly technical software engineer (tech fluency 5/5). '
            'Responsible for implementation and deployment. '
            'Cares about accuracy, maintainability, and reliability.'
        ),
        'style_desc': (
            'Medium self-efficacy, comprehensive info-processor, risk-averse, reflective tinkerer. '
            'Wants to see how the tree performs on edge cases.'
        ),
        'priorities': ['accuracy', 'stability', 'tree_complexity'],
        'complexity_threshold': 1.0,
    },
    'SWE-Tim': {
        'role': 'Software Engineer',
        'style': 'Tim',
        'role_desc': (
            'Highly technical software engineer (tech fluency 5/5). '
            'Responsible for implementation and deployment. '
            'Cares about accuracy, maintainability, and reliability.'
        ),
        'style_desc': (
            'High self-efficacy, selective info-processor, risk-tolerant, tinkerer. '
            'Wants raw metrics; skip introductions.'
        ),
        'priorities': ['accuracy', 'stability', 'tree_complexity'],
        'complexity_threshold': 1.0,
    },
}

ALL_PERSONAS = list(PERSONAS.keys())


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_pareto(path: str) -> list:
    with open(path) as f:
        data = json.load(f)
    trees = data.get('trees', [])
    for t in trees:
        t.setdefault('run_num', t.get('run', '?'))
        t.setdefault('tree_complexity', 0.0)   # absent in pre-v2 JSON files
    return trees


# ── Text Helpers ──────────────────────────────────────────────────────────────

def frontier_table(trees: list) -> str:
    header = f"{'Run':<6}{'Acc%':<8}{'Stability%':<13}{'Complexity(0-1)':<18}  Features"
    sep = '-' * 75
    rows = [header, sep]
    for t in trees:
        feats = ', '.join(t.get('features', []))
        rows.append(
            f"{t['run_num']:<6}{t['accuracy']:<8}{t['stability']:<13.1f}"
            f"{t['tree_complexity']:<18.4f}  {feats}"
        )
    return '\n'.join(rows)


def raw_rules(tree: dict) -> str:
    return tree.get('raw_output', '[no raw output]').strip()


def can_parse(persona_id: str, trees: list) -> bool:
    """True if no frontier tree exceeds the persona's complexity threshold."""
    threshold = PERSONAS[persona_id]['complexity_threshold']
    return max(t['tree_complexity'] for t in trees) <= threshold


# ── Prompt Builders ───────────────────────────────────────────────────────────

def build_phase1_prompt(persona_id: str, trees: list) -> str:
    p = PERSONAS[persona_id]
    allow_not_sure = not can_parse(persona_id, trees)
    runs_list = ', '.join(f"Run {t['run_num']}" for t in trees)
    not_sure_clause = (
        '\n  - "not sure" — if the decision rules are too complex for you to interpret'
        if allow_not_sure else ''
    )
    raw_blocks = '\n\n'.join(
        f"── Run {t['run_num']} ──────────────────────────\n{raw_rules(t)}"
        for t in trees
    )

    return f"""You are simulating a stakeholder persona for a research study.

PERSONA: {persona_id}
Role: {p['role']}
Role context: {p['role_desc']}
Cognitive style: {p['style_desc']}
Top priorities (in order): {', '.join(p['priorities'])}

─────────────────────────────────────────────────
STUDY CONTEXT
─────────────────────────────────────────────────
A machine-learning team produced {len(trees)} decision trees. These are the
PARETO FRONTIER — no tree dominates another; each represents a different trade-off.

ATTRIBUTE TABLE:
{frontier_table(trees)}

  Acc%            = hold-out accuracy (higher is better)
  Stability%      = how consistently this tree's features appear across all runs
                    (higher = more stable, less sensitive to data variations)
  Complexity(0-1) = normalized composite of tree depth + number of features
                    (lower = simpler; 0 = simplest possible, 1 = most complex)

─────────────────────────────────────────────────
RAW DECISION RULES
─────────────────────────────────────────────────
{raw_blocks}

─────────────────────────────────────────────────
TASK
─────────────────────────────────────────────────
As {persona_id}, which tree do you prefer?

Valid answers: {runs_list}{not_sure_clause}

Reply in EXACTLY this format (nothing else):

CHOICE: Run <number>   [or: CHOICE: not sure]
REASON: <one or two sentences from your persona's perspective>
""".strip()


def build_description_prompt(persona_id: str, tree: dict) -> str:
    p = PERSONAS[persona_id]
    return f"""Write a plain-language description of this decision-tree model, tailored for:

Role: {p['role']}
Cognitive style: {p['style_desc']}
What they care about most: {', '.join(p['priorities'])}

Tree metrics:
  Run number      : {tree['run_num']}
  Accuracy        : {tree['accuracy']}%
  Stability       : {tree['stability']:.1f}%
  Tree Complexity : {tree['tree_complexity']:.4f} (0=simplest, 1=most complex)
  Features used   : {', '.join(tree['features'])}

Raw decision rules:
{raw_rules(tree)}

Write 3-5 sentences. Plain prose only — no headers, no bullets. Explain what
the tree does, what its complexity score means in practice, and why this tree
might or might not suit this persona's priorities.
""".strip()


def build_phase2_prompt(persona_id: str, trees: list, descriptions: dict) -> str:
    p = PERSONAS[persona_id]
    runs_list = ', '.join(f"Run {t['run_num']}" for t in trees)
    desc_blocks = '\n\n'.join(
        f"── Run {t['run_num']} ─────\n{descriptions.get(t['run_num'], '[not available]')}"
        for t in trees
    )

    return f"""You are simulating a stakeholder persona for a research study.

PERSONA: {persona_id}
Role: {p['role']}
Role context: {p['role_desc']}
Cognitive style: {p['style_desc']}
Top priorities (in order): {', '.join(p['priorities'])}

─────────────────────────────────────────────────
STUDY CONTEXT
─────────────────────────────────────────────────
You now have plain-language descriptions of all trees written specifically
for your role and cognitive style.

ATTRIBUTE TABLE (for reference):
{frontier_table(trees)}

─────────────────────────────────────────────────
PLAIN-LANGUAGE DESCRIPTIONS (tailored for {persona_id})
─────────────────────────────────────────────────
{desc_blocks}

─────────────────────────────────────────────────
TASK
─────────────────────────────────────────────────
Which tree do you prefer now that you have clearer descriptions?
You MUST choose — "not sure" is not allowed this time.

Valid answers: {runs_list}

Reply in EXACTLY this format (nothing else):

CHOICE: Run <number>
REASON: <one or two sentences from your persona's perspective>
CHANGED: yes/no   [did you change from your Phase 1 choice?]
""".strip()


# ── Response Parsing ──────────────────────────────────────────────────────────

def parse_choice(text: str):
    for line in text.splitlines():
        if line.strip().upper().startswith('CHOICE:'):
            return line.split(':', 1)[1].strip()
    return None


def parse_reason(text: str):
    for line in text.splitlines():
        if line.strip().upper().startswith('REASON:'):
            return line.split(':', 1)[1].strip()
    return None


def parse_changed(text: str):
    for line in text.splitlines():
        if line.strip().upper().startswith('CHANGED:'):
            return line.split(':', 1)[1].strip().lower()
    return 'unknown'


# ── Study Runner ──────────────────────────────────────────────────────────────

def run_study(trees: list, model: str, dry_run: bool = False):
    rows = []
    all_descriptions = {}

    for persona_id in ALL_PERSONAS:
        p = PERSONAS[persona_id]
        print(f'\n{"─"*60}')
        print(f'PERSONA: {persona_id}  ({p["role"]} / {p["style"]})')

        row = {
            'persona':        persona_id,
            'role':           p['role'],
            'style':          p['style'],
            'can_parse_raw':  can_parse(persona_id, trees),
            'phase1_choice':  None,
            'phase1_reason':  None,
            'phase2_choice':  None,
            'phase2_reason':  None,
            'changed':        None,
        }

        # ── Phase 1 ──────────────────────────────────────────────────────────
        print('  Phase 1 (raw tree)...', end='', flush=True)
        p1_prompt = build_phase1_prompt(persona_id, trees)

        if dry_run:
            print('\n[DRY RUN — Phase 1 prompt (first 600 chars)]\n')
            print(p1_prompt[:600] + '\n...\n')
            row['phase1_choice'] = 'DRY_RUN'
        else:
            p1_resp = ollama_generate(p1_prompt, model)
            row['phase1_choice'] = parse_choice(p1_resp)
            row['phase1_reason'] = parse_reason(p1_resp)
            print(f' → {row["phase1_choice"]}')

        # ── Phase 2: generate descriptions ───────────────────────────────────
        descriptions = {}
        for tree in trees:
            run_num = tree['run_num']
            print(f'  Generating description for Run {run_num}...', end='', flush=True)
            if dry_run:
                descriptions[run_num] = f'[DRY RUN description for Run {run_num}]'
                print(' DRY RUN')
            else:
                descriptions[run_num] = ollama_generate(
                    build_description_prompt(persona_id, tree), model
                )
                print(' done')
                time.sleep(0.3)

        all_descriptions[persona_id] = descriptions

        # ── Phase 2: preference with descriptions ─────────────────────────────
        print('  Phase 2 (with descriptions)...', end='', flush=True)
        p2_prompt = build_phase2_prompt(persona_id, trees, descriptions)

        if dry_run:
            print('\n[DRY RUN — Phase 2 prompt (first 400 chars)]\n')
            print(p2_prompt[:400] + '\n...\n')
            row['phase2_choice'] = 'DRY_RUN'
            row['changed'] = 'unknown'
        else:
            p2_resp = ollama_generate(p2_prompt, model)
            row['phase2_choice'] = parse_choice(p2_resp)
            row['phase2_reason'] = parse_reason(p2_resp)
            row['changed'] = parse_changed(p2_resp)
            print(f' → {row["phase2_choice"]}  (changed: {row["changed"]})')

        rows.append(row)

    return rows, all_descriptions


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(rows: list, trees: list):
    print(f'\n{"═"*60}')
    print('STUDY SUMMARY')
    print(f'{"═"*60}')

    run_nums = [t['run_num'] for t in trees]
    p1_counts = {r: 0 for r in run_nums}
    p2_counts = {r: 0 for r in run_nums}
    p1_not_sure = 0
    changed = 0

    for row in rows:
        c1 = str(row.get('phase1_choice') or '')
        if 'not sure' in c1.lower():
            p1_not_sure += 1
        else:
            for r in run_nums:
                if str(r) in c1:
                    p1_counts[r] += 1

        c2 = str(row.get('phase2_choice') or '')
        for r in run_nums:
            if str(r) in c2:
                p2_counts[r] += 1

        if row.get('changed') == 'yes':
            changed += 1

    print(f'\nPhase 1 — raw tree ("not sure" allowed for low-tech personas):')
    print(f'  Not sure : {p1_not_sure}')
    for r in run_nums:
        print(f'  Run {r}   : {p1_counts[r]} vote(s)')

    print(f'\nPhase 2 — with descriptions (must choose):')
    for r in run_nums:
        print(f'  Run {r}   : {p2_counts[r]} vote(s)')

    print(f'\nPersonas who changed answer in Phase 2: {changed}/{len(rows)}')

    if p2_counts:
        most_common = max(p2_counts, key=lambda r: p2_counts[r])
        top_pct = 100 * p2_counts[most_common] / max(len(rows), 1)
        print(f'\nMost preferred in Phase 2: Run {most_common} ({top_pct:.0f}%)')
        print('→ CONVERGENCE' if top_pct >= 55 else '→ DIVERGENCE')

    print(f'{"═"*60}')


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Persona preference study for Pareto frontier trees (Ollama).'
    )
    ap.add_argument('--pareto', required=True,
                    help='Path to pareto_2d.json or pareto_3d.json')
    ap.add_argument('--model', default=DEFAULT_MODEL,
                    help=f'Ollama model (default: {DEFAULT_MODEL})')
    ap.add_argument('--out', default='preferences.csv',
                    help='Output CSV (default: preferences.csv)')
    ap.add_argument('--desc-out', default='descriptions.json',
                    help='Output JSON for descriptions (default: descriptions.json)')
    ap.add_argument('--dry-run', action='store_true',
                    help='Print prompts without calling Ollama')
    args = ap.parse_args()

    trees = load_pareto(args.pareto)
    if not trees:
        print(f'ERROR: No trees found in {args.pareto}')
        sys.exit(1)

    print(f'Loaded {len(trees)} Pareto frontier trees')
    print(f'Runs: {[t["run_num"] for t in trees]}\n')
    print(frontier_table(trees))

    if not args.dry_run:
        check_ollama(args.model)

    rows, all_descriptions = run_study(trees, args.model, dry_run=args.dry_run)

    if not args.dry_run:
        fieldnames = ['persona', 'role', 'style', 'can_parse_raw',
                      'phase1_choice', 'phase1_reason',
                      'phase2_choice', 'phase2_reason', 'changed']
        with open(args.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f'\nSaved preferences → {args.out}')

        with open(args.desc_out, 'w') as f:
            json.dump(all_descriptions, f, indent=2)
        print(f'Saved descriptions → {args.desc_out}')

    print_summary(rows, trees)


if __name__ == '__main__':
    main()
