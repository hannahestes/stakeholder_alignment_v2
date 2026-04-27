"""
Ollama Persona Simulator: Run persona evaluations using local Ollama models (free!)

This replaces Claude API calls with local model inference.
Models can run on CPU (slower but free) or GPU (faster if available).

Install Ollama: https://ollama.ai/
Then: ollama pull llama2  (or another model)
Then: ollama serve  (in another terminal)
"""

import json
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional


# ============================================================================
# OLLAMA CONFIGURATION
# ============================================================================

OLLAMA_CONFIG = {
    'base_url': 'http://localhost:11434',
    'model': 'neural-chat',  # Change to 'mistral', 'neural-chat', etc. if preferred
    'temperature': 0.7,
    'top_p': 0.9,
    'num_predict': 500,  # Max tokens for response
}

RECOMMENDED_MODELS = {
    # 'llama2': {
    #     'size': '3.8GB',
    #     'speed': 'moderate',
    #     'quality': 'good',
    #     'install': 'ollama pull llama2'
    # },
    # 'mistral': {
    #     'size': '4.1GB',
    #     'speed': 'fast',
    #     'quality': 'very good',
    #     'install': 'ollama pull mistral'
    # },
    'neural-chat': {
        'size': '4.1GB',
        'speed': 'moderate',
        'quality': 'excellent (best for this task)',
        'install': 'ollama pull neural-chat'
    },
    # 'orca-mini': {
    #     'size': '1.3GB',
    #     'speed': 'very fast',
    #     'quality': 'decent',
    #     'install': 'ollama pull orca-mini'
    # },
}


# ============================================================================
# PERSONA CONTEXT: GenderMag Profiles
# ============================================================================

PERSONA_CONTEXTS = {
    'Abi': """
You are Abi, a representative persona from the GenderMag framework:
- Motivation: Task-oriented (motivated to complete the job, not explore tech)
- Information Processing: Comprehensive (wants to see all available information)
- Self-Efficacy: LOW (not confident in own technical abilities)
- Risk Attitude: Risk-averse (prefers safe, proven approaches)
- Learning Style: Process-oriented (learns by following step-by-step procedures)

When evaluating explanations:
- You may feel intimidated by technical jargon
- You appreciate clear, reassuring language
- You want step-by-step walkthroughs
- You prefer conservative recommendations
""",
    
    'Pat': """
You are Pat, a representative persona from the GenderMag framework:
- Motivation: Task-oriented
- Information Processing: Comprehensive (like Abi)
- Self-Efficacy: MEDIUM (somewhat confident)
- Risk Attitude: Risk-averse
- Learning Style: Reflective tinkerer (learns by trying, reflecting, then trying again)

When evaluating explanations:
- You appreciate both clarity AND working examples
- You'll persevere through technical explanations if they're well-motivated
- You like seeing "what this means in practice"
- You want encouragement and acknowledgment of complexity
""",
    
    'Tim': """
You are Tim, a representative persona from the GenderMag framework:
- Motivation: Technology-oriented (curious about the tech itself)
- Information Processing: Selective (focuses on key metrics and insights)
- Self-Efficacy: HIGH (confident in own technical abilities)
- Risk Attitude: Risk-tolerant (willing to experiment)
- Learning Style: Tinkerer (learns by diving in and exploring)

When evaluating explanations:
- You want efficiency—skip the hand-holding
- You're interested in edge cases and interactions
- You appreciate advanced metrics and trade-offs
- Overexplained basics annoy you
""",
}

ROLE_CONTEXTS = {
    'PjM': """
You are a PROJECT MANAGER (PjM):
- Responsibility: Planning, scheduling, coordination
- Technical Level: 1/5 (non-technical)
- Key Values: Predictability, simplicity, minimal disruption
- Key Concerns: Can our teams operate this? How do we explain it to stakeholders?
""",
    
    'PdM': """
You are a PRODUCT MANAGER (PdM):
- Responsibility: Product direction and feature decisions
- Technical Level: 3/5 (moderately technical)
- Key Values: Customer impact, user outcomes, business value
- Key Concerns: How does this affect our customers? What's the user experience impact?
""",
    
    'SWE': """
You are a SOFTWARE ENGINEER (SWE):
- Responsibility: Implementation and deployment
- Technical Level: 5/5 (highly technical)
- Key Values: Accuracy, maintainability, reliability, code quality
- Key Concerns: Can I build and maintain this? What are the exact thresholds and data types?
""",
}


# ============================================================================
# OLLAMA PERSONA SIMULATOR
# ============================================================================

class OllamaPersonaSimulator:
    """
    Simulate persona evaluations using Ollama (local, free LLM).
    """
    
    def __init__(self, model=OLLAMA_CONFIG['model'], base_url=OLLAMA_CONFIG['base_url']):
        self.model = model
        self.base_url = base_url
        self.api_endpoint = f"{base_url}/api/generate"
        
        # Check Ollama is running
        self._check_ollama_running()
    
    def _check_ollama_running(self):
        """Verify Ollama server is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                print(f"✓ Ollama is running at {self.base_url}")
                models = response.json().get('models', [])
                model_names = [m['name'].split(':')[0] for m in models]
                print(f"  Available models: {', '.join(model_names)}")
                
                if self.model not in model_names:
                    print(f"\n⚠ Model '{self.model}' not found.")
                    print(f"  Install it with: ollama pull {self.model}")
                    raise RuntimeError(f"Model {self.model} not available")
            return True
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Ollama is not running at {self.base_url}\n"
                f"Start it with: ollama serve\n"
                f"(in a separate terminal)"
            )
    
    def simulate_persona(self, persona_id: str, tree_metrics: Dict, prompt: str) -> Dict:
        """
        Simulate a persona evaluating a tree explanation.
        
        Args:
            persona_id: e.g., 'SWE-Tim' (role-style)
            tree_metrics: {'complexity': 3, 'win': 77, 'num_features': 3}
            prompt: The tree explanation to evaluate
        
        Returns:
            {
                'persona': 'SWE-Tim',
                'clarity_rating': 4,
                'acceptance': 'yes',
                'clarity_reasoning': '...',
                'concerns': '...',
                'learned': '...',
                'wants_to_know': '...',
                'full_response': '...',
                'model_used': 'llama2',
                'tokens_used': 1234,
            }
        """
        
        role, style = persona_id.split('-')
        
        # Build the full prompt with persona context
        full_prompt = self._build_evaluation_prompt(
            persona_id, role, style, tree_metrics, prompt
        )
        
        # Call Ollama
        print(f"  → Evaluating with {persona_id}...", end='', flush=True)
        response_text = self._call_ollama(full_prompt)
        print(" ✓")
        
        # Parse response
        parsed = self._parse_response(response_text, persona_id)
        parsed['model_used'] = self.model
        
        return parsed
    
    def _build_evaluation_prompt(self, persona_id: str, role: str, style: str, 
                                  tree_metrics: Dict, explanation: str) -> str:
        """Build the full evaluation prompt with persona context."""
        
        prompt = f"""
{PERSONA_CONTEXTS[style]}

{ROLE_CONTEXTS[role]}

---

TREE DETAILS:
- Complexity: {tree_metrics['complexity']} features
- Accuracy: {tree_metrics['win']}%
- Num Features: {tree_metrics['num_features']}

---

TREE EXPLANATION:
{explanation}

---

YOUR EVALUATION:

Please respond in the following format:

CLARITY RATING: [1-5]
RATING REASONING: [2-3 sentences explaining your rating]

ACCEPTANCE: [yes/no/partially]
CONCERNS: [If any, describe what concerns you about this tree]

LEARNED: [What is one key thing you learned from this explanation?]
WANT_TO_KNOW: [What would you like to know more about?]

---

OVERALL PREFERENCE RANKING (with detailed reasoning):

Think about ALL FOUR trees in the Pareto frontier:
- Run 24: Complexity 2, Accuracy 68% (simplest)
- Run 22: Complexity 3, Accuracy 77% (balanced)
- Run 44: Complexity 4, Accuracy 98% (accurate)
- Run 26: Complexity 6, Accuracy 100% (perfect but complex)

From your perspective as {persona_id}, rank them. For each choice, explain your reasoning in 1-2 sentences. Consider what matters most to you: accuracy, simplicity, operational ease, customer impact, maintainability, etc.

BEST CHOICE: Run [22/24/26/44]
WHY: [Your detailed reasoning - what trade-offs are you accepting? What matters most?]

SECOND: Run [22/24/26/44]
WHY: [Your detailed reasoning]

THIRD: Run [22/24/26/44]
WHY: [Your detailed reasoning]

WORST: Run [22/24/26/44]
WHY: [Your detailed reasoning]

---

Stay in character as {persona_id}. Give honest, persona-appropriate feedback.
Be specific about trade-offs: "I chose X because Y matters more than Z to me."
"""
        
        return prompt.strip()
    
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API and stream response."""
        
        try:
            response = requests.post(
                self.api_endpoint,
                json={
                    'model': self.model,
                    'prompt': prompt,
                    'temperature': OLLAMA_CONFIG['temperature'],
                    'top_p': OLLAMA_CONFIG['top_p'],
                    'num_predict': OLLAMA_CONFIG['num_predict'],
                    'stream': False,  # False = wait for full response
                },
                timeout=300  # 5 min timeout (local inference can be slow)
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"Ollama error: {response.text}")
            
            result = response.json()
            return result.get('response', '')
        
        except requests.exceptions.Timeout:
            raise RuntimeError(
                "Ollama request timed out. Model inference is taking too long.\n"
                "Try a smaller model (orca-mini) or enable GPU acceleration."
            )
    
    def _parse_response(self, response_text: str, persona_id: str) -> Dict:
        """Extract structured data from persona response."""
        
        result = {
            'persona': persona_id,
            'clarity_rating': None,
            'rating_reasoning': '',
            'acceptance': '',
            'concerns': '',
            'learned': '',
            'want_to_know': '',
            'ranking': {
                'best': None,
                'second': None,
                'third': None,
                'worst': None,
            },
            'full_response': response_text.strip(),
        }
        
        # Parse clarity rating (1-5)
        import re
        clarity_match = re.search(r'CLARITY\s*RATING:\s*(\d)', response_text, re.IGNORECASE)
        if clarity_match:
            result['clarity_rating'] = int(clarity_match.group(1))
        
        # Parse sections
        sections = {
            'RATING REASONING': 'rating_reasoning',
            'ACCEPTANCE': 'acceptance',
            'CONCERNS': 'concerns',
            'LEARNED': 'learned',
            'WANT_TO_KNOW': 'want_to_know',
        }
        
        for section_name, result_key in sections.items():
            pattern = f"{section_name}:\\s*(.+?)(?=\\n[A-Z_]+:|$)"
            match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
            if match:
                result[result_key] = match.group(1).strip()
        
        # Parse BEST choice (REQUIRED)
        # Flexible regex to handle variations like:
        # - "BEST CHOICE: Run 22\nWHY: ..."
        # - "BEST CHOICE: Run 22 (Balanced)\nWHY: ..."
        # - "BEST CHOICE: Run 22 (Reasoning: ...)"
        
        # First, try to extract the run number from BEST CHOICE line
        best_choice_match = re.search(r'BEST\s*CHOICE:\s*Run\s*(\d+)', response_text, re.IGNORECASE)
        
        if best_choice_match:
            run_num = best_choice_match.group(1)
            
            # Now try to find the reasoning/why
            # Look for "WHY:" or "Reasoning:" after the BEST CHOICE
            why_match = re.search(r'(?:WHY|Reasoning):\s*(.+?)(?=\nSECOND:|$)', response_text, re.DOTALL | re.IGNORECASE)
            
            if why_match:
                reasoning = why_match.group(1).strip()
            else:
                # Try to get reasoning from within parentheses: "Run 22 (Reasoning: ...)"
                paren_match = re.search(r'BEST\s*CHOICE:\s*Run\s*\d+\s*\(([^)]+)\)', response_text, re.IGNORECASE)
                if paren_match:
                    reasoning = paren_match.group(1).strip()
                    # Remove "Reasoning:" or "Balanced" prefix if present
                    reasoning = re.sub(r'^(?:Reasoning|Balanced):\s*', '', reasoning, flags=re.IGNORECASE)
                else:
                    reasoning = '[No reasoning found]'
            
            # Validate that reasoning is meaningful
            if len(reasoning) > 10 and '[No reasoning' not in reasoning:
                result['ranking']['best'] = {
                    'run': int(run_num),
                    'reasoning': reasoning
                }
            else:
                result['ranking']['best'] = {
                    'run': int(run_num),
                    'reasoning': f'[Insufficient reasoning: {reasoning}]'
                }
        else:
            # No BEST choice found - this is an error
            result['ranking']['best'] = {
                'run': None,
                'reasoning': '[ERROR: No BEST CHOICE provided]'
            }
        
        return result
    
    def evaluate_batch(self, evaluations_json_path: str, output_path: str,
                       limit: Optional[int] = None) -> List[Dict]:
        """
        Evaluate all personas in phase1_evaluations.json.
        
        Args:
            evaluations_json_path: Path to phase1_evaluations.json
            output_path: Where to save phase1_results.json
            limit: Evaluate only first N (for testing)
        
        Returns:
            List of evaluation results
        """
        
        with open(evaluations_json_path, 'r') as f:
            evaluations = json.load(f)
        
        if limit:
            evaluations = evaluations[:limit]
        
        print(f"\n{'='*70}")
        print(f"PHASE 1 EVALUATION ({len(evaluations)} personas × trees)")
        print(f"Model: {self.model}")
        print(f"{'='*70}\n")
        
        results = []
        start_time = time.time()
        
        for i, eval_entry in enumerate(evaluations, 1):
            print(f"[{i}/{len(evaluations)}] {eval_entry['eval_id']}")
            
            try:
                response = self.simulate_persona(
                    persona_id=eval_entry['persona'],
                    tree_metrics=eval_entry['tree_metrics'],
                    prompt=eval_entry['prompt']
                )
                
                # Add metadata
                response['eval_id'] = eval_entry['eval_id']
                response['run_num'] = eval_entry['run_num']
                response['tree_metrics'] = eval_entry['tree_metrics']
                
                results.append(response)
                
                # Small delay between requests to avoid overload
                time.sleep(0.5)
                
            except Exception as e:
                print(f"    ✗ Error: {e}")
                results.append({
                    'eval_id': eval_entry['eval_id'],
                    'persona': eval_entry['persona'],
                    'error': str(e),
                })
        
        elapsed = time.time() - start_time
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"✓ Saved {len(results)} evaluations to {output_path}")
        print(f"  Time elapsed: {elapsed/60:.1f} minutes")
        print(f"  Avg per evaluation: {elapsed/len(results):.1f} seconds")
        print(f"{'='*70}\n")
        
        return results


# ============================================================================
# QUICK ANALYSIS: Summarize Phase 1 Results
# ============================================================================

def analyze_phase1_results(results_json_path: str):
    """Quick analysis of Phase 1 results."""
    
    with open(results_json_path, 'r') as f:
        results = json.load(f)
    
    print(f"\n{'='*70}")
    print("PHASE 1 RESULTS ANALYSIS")
    print(f"{'='*70}\n")
    
    # Clarity by persona
    clarity_by_persona = {}
    for r in results:
        if 'error' in r:
            continue
        persona = r['persona']
        rating = r['clarity_rating']
        if persona not in clarity_by_persona:
            clarity_by_persona[persona] = []
        if rating:
            clarity_by_persona[persona].append(rating)
    
    print("CLARITY RATINGS BY PERSONA:")
    print(f"{'Persona':<15} {'Avg':<6} {'Min':<6} {'Max':<6} {'Count':<6}")
    print("-" * 45)
    
    for persona in sorted(clarity_by_persona.keys()):
        ratings = clarity_by_persona[persona]
        if ratings:
            avg = sum(ratings) / len(ratings)
            print(f"{persona:<15} {avg:>5.1f} {min(ratings):>5} {max(ratings):>5} {len(ratings):>5}")
    
    # Identify struggling personas (avg clarity ≤ 2.5)
    print("\n❗ PERSONAS WHO STRUGGLED (avg clarity ≤ 2.5):")
    struggling = [
        p for p, ratings in clarity_by_persona.items()
        if ratings and sum(ratings)/len(ratings) <= 2.5
    ]
    if struggling:
        for p in struggling:
            ratings = clarity_by_persona[p]
            avg = sum(ratings) / len(ratings)
            print(f"  - {p}: {avg:.1f}/5.0 (needs Phase 2 customization)")
    else:
        print("  None! All personas understood the base explanation.")
    
    # Acceptance summary
    print("\n\nACCEPTANCE SUMMARY:")
    acceptance_counts = {'yes': 0, 'no': 0, 'partially': 0}
    for r in results:
        if 'error' not in r and r.get('acceptance'):
            acceptance = r['acceptance'].lower().split()[0]
            if acceptance in acceptance_counts:
                acceptance_counts[acceptance] += 1
    
    total = sum(acceptance_counts.values())
    for key, count in acceptance_counts.items():
        pct = 100 * count / total if total > 0 else 0
        print(f"  {key.capitalize():<10}: {count:>3} ({pct:>5.1f}%)")
    
    # Tree preference analysis (BEST choices only)
    print("\n\nTREE PREFERENCES (BEST CHOICE BY PERSONA):")
    print("=" * 70)
    
    best_choices_by_persona = {}
    for r in results:
        if 'error' not in r and 'ranking' in r:
            persona = r['persona']
            ranking = r.get('ranking', {})
            if ranking.get('best'):
                best_info = ranking['best']
                run = best_info['run']
                reasoning = best_info.get('reasoning', '[no reasoning provided]')
                
                # Shorten reasoning if too long
                if len(reasoning) > 120:
                    reasoning = reasoning[:120] + "..."
                
                best_choices_by_persona[persona] = {'run': run, 'reasoning': reasoning}
                print(f"\n{persona}:")
                print(f"  ✓ Run {run}")
                print(f"  → {reasoning}")
    
    # Count preferences by run
    print("\n\nTREE PREFERENCE COUNTS:")
    print("-" * 40)
    run_counts = {24: 0, 22: 0, 44: 0, 26: 0}
    for persona, choice in best_choices_by_persona.items():
        run = choice['run']
        if run in run_counts:
            run_counts[run] += 1
    
    total = sum(run_counts.values())
    print(f"{'Run':<8} {'Count':<8} {'Percentage':<12}")
    print("-" * 40)
    for run in sorted(run_counts.keys(), key=lambda x: run_counts[x], reverse=True):
        count = run_counts[run]
        pct = 100 * count / total if total > 0 else 0
        print(f"Run {run:<2} {count:<7} {pct:>6.1f}%")
    
    # Convergence check
    print("\n\n🎯 CONVERGENCE TOWARD PARETO KNEE (Run 22 or 44):")
    print("-" * 50)
    
    best_choices = [choice['run'] for choice in best_choices_by_persona.values()]
    if best_choices:
        knee_choices = sum(1 for r in best_choices if r in [22, 44])
        convergence_pct = 100 * knee_choices / len(best_choices)
        
        print(f"Personas choosing knee (22 or 44): {knee_choices}/{len(best_choices)} = {convergence_pct:.0f}%")
        
        if convergence_pct >= 60:
            print(f"✓ STRONG CONVERGENCE → H6 SUPPORTED")
        elif convergence_pct >= 40:
            print(f"~ MODERATE convergence")
        else:
            print(f"✗ WEAK convergence → preferences diverge")
    
    print(f"\n{'='*70}\n")





# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == '__main__':
    import sys
    
    print("""
╔════════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║              Ollama Persona Simulator for Phase 1 Evaluation           ║
║                                                                        ║
║  This simulates 9 personas evaluating your decision trees locally      ║
║  (completely free, no API costs!)                                      ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
""")
    
    print("\n1. PREREQUISITES:")
    print("   Install Ollama from: https://ollama.ai/")
    print("\n2. CHOOSE A MODEL:")
    print("   " + "-" * 60)
    for model_name, info in RECOMMENDED_MODELS.items():
        print(f"   • {model_name:<15} {info['size']:<10} {info['speed']:<12} {info['quality']}")
        print(f"     {info['install']}")
    print("   " + "-" * 60)
    
    print("\n3. START OLLAMA:")
    print("   Open a terminal and run: ollama serve")
    print("   (Keep it running in the background)")
    
    print("\n4. RUN PHASE 1 EVALUATION:")
    print("   python3 ollama_simulator.py phase1_evaluations.json results.json")
    print("   OR for quick test (first 3 evaluations):")
    print("   python3 ollama_simulator.py phase1_evaluations.json results.json --limit 3")
    
    print("\n" + "="*70 + "\n")
    
    # Parse arguments
    if len(sys.argv) < 3:
        print("Usage: python3 ollama_simulator.py <input.json> <output.json> [--limit N] [--model MODEL]")
        print("\nExample:")
        print("  python3 ollama_simulator.py phase1_evaluations.json phase1_results.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    limit = None
    model = OLLAMA_CONFIG['model']
    
    # Parse optional args
    for i, arg in enumerate(sys.argv[3:]):
        if arg == '--limit' and i+3 < len(sys.argv):
            limit = int(sys.argv[i+4])
        elif arg == '--model' and i+3 < len(sys.argv):
            model = sys.argv[i+4]
    
    # Run evaluation
    simulator = OllamaPersonaSimulator(model=model)
    results = simulator.evaluate_batch(input_file, output_file, limit=limit)
    
    # Analyze
    analyze_phase1_results(output_file)
    
    print("\n✓ Phase 1 evaluation complete!")
    print(f"  Results saved to: {output_file}")
    print(f"\nNext step: Analyze results and plan Phase 2 customizations")
