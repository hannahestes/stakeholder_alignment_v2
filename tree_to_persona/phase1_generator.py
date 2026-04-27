"""
Phase 1 Generator: Create base explanations for all Pareto frontier trees
All 9 personas evaluate the same explanation.

Output: JSON file with 36 evaluation prompts (4 trees × 9 personas)
"""

import json
from tree_dsl import TreeDSL, ExplanationTemplate


class Phase1Generator:
    """Generate Phase 1 base explanations."""
    
    def __init__(self, dsl):
        self.dsl = dsl
        self.personas = [
            'PjM-Abi', 'PjM-Pat', 'PjM-Tim',
            'PdM-Abi', 'PdM-Pat', 'PdM-Tim',
            'SWE-Abi', 'SWE-Pat', 'SWE-Tim'
        ]
    
    def generate_all(self, run_nums):
        """Generate Phase 1 prompts for all trees and personas."""
        evaluations = []
        
        for run_num in sorted(run_nums):
            template = self.dsl.get_template(run_num)
            analysis = self.dsl.analyses[run_num]
            
            # Create persona-specific evaluations (each gets their own prompt context)
            for persona in self.personas:
                persona_prompt = self._create_base_prompt(template, analysis, persona_id=persona)
                
                evaluation = {
                    'eval_id': f'{run_num}_{persona}',
                    'run_num': run_num,
                    'persona': persona,
                    'phase': 1,
                    'prompt': persona_prompt,
                    'tree_metrics': {
                        'complexity': analysis['complexity'],
                        'win': analysis['overall_win'],
                        'num_features': analysis['num_features'],
                    }
                }
                evaluations.append(evaluation)
        
        return evaluations
    
    def _create_base_prompt(self, template, analysis, persona_id=None):
        """Create the base explanation prompt with optional persona context."""
        
        features_str = ', '.join(analysis['features'])
        root_feature = analysis['root_feature']
        complexity = analysis['complexity']
        win = analysis['overall_win']
        
        num_strong = len(analysis['strong_leaves'])
        num_weak = len(analysis['weak_leaves'])
        avg_leaf = analysis['avg_leaf_win']
        
        # Persona context
        persona_context = ""
        if persona_id:
            persona_context = self._get_persona_context(persona_id)
        
        prompt = f"""
{persona_context}

---

DECISION TREE EVALUATION: Run {analysis['run_num']}

CONTEXT:
We optimized a decision tree model using real bank customer data. We ran 50 different model configurations, each with different complexity-accuracy trade-offs. From these 50 runs, we identified the 4 BEST trees that represent the Pareto frontier—meaning no tree dominates another (you can't get more accuracy without more complexity, or more simplicity without losing accuracy).

These are the 4 optimal solutions:
- Run 24: 2 features, 68% accuracy (SIMPLEST)
- Run 22: 3 features, 77% accuracy (BALANCED)
- Run 44: 4 features, 98% accuracy (ACCURATE)
- Run 26: 6 features, 100% accuracy (PERFECT but COMPLEX)

You are evaluating Run {analysis['run_num']} in the context of these 4 options.

---

OVERVIEW:
This decision tree is a predictive model that classifies bank customers into segments.
It achieves {win}% accuracy on the dataset and uses {complexity} input features.

TREE STRUCTURE:
The model makes sequential decisions on the following features:
{features_str}

The tree begins by asking ONE key question about "{root_feature}":
  - This single decision point is the most important factor in the model.
  - Based on that answer, it branches into additional decisions.
  - Each path through the tree leads to a final recommendation.

PERFORMANCE METRICS:
- Overall accuracy: {win}%
- Number of decision paths (leaves): {analysis['num_leaves']}
- Paths with strong performance (≥50% accuracy): {num_strong}
- Paths with weak performance (≤0% accuracy): {num_weak}
- Average performance across all paths: {avg_leaf:.1f}%

COMPLEXITY-ACCURACY TRADE-OFF:
This tree uses {complexity} features to achieve {win}% accuracy.
Compare this to simpler trees (using fewer features) or more complex trees
(using more features) to understand the trade-off.

---

EVALUATION TASK:

Please read the above description carefully and respond to the following:

1. CLARITY RATING: [1-5]
   How clear is this tree's logic to you? 
   RATING REASONING: [2-3 sentences explaining your rating]

2. ACCEPTANCE: [yes/no/partially]
   CONCERNS: [If any, describe what concerns you about this tree]

3. LEARNED: [What is one key thing you learned from this explanation?]
   WANT_TO_KNOW: [What would you like to know more about?]

4. YOUR TREE PREFERENCE (REQUIRED - this is the key question):

   Given the 4 Pareto frontier options, which tree would you choose as BEST?
   
   Why are you making this choice? What trade-offs matter most to YOUR role/perspective?
   - Accuracy vs Simplicity?
   - Operational ease vs Performance?
   - Team understanding vs Predictive power?
   
   You MUST answer this section. Provide your answer in exactly this format:

   BEST CHOICE: Run [22/24/26/44]
   WHY: [1-2 sentences explaining your reasoning and which trade-offs you prioritize]
"""
        
        return prompt.strip()
    
    def _get_persona_context(self, persona_id):
        """Get persona profile for context."""
        
        role, style = persona_id.split('-')
        
        style_profiles = {
            'Abi': "Low self-efficacy, comprehensive (wants all info), risk-averse, process-oriented",
            'Pat': "Medium self-efficacy, comprehensive, risk-averse, reflective (learns by trying)",
            'Tim': "High self-efficacy, selective (focuses on key metrics), risk-tolerant, tinkerer"
        }
        
        role_profiles = {
            'PjM': "Project Manager: Non-technical (1/5), values simplicity & predictability",
            'PdM': "Product Manager: Moderately technical (3/5), values customer impact & business outcomes",
            'SWE': "Software Engineer: Highly technical (5/5), values accuracy & maintainability"
        }
        
        context = f"""
PERSONA: {persona_id}

Role Context: {role_profiles.get(role, '')}
Cognitive Style: {style_profiles.get(style, '')}

Your Priorities:
- You are evaluating this tree from YOUR perspective
- Consider what matters most to YOUR role and your decision-making style
- Be honest about your concerns AND your interests"""
        
        return context
    
    def save_to_json(self, evaluations, output_path):
        """Save evaluations to JSON."""
        with open(output_path, 'w') as f:
            json.dump(evaluations, f, indent=2)
        print(f"Saved {len(evaluations)} evaluations to {output_path}")
    
    def print_sample(self, evaluations, run_num, persona):
        """Print a sample evaluation."""
        sample = next(
            (e for e in evaluations if e['run_num'] == run_num and e['persona'] == persona),
            None
        )
        if sample:
            print(f"\n{'='*70}")
            print(f"SAMPLE: Run {run_num}, Persona {persona}")
            print(f"{'='*70}")
            print(sample['prompt'])
        else:
            print("Sample not found.")


if __name__ == '__main__':
    # Load Pareto frontier
    dsl = TreeDSL()
    dsl.load_from_json('dt_results.json', [22, 24, 26, 44])
    
    # Generate Phase 1
    gen = Phase1Generator(dsl)
    evaluations = gen.generate_all([22, 24, 26, 44])
    
    # Save
    gen.save_to_json(evaluations, 'phase1_evaluations.json')
    
    # Print sample for each tree
    for run_num in [22, 24, 26, 44]:
        gen.print_sample(evaluations, run_num, 'SWE-Tim')
