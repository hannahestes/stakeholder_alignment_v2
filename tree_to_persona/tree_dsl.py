"""
Tree DSL: Domain-Specific Language for EZR Decision Trees
Inspired by ASE16's compartmental modeling DSL pattern.

This DSL converts EZR raw tree output into structured objects
that can generate persona-tailored explanations.

Pattern: Superclasses handle generic tree logic;
users/generators specify domain-specific explanation rules.
"""

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


# ============================================================================
# BASIC OBJECT CLASS (inspired by ASE16's 'o' class)
# ============================================================================

class TreeNode:
    """Base class for tree components. Emulates simple Python object."""
    
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.to_dict()})"


# ============================================================================
# DOMAIN-SPECIFIC TREE CLASSES
# ============================================================================

class Split(TreeNode):
    """A decision node: splits data on a feature."""
    
    def __init__(self, feature, operator, threshold, samples, win, depth=0, **kwargs):
        self.feature = feature          # e.g., "Avg_Utilization_Ratio"
        self.operator = operator        # "==" or ">" or "<="
        self.threshold = threshold      # e.g., 0.11 or "$80K - $120K"
        self.samples = samples          # number of rows at this split
        self.win = win                  # performance metric
        self.depth = depth              # tree depth (for indentation)
        self.left = None                # true branch
        self.right = None               # false branch
        super().__init__(**kwargs)
    
    def is_leaf(self):
        return self.left is None and self.right is None
    
    def is_strong(self, min_win=50):
        """Is this a high-performing leaf?"""
        return self.win >= min_win


class Leaf(TreeNode):
    """A terminal node: a decision outcome."""
    
    def __init__(self, win, samples, depth=0, **kwargs):
        self.win = win
        self.samples = samples
        self.depth = depth
        self.path = []  # list of (feature, operator, threshold) tuples
        super().__init__(**kwargs)
    
    def is_strong(self, min_win=50):
        return self.win >= min_win
    
    def is_weak(self, max_win=0):
        return self.win <= max_win


class TreeStructure(TreeNode):
    """The complete tree: root, metrics, feature list."""
    
    def __init__(self, run_num, features, complexity, win, **kwargs):
        self.run_num = run_num
        self.features = features        # list of feature names
        self.complexity = complexity    # number of unique features
        self.win = win                  # overall tree performance
        self.root = None                # root split node
        self.leaves = []                # all leaf nodes
        self.all_nodes = []             # all nodes (for traversal)
        super().__init__(**kwargs)
    
    def get_key_splits(self, depth=1):
        """Return splits at a certain depth (e.g., root splits)."""
        result = []
        for node in self.all_nodes:
            if hasattr(node, 'depth') and node.depth == depth and not node.is_leaf():
                result.append(node)
        return result
    
    def get_strong_leaves(self, min_win=50):
        return [leaf for leaf in self.leaves if leaf.is_strong(min_win)]
    
    def get_weak_leaves(self, max_win=0):
        return [leaf for leaf in self.leaves if leaf.is_weak(max_win)]


# ============================================================================
# PARSER: Convert raw EZR output to TreeStructure
# ============================================================================

class TreeParser(TreeNode):
    """
    Parses EZR's raw tree output format into structured TreeStructure.
    
    Raw format example:
        #rows  win
             12   24    if Avg_Utilization_Ratio <= 0.11
              8   60    |  if Avg_Utilization_Ratio > 0
              3   76    |  |  if education_Level == Graduate;
    """
    
    def parse(self, raw_output, run_num, features, complexity, win):
        """Parse raw output into TreeStructure."""
        tree = TreeStructure(
            run_num=run_num,
            features=features,
            complexity=complexity,
            win=win
        )
        
        lines = raw_output.strip().split('\n')
        
        # Skip header
        lines = [l for l in lines if l.strip() and not l.strip().startswith('#rows')]
        
        # Build tree from lines
        nodes_by_depth = defaultdict(list)
        
        for line in lines:
            if not line.strip():
                continue
            
            # Extract depth from indentation ("|" count)
            depth = line.count('|')
            
            # Parse the line
            parsed = self._parse_line(line, depth)
            if parsed:
                node_type, node_data = parsed
                
                if node_type == 'split':
                    split_node = Split(**node_data)
                    nodes_by_depth[depth].append(split_node)
                    tree.all_nodes.append(split_node)
                
                elif node_type == 'leaf':
                    leaf_node = Leaf(**node_data)
                    leaf_node.depth = depth
                    nodes_by_depth[depth].append(leaf_node)
                    tree.all_nodes.append(leaf_node)
                    tree.leaves.append(leaf_node)
        
        # Set root
        if nodes_by_depth[0]:
            tree.root = nodes_by_depth[0][0]
        
        return tree
    
    def _parse_line(self, line, depth):
        """
        Parse a single tree line.
        Returns (node_type, node_data_dict) or None.
        """
        # Remove indentation markers
        clean = line.replace('|', '').strip()
        
        # Pattern: #rows win [if condition;]
        # Examples:
        #   "12   24    if Avg_Utilization_Ratio <= 0.11"
        #   "8   60    |  if Avg_Utilization_Ratio > 0"
        #   "3   76    |  |  if education_Level == Graduate;"
        
        if not clean:
            return None
        
        parts = clean.split()
        if len(parts) < 2:
            return None
        
        try:
            samples = int(parts[0])
            win = int(parts[1])
        except ValueError:
            return None
        
        # Check if this is a split (has "if") or leaf
        if 'if' in clean:
            # It's a split
            if_idx = clean.index('if')
            condition = clean[if_idx + 2:].strip(';').strip()
            
            # Parse condition: "feature operator value"
            feature, operator, threshold = self._parse_condition(condition)
            
            if feature:
                return ('split', {
                    'feature': feature,
                    'operator': operator,
                    'threshold': threshold,
                    'samples': samples,
                    'win': win,
                    'depth': depth
                })
        else:
            # It's a leaf (no condition)
            return ('leaf', {
                'win': win,
                'samples': samples,
                'depth': depth
            })
        
        return None
    
    def _parse_condition(self, condition):
        """
        Parse a condition string.
        Examples:
            "income_Category == $80K - $120K"
            "Avg_Utilization_Ratio <= 0.11"
            "education_Level == Graduate"
        Returns: (feature, operator, threshold)
        """
        # Try to find operator
        for op in ['==', '<=', '>=', '<', '>', '!=']:
            if op in condition:
                parts = condition.split(op, 1)
                if len(parts) == 2:
                    feature = parts[0].strip()
                    threshold = parts[1].strip()
                    
                    # Try to convert threshold to number
                    try:
                        threshold = float(threshold)
                    except ValueError:
                        pass
                    
                    return (feature, op, threshold)
        
        return (None, None, None)


# ============================================================================
# ANALYZER: Extract patterns from TreeStructure
# ============================================================================

class TreeAnalyzer(TreeNode):
    """
    Analyze tree structure to extract insights for explanation generation.
    """
    
    def analyze(self, tree):
        """Return analysis dict with key patterns."""
        analysis = {
            'run_num': tree.run_num,
            'complexity': tree.complexity,
            'overall_win': tree.win,
            'num_features': len(tree.features),
            'features': tree.features,
            'root_feature': None,
            'root_split': None,
            'strong_leaves': tree.get_strong_leaves(min_win=50),
            'weak_leaves': tree.get_weak_leaves(max_win=0),
            'num_leaves': len(tree.leaves),
            'avg_leaf_win': self._avg_win(tree.leaves),
            'key_branches': self._extract_branches(tree),
        }
        
        if tree.root and hasattr(tree.root, 'feature'):
            analysis['root_feature'] = tree.root.feature
            analysis['root_split'] = tree.root
        
        return analysis
    
    def _avg_win(self, nodes):
        if not nodes:
            return 0
        return sum(n.win for n in nodes) / len(nodes)
    
    def _extract_branches(self, tree, max_depth=2):
        """Extract key decision paths up to max_depth."""
        branches = []
        
        def traverse(node, path, depth):
            if depth > max_depth:
                return
            
            if hasattr(node, 'feature'):  # Split
                new_path = path + [(node.feature, node.operator, node.threshold)]
                branches.append({
                    'path': new_path,
                    'win': node.win,
                    'samples': node.samples
                })
                if node.left:
                    traverse(node.left, new_path, depth + 1)
                if node.right:
                    traverse(node.right, new_path, depth + 1)
        
        if tree.root:
            traverse(tree.root, [], 0)
        
        return branches


# ============================================================================
# EXPLANATION GENERATOR: Create persona hooks for explanations
# ============================================================================

class ExplanationTemplate(TreeNode):
    """
    Template for generating explanations.
    Stores both base description and persona-specific hooks.
    """
    
    def __init__(self, tree, analysis):
        self.tree = tree
        self.analysis = analysis
        self.base_description = None
        self.persona_hooks = {}  # persona_id -> customization dict
        super().__init__()
    
    def generate_base(self):
        """Generate base explanation (Phase 1)."""
        a = self.analysis
        
        base = f"""
TREE RUN {a['run_num']} | Complexity: {a['complexity']} | Overall Win: {a['overall_win']}

STRUCTURE:
- Uses {a['num_features']} features: {', '.join(a['features'])}
- Makes {len(a['key_branches'])} key decision points
- Produces {a['num_leaves']} distinct recommendations

ROOT DECISION:
The tree starts by splitting on "{a['root_feature']}", which is the most important factor.

KEY FINDINGS:
- Strong outcomes (win >= 50): {len(a['strong_leaves'])} branches
- Weak outcomes (win <= 0): {len(a['weak_leaves'])} branches
- Average performance across leaves: {a['avg_leaf_win']:.1f}

EXPLANATION:
[Will be expanded by LLM based on tree structure]
"""
        self.base_description = base.strip()
        return self.base_description
    
    def add_persona_hook(self, persona_id, hook_dict):
        """
        Add persona-specific customization.
        
        Examples:
        add_persona_hook('SWE-Abi', {
            'emphasis': 'stability',
            'detail_level': 'feature-level',
            'metrics': ['stability %', 'coverage']
        })
        """
        self.persona_hooks[persona_id] = hook_dict
    
    def get_persona_prompt(self, persona_id, base_prompt=None):
        """
        Get LLM prompt for a specific persona.
        This merges base description with persona hooks.
        """
        if persona_id not in self.persona_hooks:
            # Phase 1: no persona customization yet
            return base_prompt or self.base_description
        
        hook = self.persona_hooks[persona_id]
        
        # Phase 2: customize the base
        customized = (base_prompt or self.base_description) + f"\n\nPERSONA CUSTOMIZATION ({persona_id}):\n"
        
        if 'emphasis' in hook:
            customized += f"- Emphasize: {hook['emphasis']}\n"
        
        if 'detail_level' in hook:
            customized += f"- Detail level: {hook['detail_level']}\n"
        
        if 'priority' in hook:
            customized += f"- Priority: {hook['priority']}\n"
        
        return customized


# ============================================================================
# UTILITY: Load and process multiple trees
# ============================================================================

class TreeDSL(TreeNode):
    """
    High-level orchestrator: load, parse, analyze, generate.
    """
    
    def __init__(self):
        self.parser = TreeParser()
        self.analyzer = TreeAnalyzer()
        self.trees = {}  # run_num -> TreeStructure
        self.analyses = {}  # run_num -> analysis dict
        self.templates = {}  # run_num -> ExplanationTemplate
    
    def load_from_json(self, json_path, run_nums):
        """Load specific runs from results.json."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        for tree_data in data['all_trees']:
            if tree_data['run_num'] in run_nums:
                self._process_tree(tree_data)
    
    def _process_tree(self, tree_data):
        """Parse a single tree and generate explanation template."""
        run_num = tree_data['run_num']
        
        # Phase 1: Parse
        tree = self.parser.parse(
            tree_data['raw_output'],
            tree_data['run_num'],
            tree_data['features'],
            tree_data['complexity'],
            tree_data['win']
        )
        self.trees[run_num] = tree
        
        # Phase 2: Analyze
        analysis = self.analyzer.analyze(tree)
        self.analyses[run_num] = analysis
        
        # Phase 3: Generate template
        template = ExplanationTemplate(tree, analysis)
        template.generate_base()
        self.templates[run_num] = template
    
    def get_template(self, run_num):
        """Get explanation template for a tree."""
        return self.templates.get(run_num)
    
    def summary(self):
        """Print summary of all loaded trees."""
        for run_num in sorted(self.trees.keys()):
            a = self.analyses[run_num]
            print(f"\nRun {run_num}:")
            print(f"  Complexity: {a['complexity']}, Win: {a['overall_win']}")
            print(f"  Features: {', '.join(a['features'][:3])}...")
            print(f"  Root split: {a['root_feature']}")


if __name__ == '__main__':
    # Test: Load pareto frontier
    dsl = TreeDSL()
    dsl.load_from_json('dt_results.json', [22, 24, 26, 44])
    dsl.summary()
    
    # Get a template
    print("\n" + "="*70)
    template = dsl.get_template(22)
    print(template.generate_base())
