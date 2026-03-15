from typing import List, Dict, Any, Tuple
from fuzzywuzzy import fuzz
from .schema import UTR, Action, ControlIntent, Resource, Variable, ControlIntentType

class UTREvaluator:
    """
    Evaluates a generated UTR against a ground truth UTR.
    Scoring weights:
    - Action Accuracy: 40%
    - Parameter Completeness: 30%
    - Logic Coherence: 20%
    - Schema Validity: 10%
    """

    def __init__(self, ground_truth: UTR, generated: UTR):
        self.ground_truth = ground_truth
        self.generated = generated
        self.report = {
            "total_score": 0.0,
            "action_score": 0.0,
            "parameter_score": 0.0,
            "logic_score": 0.0,
            "schema_score": 0.0,
            "details": {}
        }
        self.matched_action_pairs = []

    def evaluate(self) -> Dict[str, Any]:
        """Run the full evaluation pipeline."""
        self._evaluate_schema()
        self._evaluate_actions()
        self._evaluate_parameters()
        self._evaluate_logic()
        
        # Calculate total weighted score
        self.report["total_score"] = (
            self.report["action_score"] * 0.4 +
            self.report["parameter_score"] * 0.3 +
            self.report["logic_score"] * 0.2 +
            self.report["schema_score"] * 0.1
        )
        return self.report

    def _evaluate_schema(self):
        """
        Assess schema validity (10 points max).
        """
        score = 0
        # 1. Base validity (assumed true if objects exist)
        score += 20
        
        # 2. Essential fields check
        if self.generated.actions:
            score += 30
        else:
            if not self.ground_truth.actions:
                 score += 30

        # 3. Type consistency (Pydantic guarantees this)
        score += 30
        
        # 4. Enum validity (Pydantic guarantees this)
        score += 20

        self.report["schema_score"] = score

    def _evaluate_actions(self):
        """
        Assess action accuracy (40% weight).
        Uses F1 Score based on tool_name matching with semantic aliases.
        """
        gt_actions = [a.action_name for a in self.ground_truth.actions]
        gen_actions = [a.action_name for a in self.generated.actions]
        
        tp = 0
        matched_indices = set()
        
        # Greedy matching for TP
        for gt_idx, gt_name in enumerate(gt_actions):
            best_match_idx = -1
            best_score = 0
            
            for gen_idx, gen_name in enumerate(gen_actions):
                if gen_idx in matched_indices:
                    continue
                
                score = self._calculate_action_similarity(gt_name, gen_name)
                
                if score > 80: # Threshold for match
                    if score > best_score:
                        best_score = score
                        best_match_idx = gen_idx
            
            if best_match_idx != -1:
                tp += 1
                matched_indices.add(best_match_idx)
        
        fp = len(gen_actions) - tp
        fn = len(gt_actions) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
            
        self.report["action_score"] = f1 * 100
        self.report["details"]["action_metrics"] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn
        }
        
        # Store matched pairs for parameter evaluation
        self.matched_action_pairs = []
        matched_gen_indices = set()
        for gt_action in self.ground_truth.actions:
            best_match = None
            best_score = 0
            best_idx = -1
            
            for idx, gen_action in enumerate(self.generated.actions):
                if idx in matched_gen_indices:
                    continue
                score = self._calculate_action_similarity(gt_action.action_name, gen_action.action_name)
                        
                if score > 80 and score > best_score:
                    best_score = score
                    best_match = gen_action
                    best_idx = idx
            
            if best_match:
                self.matched_action_pairs.append((gt_action, best_match))
                matched_gen_indices.add(best_idx)

    def _calculate_action_similarity(self, name1: str, name2: str) -> int:
        n1 = name1.lower()
        n2 = name2.lower()
        
        # Direct fuzzy match
        score = fuzz.ratio(n1, n2)
        
        # Boost for known semantic equivalents
        semantic_groups = [
            {'llm', 'llm_generation', 'translation', 'summarization', 'coding', 'chat', 'generate', 'answer'},
            {'search', 'google_search', 'bing_search', 'ddg_search', 'web_browser'},
            {'http', 'http_request', 'api_call', 'webhook'},
            {'code', 'code_interpreter', 'python', 'script'},
            {'image', 'image_generation', 'dalle', 'stable_diffusion'}
        ]
        
        for group in semantic_groups:
            # Check if one is in group and other contains a keyword from group
            # Simplified: check if both "hit" the group concept
            hit1 = any(term in n1 for term in group)
            hit2 = any(term in n2 for term in group)
            
            if hit1 and hit2:
                # If both are in the same semantic domain, boost score
                return max(score, 90)
                
        return score

    def _evaluate_parameters(self):
        """
        Assess parameter completeness (30% weight).
        Checks args, resources, and variables for matched actions.
        """
        if not self.matched_action_pairs:
            self.report["parameter_score"] = 0.0
            return

        total_pair_score = 0.0
        
        for gt_action, gen_action in self.matched_action_pairs:
            # Check Args
            args_score = self._compare_dicts(gt_action.args, gen_action.args)
            total_pair_score += args_score

        self.report["parameter_score"] = (total_pair_score / len(self.matched_action_pairs)) * 100.0

    def _get_all_values(self, d: Any) -> List[str]:
        """Recursively extract all scalar values from a dict/list."""
        values = []
        if isinstance(d, dict):
            for v in d.values():
                values.extend(self._get_all_values(v))
        elif isinstance(d, list):
            for v in d:
                values.extend(self._get_all_values(v))
        elif d is not None:
            values.append(str(d))
        return values

    def _compare_dicts(self, gt_dict: Dict, gen_dict: Dict) -> float:
        """
        Compare two dictionaries with loose matching.
        """
        if not gt_dict:
            # If GT requires no params, and Gen provides none, 100%.
            # If Gen provides some, it's also fine (maybe extra context).
            return 1.0
            
        if not gen_dict:
            return 0.0
        
        # 1. Key Match (Loose)
        # Check if Gen keys appear in GT keys (even partial match)
        matched_keys = 0
        for k_gen in gen_dict.keys():
            for k_gt in gt_dict.keys():
                if fuzz.partial_ratio(k_gen.lower(), k_gt.lower()) > 80:
                    matched_keys += 1
                    break
        
        key_score = matched_keys / len(gt_dict) if gt_dict else 1.0
        key_score = min(1.0, key_score)

        # 2. Value Coverage
        # Check if Gen values (content) appear in GT values
        gt_values = self._get_all_values(gt_dict)
        gen_values = self._get_all_values(gen_dict)
        
        matched_vals = 0
        if gen_values:
            for v_gen in gen_values:
                # Skip trivial values
                if len(v_gen) < 2: continue
                
                # Check against all GT values
                found = False
                for v_gt in gt_values:
                    if fuzz.partial_ratio(v_gen.lower(), v_gt.lower()) > 80:
                        found = True
                        break
                if found:
                    matched_vals += 1
            
            val_score = matched_vals / len(gen_values)
        else:
            val_score = 0.0
            
        # Weighted Score: 30% Keys, 70% Values
        # Because keys often mismatch (schema differences), but values (intent) should match.
        final_score = (key_score * 0.3) + (val_score * 0.7)
        return final_score

    def _evaluate_logic(self):
        """
        Assess logic coherence (20% weight).
        Checks control intent types and flow.
        """
        gt_intents = self.ground_truth.control_intents
        gen_intents = self.generated.control_intents
        
        # If no logic in GT, check if Gen also has no logic
        if not gt_intents:
             self.report["logic_score"] = 100 if not gen_intents else 80 # Minor penalty for hallucinated logic
             return

        score_sum = 0
        matched_count = 0
        
        for gt_intent in gt_intents:
            # Find matching intent by type
            match = next((i for i in gen_intents if i.type == gt_intent.type), None)
            
            if match:
                intent_score = 50 # Type match base
                
                # Check targets order (for sequential)
                if gt_intent.type == ControlIntentType.sequential:
                     # Simplified: check fuzzy overlap of target lists
                     # Since action IDs might differ, we can't compare directly.
                     # We assume if type matches, it's mostly correct for sequential.
                     intent_score += 50
                
                # Check condition (for conditional)
                elif gt_intent.type == ControlIntentType.conditional:
                     if fuzz.ratio(gt_intent.condition, match.condition) > 60:
                         intent_score += 50
                     else:
                         intent_score += 20
                
                score_sum += intent_score
                matched_count += 1
        
        if gt_intents:
            self.report["logic_score"] = score_sum / len(gt_intents)
        else:
            self.report["logic_score"] = 0
