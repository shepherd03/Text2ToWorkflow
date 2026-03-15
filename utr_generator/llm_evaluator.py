from typing import Dict, Any
import json
from .schema import UTR
from .llm_client import DeepSeekClient
from .config import load_settings

class LLMUTREvaluator:
    """
    Evaluates UTR quality using an LLM as a judge.
    Compares Generated UTR against Ground Truth UTR based on the user instruction.
    """

    def __init__(self, client: DeepSeekClient = None):
        if client:
            self.client = client
        else:
            settings = load_settings()
            if not settings.llm_enabled:
                 raise ValueError("LLM is not enabled in settings.")
            self.client = DeepSeekClient(settings)

    def evaluate(self, instruction: str, gt_utr: UTR, gen_utr: UTR) -> Dict[str, Any]:
        """
        Ask LLM to evaluate the generated UTR.
        """
        system_prompt = """You are an expert judge for evaluating Unified Task Representations (UTR).
Your task is to compare a "Generated UTR" against a "Ground Truth UTR" based on a user "Instruction".
You must evaluate the quality of the Generated UTR in terms of semantic equivalence, allowing for different tool names if they represent the same capability.

Evaluate on these dimensions:
1. Intent Match (40%): Does the generated UTR achieve the same goal? (e.g., 'translation' action is equivalent to 'llm_generation' with translation prompt).
2. Parameter Accuracy (30%): Are key parameters present and semantically correct? (Ignore exact string matches for long prompts).
3. Logic/Flow (20%): Is the execution order and control flow (sequential/conditional) correct?
4. Completeness (10%): Are all necessary steps included without unnecessary hallucinations?

Output valid JSON only.
"""

        user_prompt = f"""
Instruction: {instruction}

Ground Truth UTR:
```json
{gt_utr.model_dump_json(indent=2)}
```

Generated UTR:
```json
{gen_utr.model_dump_json(indent=2)}
```

Provide a JSON output with this structure:
{{
    "intent_score": <int 0-100>,
    "parameter_score": <int 0-100>,
    "logic_score": <int 0-100>,
    "completeness_score": <int 0-100>,
    "total_score": <float 0-100>,
    "reasoning": "<concise explanation of the scores, highlighting matches and mismatches>"
}}
"""
        
        try:
            # We use a slightly higher temperature for evaluation reasoning
            response = self.client.chat_json(system_prompt, user_prompt, temperature=0.2)
            
            # Ensure total score is calculated if not present or verify it
            if "total_score" not in response:
                response["total_score"] = (
                    response.get("intent_score", 0) * 0.4 +
                    response.get("parameter_score", 0) * 0.3 +
                    response.get("logic_score", 0) * 0.2 +
                    response.get("completeness_score", 0) * 0.1
                )
            
            return response
            
        except Exception as e:
            print(f"LLM Evaluation failed: {e}")
            return {
                "intent_score": 0,
                "parameter_score": 0,
                "logic_score": 0,
                "completeness_score": 0,
                "total_score": 0,
                "reasoning": f"Evaluation failed: {str(e)}"
            }
