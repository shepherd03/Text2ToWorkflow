import sys
import os
import json
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from utr_generator.utr_evaluator import UTREvaluator
from utr_generator.schema import UTR, Action, ControlIntent, Resource, Variable, ControlIntentType

def test_evaluator():
    # 1. Create Ground Truth UTR
    gt_utr = UTR(
        actions=[
            Action(action_name="google_search", order=1, args={"query": "weather"}),
            Action(action_name="send_email", order=2, args={"to": "boss@test.com", "body": "report"})
        ],
        control_intents=[
            ControlIntent(type=ControlIntentType.sequential, target_actions=["google_search", "send_email"])
        ]
    )
    
    # 2. Create Generated UTR (Perfect Match)
    gen_utr_perfect = UTR(
        actions=[
            Action(action_name="google_search", order=1, args={"query": "weather"}),
            Action(action_name="send_email", order=2, args={"to": "boss@test.com", "body": "report"})
        ],
        control_intents=[
            ControlIntent(type=ControlIntentType.sequential, target_actions=["google_search", "send_email"])
        ]
    )
    
    evaluator = UTREvaluator(gt_utr, gen_utr_perfect)
    report = evaluator.evaluate()
    print("Perfect Match Score:", report["total_score"])
    assert report["total_score"] == 100.0
    
    # 3. Create Generated UTR (Partial Match)
    # - Action: "google_search" matched, "send_email" missing (Recall 0.5)
    # - Parameter: "query" matched
    # - Logic: Partial match
    gen_utr_partial = UTR(
        actions=[
            Action(action_name="google_search", order=1, args={"query": "weather"})
        ],
        control_intents=[]
    )
    
    evaluator_partial = UTREvaluator(gt_utr, gen_utr_partial)
    report_partial = evaluator_partial.evaluate()
    print("Partial Match Score:", report_partial["total_score"])
    
    # Manual calc check:
    # Action: TP=1, FP=0, FN=1. P=1.0, R=0.5. F1=0.666. Score = 0.666 * 100 = 66.6
    # Param: 1 matched action. Params match perfect. Score = 100.
    # Logic: 0 score (no intents generated vs 1 expected).
    # Schema: 100.
    # Total = 66.6*0.4 + 100*0.3 + 0*0.2 + 100*0.1 = 26.64 + 30 + 0 + 10 = 66.64
    assert report_partial["total_score"] > 60

if __name__ == "__main__":
    test_evaluator()
