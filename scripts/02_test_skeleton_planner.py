import os
import sys
import json

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.schema import UTR
from src.skeleton_planning.skeleton_planner import SkeletonPlanner
from src.core.utils import format_json_for_readability

def run_skeleton_tests():
    print("Starting SkeletonPlanner Batch Test...")
    input_file = os.path.join("generated_data", "utr_generation", "utrs.jsonl")
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found. Please run 01_generate_utrs.py first.")
        return

    output_dir = os.path.join("generated_data", "skeleton_planning", "iter2")
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "skeletons.jsonl")
    error_file = os.path.join(output_dir, "errors.json")
    
    planner = SkeletonPlanner()
    results = []
    errors = []
    
    # Read generated UTRs
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            record_id = data["id"]
            instruction = data["instruction"]
            utr_dict = data["utr"]
            
            print(f"Planning skeleton for {record_id}...")
            try:
                # Convert dict back to UTR model
                utr = UTR(**utr_dict)
                tree = planner.plan(utr)
                
                # Hacky way to remove IDs deeply for cleaner visual inspection
                def remove_ids(d):
                    if isinstance(d, dict):
                        return {k: remove_ids(v) for k, v in d.items() if k != 'id'}
                    elif isinstance(d, list):
                        return [remove_ids(i) for i in d]
                    return d
                    
                clean_dump = remove_ids(tree.model_dump())
                
                results.append({
                    "id": record_id,
                    "instruction": instruction,
                    "skeleton": clean_dump
                })
            except Exception as e:
                print(f"[ERROR] on {record_id}: {str(e)}")
                errors.append({
                    "id": record_id,
                    "error": str(e)
                })

    # Save Results
    with open(output_file, "w", encoding="utf-8") as out_f:
        for res in results:
            out_f.write(format_json_for_readability(res) + "\n")
            
    # Save Errors
    with open(error_file, "w", encoding="utf-8") as err_f:
        err_f.write(format_json_for_readability(errors))

    print(f"\n[OK] Completed. Success: {len(results)}, Errors: {len(errors)}")
    print(f"Results saved to: {output_file}")
    if errors:
        print(f"Errors saved to: {error_file}")

if __name__ == "__main__":
    run_skeleton_tests()
