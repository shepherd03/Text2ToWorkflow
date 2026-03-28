import os
import sys
import json
import random
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utr_generation.pipeline import UTRGenerationPipeline
from src.core.config import load_settings
from src.core.utils import format_json_for_readability

def generate_utrs():
    print("Starting UTR Generation Batch Process...")
    settings = load_settings()
    pipeline = UTRGenerationPipeline(settings)
    
    # Use timestamp for unique output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("generated_data", "utr_generation", f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "utrs.jsonl")
    latest_file = os.path.join("generated_data", "utr_generation", "utrs.jsonl")
    
    print(f"Output will be saved to: {output_file} and {latest_file}")
    
    # Load all data from dataset
    all_data = []
    with open("dataset/dataset.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_data.append(json.loads(line))
                
    # Define specific IDs we want to test to ensure we cover conditionals and loops
    target_ids = ["wf_0001", "wf_0003", "wf_0004", "wf_0005", "wf_0009"]
    
    # Filter dataset for these specific IDs, and add a few random ones if needed
    sampled_data = [d for d in all_data if d["id"] in target_ids]
    
    # Fill up to 50 with random items if we have less than 50
    remaining_count = 50 - len(sampled_data)
    if remaining_count > 0:
        other_data = [d for d in all_data if d["id"] not in target_ids]
        sampled_data.extend(random.sample(other_data, min(remaining_count, len(other_data))))
        
    print(f"Selected {len(sampled_data)} records for processing (including {len(target_ids)} target cases).")
    
    # Clear the latest file if it exists
    if os.path.exists(latest_file):
        os.remove(latest_file)
    
    for data in sampled_data:
        record_id = data["id"]
        instruction = data["instruction"]
        
        print(f"Processing {record_id}: {instruction}")
        try:
            output = pipeline.run(instruction)
            utr = output.utr
            
            # Format JSON for readability while keeping it on one line (JSONL format requires one JSON object per line)
            # We'll use separators=(',', ': ') to ensure standard formatting without newlines
            res = {
                "id": record_id,
                "instruction": instruction,
                "utr": utr.model_dump()
            }
            json_str = json.dumps(res, ensure_ascii=False, separators=(',', ': '))
            
            with open(output_file, "a", encoding="utf-8") as out_f:
                out_f.write(json_str + "\n")
            with open(latest_file, "a", encoding="utf-8") as latest_f:
                latest_f.write(json_str + "\n")
                
            print(f"[OK] Saved {record_id}")
        except Exception as e:
            print(f"Error extracting core elements: {e}")

if __name__ == "__main__":
    generate_utrs()
