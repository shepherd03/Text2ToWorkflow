import os
import json
import yaml
import tempfile
import pandas as pd
from tqdm import tqdm
from utr_generator.pipeline import UTRGenerationPipeline
from utr_generator.dify_dsl_converter import DifyDSLConverter
from utr_generator.utr_evaluator import UTREvaluator
from utr_generator.llm_evaluator import LLMUTREvaluator

def main():
    dataset_path = r"e:\Desktop\论文\工作流\utr\dataset\dataset.jsonl"
    output_dir = r"e:\Desktop\论文\工作流\utr\evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    output_jsonl_path = os.path.join(output_dir, "dataset_evaluation_details.jsonl")
    output_csv_path = os.path.join(output_dir, "dataset_evaluation_summary.csv")

    # Load existing results to resume
    processed_ids = set()
    if os.path.exists(output_jsonl_path):
        with open(output_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        processed_ids.add(data.get("id"))
                    except:
                        pass
    print(f"Resuming evaluation. Found {len(processed_ids)} processed items.")

    pipeline = UTRGenerationPipeline()
    try:
        llm_evaluator = LLMUTREvaluator()
        print("LLM Evaluator initialized.")
    except Exception as e:
        print(f"Warning: LLM Evaluator failed to init ({e}). Skipping LLM evaluation.")
        llm_evaluator = None
    
    results = []
    summary_data = []

    with open(dataset_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Limit to 20 for demo purposes, or process all if desired (comment out slice)
    lines = lines[:20] 

    for line in tqdm(lines, desc="Evaluating dataset"):
        if not line.strip():
            continue
        
        data = json.loads(line)
        wf_id = data.get("id", "unknown")
        
        if wf_id in processed_ids:
            continue
            
        instruction = data.get("instruction", "")
        dsl_content = data.get("dsl", "")
        
        # Write DSL to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as tmp:
            tmp.write(dsl_content)
            tmp_path = tmp.name
        
        try:
            # Get Ground Truth UTR
            converter = DifyDSLConverter(tmp_path)
            gt_utr = converter.convert()
            
            # Generate UTR
            pipeline_output = pipeline.run(instruction)
            gen_utr = pipeline_output.utr
            
            # Rule-based Evaluate
            evaluator = UTREvaluator(gt_utr, gen_utr)
            rule_scores = evaluator.evaluate()

            # LLM Evaluate
            llm_scores = {}
            if llm_evaluator:
                try:
                    llm_scores = llm_evaluator.evaluate(instruction, gt_utr, gen_utr)
                except Exception as e:
                    print(f"LLM eval error for {wf_id}: {e}")

            # Prepare result object
            result_obj = {
                "id": wf_id,
                "instruction": instruction,
                "rule_scores": rule_scores,
                "llm_scores": llm_scores,
                "gt_utr": gt_utr.model_dump(),
                "gen_utr": gen_utr.model_dump()
            }
            
            # Save immediately to JSONL
            with open(output_jsonl_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result_obj, ensure_ascii=False) + "\n")
            
            # Add to summary list (memory)
            # No need to store in memory if we read from file at end
            
        except Exception as e:
            print(f"Error evaluating {wf_id}: {str(e)}")
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except:
                    pass

    # Read all results for final CSV summary (including previously processed ones)
    all_summary_data = []
    if os.path.exists(output_jsonl_path):
        with open(output_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        res = json.loads(line)
                        r_scores = res.get("rule_scores", {})
                        l_scores = res.get("llm_scores", {})
                        gt = res.get("gt_utr", {})
                        gen = res.get("gen_utr", {})
                        
                        all_summary_data.append({
                            "id": res.get("id"),
                            "instruction": res.get("instruction"),
                            # Rule Metrics
                            "rule_total": r_scores.get("total_score", 0),
                            "rule_action": r_scores.get("action_score", 0),
                            "rule_param": r_scores.get("parameter_score", 0),
                            "rule_logic": r_scores.get("logic_score", 0),
                            # LLM Metrics
                            "llm_total": l_scores.get("total_score", 0),
                            "llm_intent": l_scores.get("intent_score", 0),
                            "llm_param": l_scores.get("parameter_score", 0),
                            "llm_logic": l_scores.get("logic_score", 0),
                            "llm_reasoning": l_scores.get("reasoning", ""),
                            # Counts
                            "gt_actions": len(gt.get("actions", [])),
                            "gen_actions": len(gen.get("actions", []))
                        })
                    except:
                        pass

    # Save summary CSV
    if all_summary_data:
        df = pd.DataFrame(all_summary_data)
        df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        
        print(f"\nEvaluation complete. Results saved to {output_dir}")
        print(f"Processed {len(all_summary_data)} items.")
        print(f"Avg Rule Total: {df['rule_total'].mean():.2f}")
        print(f"Avg LLM Total: {df['llm_total'].mean():.2f}")
    else:
        print("No valid results generated.")

if __name__ == "__main__":
    main()
