import os
import sys
import glob
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from main import UTRGenerationPipeline
from utr_generator.dify_dsl_converter import DifyDSLConverter
from utr_generator.utr_evaluator import UTREvaluator
from utr_generator.schema import UTR

# Load environment variables for API key
load_dotenv(override=True)

class BatchEvaluator:
    def __init__(self, dsl_dir: str, output_dir: str = "evaluation_results"):
        self.dsl_dir = dsl_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.pipeline = UTRGenerationPipeline()
        
    def run_evaluation(self, limit: int = None):
        """
        Run evaluation on a batch of DSL files.
        :param limit: Max number of files to process (for testing)
        """
        dsl_files = glob.glob(os.path.join(self.dsl_dir, "*.yml")) + \
                    glob.glob(os.path.join(self.dsl_dir, "*.yaml"))
        
        if limit:
            dsl_files = dsl_files[:limit]
            
        results = []
        
        print(f"Starting evaluation on {len(dsl_files)} files...")
        
        for dsl_file in tqdm(dsl_files, desc="Evaluating"):
            file_name = os.path.basename(dsl_file)
            try:
                # 1. Convert DSL to Ground Truth UTR
                converter = DifyDSLConverter(dsl_file)
                gt_utr = converter.convert()
                
                # Extract natural language description (Task Input)
                task_description = converter.extract_description()
                
                if not task_description or len(gt_utr.actions) == 0:
                    print(f"Skipping {file_name}: No description or actions found.")
                    continue
                
                # 2. Generate UTR using Model (Candidate)
                # We use the pipeline directly.
                # Note: Pipeline returns a UTR object.
                pipeline_output = self.pipeline.run(task_description)
                gen_utr = pipeline_output.utr
                
                # 3. Evaluate Candidate against Ground Truth
                evaluator = UTREvaluator(gt_utr, gen_utr)
                report = evaluator.evaluate()
                
                # 4. Record Results
                result_entry = {
                    "file_name": file_name,
                    "task_description": task_description[:100] + "..." if len(task_description) > 100 else task_description,
                    "total_score": report["total_score"],
                    "action_score": report["action_score"],
                    "parameter_score": report["parameter_score"],
                    "logic_score": report["logic_score"],
                    "schema_score": report["schema_score"],
                    "gt_action_count": len(gt_utr.actions),
                    "gen_action_count": len(gen_utr.actions)
                }
                results.append(result_entry)
                
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                
        # 5. Save Summary
        if results:
            df = pd.DataFrame(results)
            csv_path = os.path.join(self.output_dir, "batch_evaluation_summary.csv")
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"\nEvaluation complete. Results saved to {csv_path}")
            print(f"Average Score: {df['total_score'].mean():.2f}")
        else:
            print("No valid results generated.")

if __name__ == "__main__":
    # Path to user's DSL directory
    DSL_DIR = r"e:\Desktop\论文\工作流\utr\dify_workflows"
    
    # Run for 5-10 samples as requested
    evaluator = BatchEvaluator(DSL_DIR)
    evaluator.run_evaluation(limit=10)
