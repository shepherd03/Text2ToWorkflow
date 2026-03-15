import sys
import os
import json
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from utr_generator.dify_dsl_converter import DifyDSLConverter

def test_convert():
    fixture_path = Path(__file__).parent / "fixtures" / "dify_dsl_v2.yml"
    converter = DifyDSLConverter(str(fixture_path))
    utr = converter.convert()
    
    print("Converted UTR:")
    print(utr.model_dump_json(indent=2))
    
    # Assertions
    # We expect 2 actions: google_search and llm
    assert len(utr.actions) == 2
    # Check action names based on logic
    # The first action is tool_1 (google_search)
    assert utr.actions[0].action_name == "google_search"
    # The second action is llm_1 (llm_gpt-3.5-turbo)
    assert "gpt-3.5-turbo" in utr.actions[1].action_name
    
    assert len(utr.variables) == 1
    assert utr.variables[0].name == "query"
    
    assert len(utr.control_intents) == 1
    assert utr.control_intents[0].type == "sequential"

if __name__ == "__main__":
    test_convert()
