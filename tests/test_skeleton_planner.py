import json
import uuid
import datetime
from src.core.schema import UTR, UTRMetadata, Action
from src.skeleton_planning.skeleton_planner import SkeletonPlanner

def run_test_case(name, task_desc, actions, deps):
    print(f"\n{'='*50}")
    print(f"Running Test Case: {name}")
    print(f"{'='*50}")
    
    metadata = UTRMetadata(
        task_goal=task_desc,
        core_actions=actions,
        core_resources=[],
        core_variables=[],
        implicit_dependencies=deps
    )
    utr = UTR(
        task_id=str(uuid.uuid4()),
        task_desc=task_desc,
        metadata=metadata,
        create_time=datetime.datetime.now().isoformat()
    )
    planner = SkeletonPlanner()
    
    try:
        tree = planner.plan(utr)
        # Hacky way to remove IDs deeply
        def remove_ids(d):
            if isinstance(d, dict):
                return {k: remove_ids(v) for k, v in d.items() if k != 'id'}
            elif isinstance(d, list):
                return [remove_ids(i) for i in d]
            return d
            
        clean_dump = remove_ids(tree.model_dump())
        print(json.dumps(clean_dump, indent=2, ensure_ascii=False))
        return True, clean_dump
    except Exception as e:
        print(f"Error during planning: {e}")
        return False, str(e)

def test_1_simple_sequential():
    task_desc = "下载数据并清理数据"
    actions = [
        Action(action_id='act_1', action_name='download_data'),
        Action(action_id='act_2', action_name='clean_data'),
    ]
    deps = [
        {"from": "act_1", "to": "act_2", "reason": "需要下载后清理"}
    ]
    run_test_case("1. Simple Sequential", task_desc, actions, deps)

def test_2_parallel():
    task_desc = "下载数据，然后同时清理数据和分析数据，最后发送报告"
    actions = [
        Action(action_id='act_1', action_name='download_data'),
        Action(action_id='act_2', action_name='clean_data'),
        Action(action_id='act_3', action_name='analyze_data'),
        Action(action_id='act_4', action_name='send_report')
    ]
    deps = [
        {"from": "act_1", "to": "act_2", "reason": ""},
        {"from": "act_1", "to": "act_3", "reason": ""},
        {"from": "act_2", "to": "act_4", "reason": ""},
        {"from": "act_3", "to": "act_4", "reason": ""}
    ]
    run_test_case("2. Parallel Block (Clean and Analyze can run in parallel)", task_desc, actions, deps)

def test_3_conditional():
    task_desc = "获取用户输入，如果输入为空则提示错误，否则保存输入"
    actions = [
        Action(action_id='act_1', action_name='get_input'),
        Action(action_id='act_2', action_name='show_error'),
        Action(action_id='act_3', action_name='save_input')
    ]
    deps = [
        {"from": "act_1", "to": "act_2", "reason": "需要根据输入判断"},
        {"from": "act_1", "to": "act_3", "reason": "需要保存获取的输入"}
    ]
    # 这里我们期望 LLM 能够识别出 act_2 需要条件分支
    run_test_case("3. Conditional Block", task_desc, actions, deps)

def test_4_loop():
    task_desc = "获取文件列表，然后遍历并批量处理每个文件"
    actions = [
        Action(action_id='act_1', action_name='get_file_list'),
        Action(action_id='act_2', action_name='process_single_file')
    ]
    deps = [
        {"from": "act_1", "to": "act_2", "reason": "需要先获取列表才能处理"}
    ]
    # 这里我们期望 LLM 能够识别出 act_2 需要循环
    run_test_case("4. Loop Block", task_desc, actions, deps)

if __name__ == "__main__":
    test_1_simple_sequential()
    test_2_parallel()
    test_3_conditional()
    test_4_loop()

