import json
import yaml
import os
from collections import defaultdict

DATASET_PATH = os.path.join("dataset", "dataset.jsonl")
OUTPUT_PATH = os.path.join("generated_data", "utr_generation", "utr_ground_truth.jsonl")

# 定义不需要转换为 core_actions 的控制流/系统节点
CONTROL_NODES = {"start", "end", "if-else", "iteration"}

def parse_dsl_to_utr(dsl_yaml, instruction):
    try:
        dsl = yaml.safe_load(dsl_yaml)
    except Exception as e:
        print(f"YAML parsing error: {e}")
        return None
        
    workflow = dsl.get("workflow", {})
    graph = workflow.get("graph", {})
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    
    # 1. 解析节点
    node_dict = {str(n["id"]): n["data"] for n in nodes}
    
    core_actions = []
    core_variables = []
    
    # 提取变量 (从 start 节点中提取用户输入作为核心变量)
    for node_id, data in node_dict.items():
        if data.get("type") == "start":
            for var in data.get("variables", []):
                core_variables.append({
                    "var_id": f"var_{var.get('variable', '')}",
                    "name": var.get("variable", ""),
                    "type": "string", # 默认当做string
                    "source": "用户输入"
                })
                
    # 提取动作
    action_nodes = set()
    for node_id, data in node_dict.items():
        node_type = data.get("type", "")
        if node_type not in CONTROL_NODES:
            action_nodes.add(node_id)
            core_actions.append({
                "action_id": node_id,
                "action_name": node_type.replace("-", "_"),
                "description": data.get("title", node_type),
                "inputs": [],
                "outputs": []
            })
            
    # 2. 依赖边搭桥 (Edge Bridging)
    # 构建有向图
    adj_list = defaultdict(list)
    for edge in edges:
        src = str(edge.get("source"))
        tgt = str(edge.get("target"))
        adj_list[src].append(tgt)
        
    implicit_dependencies = []
    
    # 寻找从每个 action 节点出发，到达下一个 action 节点的路径（跳过控制节点）
    def find_reachable_actions(start_node, current_node, visited):
        if current_node in visited:
            return []
        visited.add(current_node)
        
        reachable = []
        for neighbor in adj_list[current_node]:
            if neighbor in action_nodes:
                reachable.append(neighbor)
            else:
                # 如果是控制节点，继续往下找
                reachable.extend(find_reachable_actions(start_node, neighbor, visited))
        return reachable

    # 为每个真正的动作节点寻找依赖
    for action_id in action_nodes:
        visited = set()
        reachable_actions = find_reachable_actions(action_id, action_id, visited)
        # 去重
        reachable_actions = list(set(reachable_actions))
        
        for target_action in reachable_actions:
            implicit_dependencies.append({
                "from": action_id,
                "to": target_action,
                "reason": f"{node_dict[action_id].get('title', '节点')} 是 {node_dict[target_action].get('title', '节点')} 的前置条件"
            })
            
    # 3. 构建 UTR 结构
    utr_metadata = {
        "task_goal": instruction,
        "core_actions": core_actions,
        "core_resources": [], # 简化处理，暂不从规则提取资源
        "core_variables": core_variables,
        "implicit_dependencies": implicit_dependencies
    }
    
    return utr_metadata

def main():
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at {DATASET_PATH}")
        return
        
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    print(f"Parsing dataset from {DATASET_PATH} ...")
    success_count = 0
    with open(DATASET_PATH, 'r', encoding='utf-8') as fin, \
         open(OUTPUT_PATH, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                dsl_yaml = item.get("dsl", "")
                instruction = item.get("instruction", "")
                
                utr = parse_dsl_to_utr(dsl_yaml, instruction)
                if utr:
                    # 保存为 JSONL
                    record = {
                        "id": item.get("id"),
                        "instruction": instruction,
                        "ground_truth_utr": utr
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + '\n')
                    success_count += 1
            except Exception as e:
                print(f"Error processing line: {e}")
                
    print(f"Done! Successfully generated {success_count} Ground Truth UTRs.")
    print(f"Saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
