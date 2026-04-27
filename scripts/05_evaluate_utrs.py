import json
import os
import networkx as nx
from collections import defaultdict
import sys
from dotenv import load_dotenv

# 加载环境变量 (用于 LLM 调用)
load_dotenv()

# 引入项目中的 Schema 和 LLM 客户端
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from core.schema import UTRMetadata
from core.llm_client import DeepSeekClient
from core.config import load_settings

# 初始化 LLM 客户端用于语义裁判
settings = load_settings()
llm_client = DeepSeekClient(settings)

GT_FILE = os.getenv("UTR_GT_FILE", os.path.join("generated_data", "utr_generation", "utr_ground_truth.jsonl"))
PRED_FILE = os.getenv("UTR_PRED_FILE", os.path.join("generated_data", "utr_generation", "utrs.jsonl"))

# 控制流词汇黑名单 (已扩充，增加严苛度)
CTRL_BLACKLIST = [
    "如果", "循环", "判断", "遍历", "条件", "分支", "每一个", "每个", 
    "if", "loop", "for", "each", "switch", "case", "whether", "while", 
    "当...时", "否则", "else", "否则的话", "若", "是否", "当", "重复", 
    "递归", "或者", "and", "or", "not", "then", "continue", "break"
]

def load_jsonl(filepath):
    data = {}
    if not os.path.exists(filepath):
        return data
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            # 根据文件不同，提取对应的 UTR 字段
            if "ground_truth_utr" in item:
                data[item["id"]] = item["ground_truth_utr"]
            elif "utr" in item:
                utr_obj = item["utr"]
                if "metadata" in utr_obj:
                    data[item["id"]] = utr_obj["metadata"]
                else:
                    data[item["id"]] = utr_obj
    return data

def calc_purity(utr):
    actions = utr.get("core_actions", [])
    if not actions:
        return 1.0
        
    ctrl_count = 0
    for act in actions:
        text_to_check = (act.get("action_name", "") + " " + act.get("description", "")).lower()
        if any(keyword in text_to_check for keyword in CTRL_BLACKLIST):
            ctrl_count += 1
            
    return 1.0 - (ctrl_count / len(actions))

def calc_dependency(gt_utr, pred_utr):
    edges = pred_utr.get("implicit_dependencies", [])
    
    # 如果没有生成任何依赖边
    if not edges:
        gt_edges = gt_utr.get("implicit_dependencies", [])
        if gt_edges:
            return 0.0 # GT中有依赖，但Pred没有，严重缺失
        else:
            return 1.0 # 确实不需要边
    
    # 1. 基础检查：是否为DAG
    G = nx.DiGraph()
    for edge in edges:
        G.add_edge(edge["from"], edge["to"])
        
    try:
        # 检测是否有环
        cycles = list(nx.simple_cycles(G))
        is_dag = 0 if cycles else 1
    except Exception:
        is_dag = 0
        
    if is_dag == 0:
        return 0.0 # 有环直接不合格
        
    # 2. 语义合理性检查 (LLM-as-a-Judge)
    task_goal = pred_utr.get("task_goal", "")
    pred_actions = [f"ID:{a.get('action_id')} Name:{a.get('action_name')}" for a in pred_utr.get("core_actions", [])]
    
    prompt = f"""
    请评估以下工作流动作之间的依赖关系是否合理。
    
    任务目标: {task_goal}
    包含的动作: {pred_actions}
    生成的依赖关系: {edges} (from表示前置节点, to表示后置节点)
    
    评估规则：
    请逐条评估生成的依赖关系边。对于每一条边，判断它是否符合时序逻辑、数据流转是否必要。
    如果不合理（例如时序颠倒，或者毫无必要的强制依赖），记为无效边。
    
    此外，整体评估是否存在【严重缺失】。如果存在必须的依赖关系却没有连线，请在惩罚系数上进行扣分。
    
    请严格按照以下 JSON 格式输出结果：
    {{
        "total_edges": <int, 传入的总边数>,
        "valid_edges": <int, 评估为合理的边数>,
        "missing_penalty": <float, 0.0到1.0之间的系数，1.0表示无缺失不扣分，0.5表示缺失严重>,
        "reasoning": "扣分/得分理由..."
    }}
    """
    
    semantic_score = 1.0
    try:
        result = llm_client.chat_json("你是一个严苛的工作流架构师。", prompt)
        print(f"  [Debug] Dependency LLM Judge result: {result}")
        
        total_edges = max(int(result.get("total_edges", len(edges))), 1)
        valid_edges = int(result.get("valid_edges", total_edges))
        missing_penalty = float(result.get("missing_penalty", 1.0))
        
        semantic_score = (valid_edges / total_edges) * missing_penalty
        
    except Exception as e:
        print(f"  [Warning] Dependency LLM Judge failed, defaulting to 1.0: {e}")
    
    return is_dag * semantic_score

def calc_completeness(gt_utr, pred_utr):
    # 提取变量
    gt_vars = [v.get("name", "") for v in gt_utr.get("core_variables", []) if v.get("name")]
    pred_vars = [v.get("name", "") for v in pred_utr.get("core_variables", []) if v.get("name")]
    
    # 提取动作
    gt_actions = [f"{a.get('action_name', '')}: {a.get('description', '')}" for a in gt_utr.get("core_actions", [])]
    pred_actions = [f"{a.get('action_name', '')}: {a.get('description', '')}" for a in pred_utr.get("core_actions", [])]
    
    # 如果真值既没有变量也没有动作，算作 100% 召回
    if not gt_vars and not gt_actions:
        return 1.0 

    prompt = f"""
    请评估预测结果与标准答案（Ground Truth）的语义匹配度。
    
    【变量匹配】
    标准变量 (GT): {gt_vars}
    预测变量 (Pred): {pred_vars}
    
    【动作匹配】
    标准动作 (GT): {gt_actions}
    预测动作 (Pred): {pred_actions}
    
    判断规则：
    1. 你需要评估「匹配质量」。即使字面不完全一致，只要在工作流上下文中可能指代同一个实体，即可算作匹配。
    2. 采用梯度打分机制，不要只给 0 或 1。
    3. 召回得分 (Recall Score): 对于每个 GT 元素，如果 Pred 中有完全一致的对应记 1.0，如果有部分覆盖/相关记 0.5，完全没有记 0.0。将所有 GT 元素的得分相加，得到 recall_score。
    4. 精确得分 (Precision Score): 对于每个 Pred 元素，如果它在 GT 中有对应或者是完成任务必须的合理步骤，记 1.0；如果是无中生有的幻觉动作/冗余变量，记 0.0；介于两者之间记 0.5。将所有 Pred 元素的得分相加，得到 precision_score。
    
    请严格按照以下 JSON 格式输出：
    {{
        "vars_recall_score": <float, GT变量中被匹配的得分总和>,
        "vars_precision_score": <float, Pred变量中合理/匹配的得分总和>,
        "actions_recall_score": <float, GT动作中被匹配的得分总和>,
        "actions_precision_score": <float, Pred动作中合理/匹配的得分总和>,
        "reasoning": "<简短的分析理由>"
    }}
    """
    
    vars_recall_score = 0.0
    vars_precision_score = 0.0
    actions_recall_score = 0.0
    actions_precision_score = 0.0
    
    try:
        result = llm_client.chat_json("你是一个极其严苛的评估专家。", prompt)
        print(f"  [Debug] Completeness LLM Judge result: {result}")
        
        vars_recall_score = float(result.get("vars_recall_score", 0.0))
        vars_precision_score = float(result.get("vars_precision_score", 0.0))
        actions_recall_score = float(result.get("actions_recall_score", 0.0))
        actions_precision_score = float(result.get("actions_precision_score", 0.0))
        
    except Exception as e:
        print(f"  [Warning] Completeness LLM Judge failed, using fallback: {e}")
        # 退回原有的模糊匹配逻辑作为保底
        for gv in gt_vars:
            if any(gv.lower() in pv.lower() or pv.lower() in gv.lower() for pv in pred_vars):
                vars_recall_score += 1.0
        for pv in pred_vars:
            if any(pv.lower() in gv.lower() or gv.lower() in pv.lower() for gv in gt_vars):
                vars_precision_score += 1.0
        for ga in gt_actions:
            if any(ga.split(':')[0].lower() in pa.lower() for pa in pred_actions):
                actions_recall_score += 1.0
        for pa in pred_actions:
            if any(pa.split(':')[0].lower() in ga.lower() for ga in gt_actions):
                actions_precision_score += 1.0

    # 计算变量的 F1
    var_recall = min(vars_recall_score / len(gt_vars), 1.0) if gt_vars else 1.0
    var_precision = min(vars_precision_score / len(pred_vars), 1.0) if pred_vars else 1.0
    var_f1 = 2 * var_recall * var_precision / (var_recall + var_precision) if (var_recall + var_precision) > 0 else 0.0

    # 计算动作的 F1
    action_recall = min(actions_recall_score / len(gt_actions), 1.0) if gt_actions else 1.0
    action_precision = min(actions_precision_score / len(pred_actions), 1.0) if pred_actions else 1.0
    action_f1 = 2 * action_recall * action_precision / (action_recall + action_precision) if (action_recall + action_precision) > 0 else 0.0
    
    # 动作更重要，给予0.6权重；变量给予0.4权重
    return 0.4 * var_f1 + 0.6 * action_f1

def calc_schema(pred_utr):
    try:
        # 使用 Pydantic 进行校验
        UTRMetadata(**pred_utr)
        return 1.0
    except Exception as e:
        return 0.0

def main():
    print("Loading datasets...")
    gt_data = load_jsonl(GT_FILE)
    pred_data = load_jsonl(PRED_FILE)
    
    common_ids = set(gt_data.keys()).intersection(set(pred_data.keys()))
    if not common_ids:
        print("No matching IDs found between GT and Predictions.")
        return
        
    print(f"Found {len(common_ids)} overlapping samples. Starting evaluation...\n")
    
    scores = {
        "M_pur": [],
        "M_dep": [],
        "M_comp": [],
        "M_sch": [],
        "Overall": []
    }
    
    for uid in common_ids:
        gt_utr = gt_data[uid]
        pred_utr = pred_data[uid]
        
        print(f"Evaluating {uid}...")
        
        m_pur = calc_purity(pred_utr)
        m_dep = calc_dependency(gt_utr, pred_utr)
        m_comp = calc_completeness(gt_utr, pred_utr)
        m_sch = calc_schema(pred_utr)
        
        # 调整权重：M_comp(0.4), M_dep(0.3), M_pur(0.2), M_sch(0.1)
        overall = 0.2 * m_pur + 0.3 * m_dep + 0.4 * m_comp + 0.1 * m_sch
        
        scores["M_pur"].append(m_pur)
        scores["M_dep"].append(m_dep)
        scores["M_comp"].append(m_comp)
        scores["M_sch"].append(m_sch)
        scores["Overall"].append(overall)
        
    # 汇总平均分
    avg_pur = sum(scores["M_pur"]) / len(common_ids)
    avg_dep = sum(scores["M_dep"]) / len(common_ids)
    avg_comp = sum(scores["M_comp"]) / len(common_ids)
    avg_sch = sum(scores["M_sch"]) / len(common_ids)
    avg_overall = sum(scores["Overall"]) / len(common_ids)
    
    print("=====================================")
    print("       UTR 量化评估结果 (Averages)   ")
    print("=====================================")
    print(f"测试样本数: {len(common_ids)}")
    print(f"M_pur  (元数据纯度):     {avg_pur:.4f}")
    print(f"M_dep  (依赖关系合理性): {avg_dep:.4f}")
    print(f"M_comp (实体提取完备性): {avg_comp:.4f}")
    print(f"M_sch  (Schema 依从性):  {avg_sch:.4f}")
    print("-------------------------------------")
    print(f"Overall (综合得分):      {avg_overall:.4f}")
    print("=====================================")

if __name__ == "__main__":
    main()
