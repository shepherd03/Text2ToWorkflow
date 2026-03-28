# Skeleton Planner 批量生成与评估报告

**测试时间**：2026-03-22
**测试对象**：`utr_generator.skeleton_planner.SkeletonPlanner`
**数据来源**：`generated_data/utr_generation/utrs.jsonl` (包含完整流程生成的真实 UTR)
**保存路径**：`generated_data/skeleton_planning/`

---

## 一、 数据持久化与复用机制的建立
根据最新的项目规范，所有模块生成的数据（如生成的 UTR、骨架树等）已统一保存在项目根目录的 `generated_data/` 文件夹中。
*   **UTR 生成结果**：保存在 `generated_data/utr_generation/utrs.jsonl`，用于持久化 `UTRPipeline` 跑出的数据，避免重复的大模型 API 请求。
*   **Skeleton 生成结果**：保存在 `generated_data/skeleton_planning/skeletons.jsonl`，作为本模块全量测试的结果。

## 二、 运行结果概述
使用全量（真实LLM生成的）数据集进行骨架生成，运行结果如下：
*   **总测试样本**：2个（受限于测试成本和API，目前跑通了2个真实长样本）
*   **成功数**：2
*   **错误数**：0
*   **执行表现**：没有触发异常崩溃，`SkeletonPlanner` 稳定输出了树状 JSON 结构。

## 三、 结构质量评估与案例分析

通过观察生成的 `skeletons.jsonl`，我们提取了真实案例进行分析：

### 案例 1：代码翻译 (wf_0001)
*   **指令**：我想把一种编程语言的代码翻译成另一种语言。
*   **生成的骨架**：
    ```json
    Sequential:
      1. Start
      2. variable_assignment
      3. code_execution
      4. Loop (target: code_units) ->
           Sequential:
             1. llm_generation
      5. End
    ```
*   **评估**：✅ **表现完美**。大模型正确提取了循环意图，骨架规划器成功将其映射为 `Loop` 块，并将 `llm_generation` 包裹在循环体内。起止节点也被正确补全。

### 案例 2：运营一条龙 (wf_0002)
*   **指令**：我想做个能自动生成小红书、微博这些平台的封面图和文案的工具。
*   **生成的骨架**：
    ```json
    Sequential:
      1. Start
      2. variable_assignment
      3. Loop (target: 平台列表) ->
           Sequential:
             1. llm_generation
      4. code_execution
      5. image_generation
      6. Conditional (condition: condition_branch) ->
           True Branch:  []
           False Branch: []
      7. End
    ```
*   **评估**：⚠️ **存在瑕疵**。
    *   **亮点**：循环块生成成功（对平台列表进行 LLM 生成）。
    *   **问题 (Empty Branches)**：大模型提取出了 `condition_branch` 动作和 `Conditional` 意图，但它没能识别出哪些动作应该放在 True/False 分支里，导致最终生成的条件块的分支是空的。

---

## 四、 发现的问题与优化策略 (Issues & Strategies)

基于真实数据的全量测试，我们发现了以下潜在需要优化的边缘情况：

### 1. 悬空条件分支 (Empty Conditional Branches)
*   **现象**：如案例 2 所示，生成了一个带有 `condition_slot` 但 `true_branch` 和 `false_branch` 均为空的 `ConditionalBlock`。
*   **原因分析**：大模型在生成时，仅输出了“条件判断”这一动作，但没有为该意图绑定足够多的后续动作（`target_actions` 数量不足，或提取 ID 失败）。
*   **优化策略 (Pruning / Dummy Node)**：
    *   在构建结束后增加一个**后处理剪枝步骤 (Post-processing Pruning)**。
    *   规则：如果一个 `ConditionalBlock` 的 True 和 False 分支都为空，这个条件判断就失去了意义。此时应该将该条件块**降级（退化）**为普通的顺序节点，或者直接删除，以避免在 Dify 中生成无效的 If-Else 节点。

### 2. 意图范围包络的准确性
*   **现象**：当前的启发式对齐是按顺序将匹配到的 ID 打包。但在真实的复杂工作流中，`code_execution` 和 `image_generation`（案例 2 中的步骤 4,5）可能实际上应该是被包含在循环体内的。
*   **优化策略 (Dependency-Aware Alignment)**：
    *   单纯依靠 LLM 提取的 `target_actions` 容易漏掉节点。未来需要引入数据依赖流（Data Flow）。如果 `image_generation` 依赖了循环体内部产生的变量，那么在骨架规划时，应强制将其吸纳入 `LoopBlock` 中。

## 五、 下一步行动计划
基于目前的表现，`SkeletonPlanner` 的主体架构非常稳健。下一个开发阶段应该：
1.  **实现空分支剪枝**：在 `SkeletonPlanner.plan()` 的最后，增加对树中空分支的检查和清理。
2.  **进入节点填充 (Node Filling) 开发**：当前骨架里的 `ActionSlot` 已经非常清晰，下一步是编写 `NodeFiller`，将 UTR 中的 `args` 转化为 Dify 引擎实际可用的格式。