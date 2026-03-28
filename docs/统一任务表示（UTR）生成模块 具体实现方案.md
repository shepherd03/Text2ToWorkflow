### 1. 模块定位

整个工作流生成系统的**入口模块**，承接自然语言输入，输出标准化 UTR 结构，为后续骨架规划、节点填充提供唯一语义数据源，是全流程正确性的基础。

### 2. 核心目标

+ 精准解析自然语言中的**动作序列、资源对象、控制意图、上下文变量**
+ 输出严格符合 JSON Schema 的结构化 UTR，无格式错误
+ 支持复杂句式（多条件、多层循环、并行任务）的解析
+ 具备**校验与纠错能力**，降低下游模块错误传导

## 二、UTR 标准化结构定义（JSON Schema）

```plain
{
  "UTR": {
    "actions": [
      {
        "action_id": "string",
        "action_name": "string",
        "description": "string",
        "order": "number"
      }
    ],
    "resources": [
      {
        "resource_id": "string",
        "name": "string",
        "type": "enum[data, file, service, target, variable]",
        "description": "string"
      }
    ],
    "control_intents": [
      {
        "intent_id": "string",
        "type": "enum[sequential, conditional, parallel, iteration]",
        "condition": "string",
        "target_actions": ["string"],
        "loop_target": "string",
        "loop_condition": "string"
      }
    ],
    "variables": [
      {
        "var_id": "string",
        "name": "string",
        "type": "enum[string, number, boolean, list, object]",
        "value": "any",
        "source": "string"
      }
    ]
  }
}
```

## 三、整体实现流程（四阶段 pipeline）

### 阶段 1：输入预处理（清洗 + 增强）

1. **文本清洗**
   - 去除冗余标点、换行、空格
   - 统一口语化表达（例："弄一下"→"处理"，"发一下"→"发送"）
   - 拆分长句（按 "然后、同时、如果、对所有" 等关键词分句）
2. **语义增强**
   - 补全省略主语 / 宾语（例："下载并清洗"→"下载数据并清洗数据"）
   - 标准化动作表述（统一动词：下载 / 获取 / 拉取→download）

### 阶段 2：四大核心子模块并行解析

#### 子模块 1：任务动作序列解析

1. **LLM 动作抽取**
   - Prompt 指令：提取所有核心动作，标注时序顺序，输出标准化动词
   - 示例输出：`[{"action_id":"act_1","action_name":"download_data","order":1},...]`
2. **动作去重与排序**
   - 按 `order`字段排序，确保时序正确
   - 合并重复动作（例：连续两次 "清洗"→合并为一个）

#### 子模块 2：资源对象抽取

1. **规则 + LLM 双抽取**
   - 规则：匹配关键词（数据、文件、API、邮箱、链接）
   - LLM：补充规则未覆盖的隐性资源（例："销售报表"→资源类型：file）
2. **资源分类**
   - 固定分类：data、file、service、target、variable
   - 自动生成 `resource_id`（res_序号）

#### 子模块 3：控制结构意图识别

1. **关键词触发识别**
   - 条件：如果、若、否则、失败则、成功则
   - 并行：同时、并行、一起、分别
   - 迭代：对所有、循环、遍历、每一个
2. **LLM 逻辑绑定**
   - 识别控制结构作用的目标动作
   - 提取条件 / 循环表达式
   - 示例：`{"type":"conditional","condition":"下载失败","target_actions":["act_4"]}`

#### 子模块 4：上下文变量提取

1. **常量提取**
   - 邮箱、路径、数字、文件名、URL 等具体值
2. **变量类型标注**
   - 自动判断类型：string/number/boolean/list
3. **变量来源标记**
   - 来源：用户输入、资源引用、动作输出

### 阶段 3：UTR 整合与 ID 关联

1. 统一分配唯一 ID（act_xx /res_xx/intent_xx /var_xx）
2. 建立关联关系：
   - 动作引用资源 → 资源 ID
   - 控制意图绑定动作 → 动作 ID
   - 变量关联动作 / 资源 → 对应 ID
3. 填充完整 UTR 结构，确保无缺失字段

### 阶段 4：三级校验（核心保障）

1. **Schema 校验**：严格校验 JSON 格式与字段类型
2. **逻辑校验**
   - 动作 `order`连续无重复
   - 控制意图 `target_actions`存在
   - 变量引用有效
3. **完整性校验**：四大子模块均非空，无关键信息缺失

## 四、Prompt 工程

## 五、技术选型与工程实现

### 1. 技术栈

+ LLM：GPT-4o / 通义千问 / 本地微调模型（支持结构化输出）
+ 校验库：Pydantic / JSON Schema
+ 文本处理：正则表达式 + LLM 语义增强
+ 工程框架：Python + FastAPI（接口化）

### 2. 代码架构（模块化）

```plain
utr_generator/
├── preprocessor.py    # 输入预处理
├── action_extractor.py # 动作解析
├── resource_extractor.py # 资源抽取
├── control_intent_extractor.py # 控制意图识别
├── variable_extractor.py # 变量提取
├── utr_combiner.py # 整合与ID关联
├── validator.py # 三级校验
└── main.py # 入口调用
```

### 3. 调用示例（输入→输出）

**输入**：下载销售数据，如果下载失败则发送告警；同时清洗数据并分析，最后生成报告发送到 test@example.com

**输出 UTR（精简）**

```plain
{
  "actions": [
    {"action_id":"act_1","action_name":"download_data","order":1},
    {"action_id":"act_2","action_name":"send_alert","order":2},
    {"action_id":"act_3","action_name":"clean_data","order":3},
    {"action_id":"act_4","action_name":"analyze_data","order":4},
    {"action_id":"act_5","action_name":"generate_report","order":5},
    {"action_id":"act_6","action_name":"send_email","order":6}
  ],
  "control_intents": [
    {"intent_id":"intent_1","type":"conditional","condition":"下载失败","target_actions":["act_2"]},
    {"intent_id":"intent_2","type":"parallel","target_actions":["act_3","act_4"]}
  ],
  "variables": [
    {"var_id":"var_1","name":"recipient","type":"string","value":"test@example.com"}
  ]
}
```

## 六、鲁棒性优化方案

1. **低置信度兜底**：LLM 置信度 < 0.7 时，标记待人工确认
2. **多模型投票**：2 个模型解析结果不一致时，触发校验告警
3. **错误日志**：记录解析失败案例，用于迭代优化 Prompt
4. **版本管理**：UTR Schema 版本化，兼容后续迭代
