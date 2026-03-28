# UTR Evaluation Metrics Documentation

## Overview
The UTR Evaluation system employs a **dual-track evaluation mechanism** combining **Rule-based Scoring** (objective) and **LLM-as-a-Judge** (subjective/semantic). This document explains the specific metrics, their weights, and calculation logic.

## 1. Rule-based Evaluation (`UTREvaluator`)

This method compares the Generated UTR against the Ground Truth (converted from Dify DSL) using deterministic algorithms.

### 1.1 Metrics Breakdown

| Metric | Weight (Standard) | Weight (No Params) | Description |
| :--- | :--- | :--- | :--- |
| **Action Accuracy** | **40%** | **~57%** | Uses fuzzy matching and semantic grouping (e.g., `translation` ≈ `llm_generation`) to check if the correct tools/steps are selected. |
| **Parameter Completeness** | **30%** | **0%** | Checks if key arguments (e.g., `prompt_template`, `variables`) are present. Uses "loose matching" (value coverage) to handle structural differences. |
| **Logic Coherence** | **20%** | **~29%** | Evaluates the correctness of `control_intents` (Sequential, Conditional, Iteration). Checks if condition strings roughly match. |
| **Schema Validity** | **10%** | **~14%** | Checks compliance with the UTR Pydantic schema (valid JSON, required fields present). |

### 1.2 Calculation Formulas

*   **Standard Total Score**:
    ```python
    Total = (Action * 0.4) + (Parameter * 0.3) + (Logic * 0.2) + (Schema * 0.1)
    ```

*   **No-Params Total Score** (New):
    *   *Purpose*: To evaluate the quality of the workflow "skeleton" (structure and logic) without being penalized for missing configuration details (which are often specific to Dify's internal implementation).
    *   *Normalization*: Weights are re-normalized based on the remaining 70% mass (0.4 + 0.2 + 0.1 = 0.7).
    ```python
    Total_NoParams = (Action * 0.4 + Logic * 0.2 + Schema * 0.1) / 0.7
    ```

---

## 2. LLM-as-a-Judge Evaluation (`LLMUTREvaluator`)

This method uses a large language model (DeepSeek-V3) to review the Instruction, Ground Truth, and Generated UTR, providing a semantic score.

### 2.1 Metrics Breakdown

| Metric | Weight | Description |
| :--- | :--- | :--- |
| **Intent Match** | **40%** | Does the generated workflow achieve the user's goal? (e.g., "Translate code" vs "Analyze code"). |
| **Parameter Accuracy** | **30%** | Are the parameters meaningful and correct in context? (e.g., Is the prompt template relevant?). |
| **Logic/Flow** | **20%** | Is the execution order logical? Are necessary checks (e.g., `if success`) present? |
| **Completeness** | **10%** | Are all necessary steps included? Does it hallucinate unnecessary steps? |

### 2.2 Why use LLM Evaluation?
*   **Semantic Understanding**: It can tell that `web_scraper` is a bad choice for reading a local file, whereas rule-based matching might just see it as a "mismatched string".
*   **Complexity Assessment**: It penalizes "oversimplified" workflows (e.g., 1-step translation) that miss the robustness of production workflows.

---

## 3. Interpretation Guide

*   **High Rule Score, Low LLM Score**: The model generated the correct *names* of actions, but the logic or parameters make no sense semantically.
*   **Low Rule Score, High LLM Score**: The model used different tool names (e.g., `custom_tool` vs `http_request`) but achieved the correct goal semantically.
*   **High No-Param Score, Low Standard Score**: The **Skeleton** (Actions + Logic) is correct, but the **Flesh** (Parameters/Prompts) is weak. *This is the current state of our system.*
