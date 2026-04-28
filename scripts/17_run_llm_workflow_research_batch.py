from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import load_settings
from src.workflow_pipeline import WorkflowBuildPipeline

DEFAULT_TASKS = [
    "我想读取一篇文章，提取核心观点，生成结构化摘要并保存结果。",
    "帮我批量读取客户反馈，判断情绪倾向，按严重程度分流给不同负责人。",
    "我想输入多个网页链接，抓取内容后逐篇总结，再合并成一份研究简报。",
    "帮我把上传的合同提取关键条款，检查是否缺少金额、期限和责任方。",
    "我想定时调用天气接口，生成出行建议，并在异常天气时发送提醒。",
    "帮我把销售线索先抽取联系人信息，再判断质量，最后生成跟进话术。",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a real-LLM workflow research batch and record auditable artifacts."
    )
    parser.add_argument(
        "--tasks-file",
        default="",
        help="Optional JSONL/TXT file. JSONL can contain instruction/text/task fields.",
    )
    parser.add_argument(
        "--output-dir",
        default="generated_data/llm_workflow_research/latest",
        help="Directory for records.jsonl, errors.jsonl and manifest.json.",
    )
    parser.add_argument(
        "--stage",
        choices=["utr", "skeleton", "dsl"],
        default="dsl",
        help="Pipeline stage to run.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=6,
        help="Maximum tasks to run.",
    )
    return parser.parse_args(argv)


def load_tasks(path: str) -> list[str]:
    if not path:
        return DEFAULT_TASKS
    task_path = Path(path)
    tasks: list[str] = []
    for line in task_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if task_path.suffix.lower() == ".jsonl":
            item = json.loads(line)
            text = item.get("instruction") or item.get("text") or item.get("task")
            if text:
                tasks.append(str(text))
        else:
            tasks.append(line.strip())
    return tasks


def write_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(item, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    settings = load_settings()
    if not settings.llm_enabled:
        raise RuntimeError("DEEPSEEK_API_KEY 未配置，不能运行真实 LLM 研究批处理。")

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    errors_path = output_dir / "errors.jsonl"
    manifest_path = output_dir / "manifest.json"
    for path in [records_path, errors_path, manifest_path]:
        if path.exists():
            path.unlink()

    pipeline = WorkflowBuildPipeline(settings)
    tasks = load_tasks(args.tasks_file)[: args.max_records]
    manifest: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "task_count": len(tasks),
        "settings": {
            "llm_enabled": settings.llm_enabled,
            "deepseek_base_url": settings.deepseek_base_url,
            "deepseek_model": settings.deepseek_model,
            "semantic_backend": settings.semantic_backend,
        },
        "success_count": 0,
        "error_count": 0,
        "llm_call_count": 0,
        "records_path": str(records_path.relative_to(ROOT)),
        "errors_path": str(errors_path.relative_to(ROOT)),
    }

    for index, task_text in enumerate(tasks, start=1):
        record_id = f"llm_research_{index:04d}"
        try:
            output = pipeline.run(task_text, stage=args.stage)
            record = {
                "id": record_id,
                "instruction": task_text,
                "output": output.model_dump(),
            }
            write_jsonl(records_path, record)
            meta = output.utr_output.meta
            manifest["success_count"] += 1
            manifest["llm_call_count"] += int(meta.get("llm_call_count", 0))
        except Exception as exc:
            manifest["error_count"] += 1
            write_jsonl(
                errors_path,
                {
                    "id": record_id,
                    "instruction": task_text,
                    "error": str(exc),
                },
            )

    manifest["used_real_llm"] = manifest["llm_call_count"] > 0
    manifest["records"] = manifest["success_count"] + manifest["error_count"]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
