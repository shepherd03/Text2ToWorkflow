import argparse
import json
import sys

from src.workflow_pipeline import WorkflowBuildPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="自然语言任务到 UTR/Skeleton/Dify DSL 的单条链路入口")
    parser.add_argument(
        "--text",
        required=True,
        help="自然语言任务描述",
    )
    parser.add_argument(
        "--stage",
        choices=["utr", "skeleton", "dsl"],
        default="utr",
        help="输出到哪个阶段。默认仅输出 UTR；使用 dsl 可跑完整链路。",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="是否格式化输出",
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    pipeline = WorkflowBuildPipeline()
    output = pipeline.run(args.text, stage=args.stage)
    payload = output.model_dump()
    if args.pretty:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
