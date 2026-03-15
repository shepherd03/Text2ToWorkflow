import argparse
import json

from utr_generator.pipeline import UTRGenerationPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UTR生成模块")
    parser.add_argument(
        "--text",
        required=True,
        help="自然语言任务描述",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="是否格式化输出",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = UTRGenerationPipeline()
    output = pipeline.run(args.text)
    payload = {
        "UTR": output.utr.model_dump(),
        "validation_report": output.report.model_dump(),
        "meta": output.meta,
    }
    if args.pretty:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
