from __future__ import annotations

import argparse
from .workflow import ResearchWorkflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Run medium-frequency equity signal research workflow")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", default="config/default.yaml")
    run.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run an offline synthetic smoke test; not research evidence.",
    )
    args = parser.parse_args()

    if args.command == "run":
        result = ResearchWorkflow(args.config, smoke_test=args.smoke_test).run()
        print(f"Data mode: {result.data_mode}")
        print(f"Report written to {result.report_path}")
        print(result.backtest_metrics)


if __name__ == "__main__":
    main()
