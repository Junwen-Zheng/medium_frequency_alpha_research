from __future__ import annotations

import argparse
from .workflow import ResearchWorkflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Run medium-frequency equity signal research workflow")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", default="config/default.yaml")
    run.add_argument("--synthetic", action="store_true", help="Run no-internet synthetic demo")
    args = parser.parse_args()

    if args.command == "run":
        result = ResearchWorkflow(args.config, synthetic=args.synthetic).run()
        print(f"Report written to {result.report_path}")
        print(result.backtest_metrics)


if __name__ == "__main__":
    main()
