"""Command-line entry point for independent real-world dataset pipelines and paper assets."""

from __future__ import annotations

import argparse
import sys
from typing import Any

from eval_llm import run_llm_benchmark
from make_paper_assets import main as generate_paper_assets
from train_model import train_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute reproducible e-commerce customer loyalty analytics pipeline.")
    parser.add_argument(
        "--dataset",
        choices=["online_retail_ii", "olist", "all"],
        default="all",
        help="Target dataset to process, or 'all' for both Online Retail II and Olist.",
    )
    parser.add_argument(
        "--skip-paper-assets",
        action="store_true",
        help="Skip automatic generation of 300 DPI research figures and paper tables.",
    )
    args = parser.parse_args()

    datasets = ["online_retail_ii", "olist"] if args.dataset == "all" else [args.dataset]
    results: dict[str, Any] = {}

    print(f"[run_all] Executing pipeline for datasets: {datasets}")
    for key in datasets:
        print(f"\n========================================================")
        print(f"  RUNNING PIPELINE: {key}")
        print(f"========================================================")
        res = train_dataset(key)
        results[key] = res
        print(f"[run_all] {key} complete: Selected Algorithm={res['selected_algorithm']} (K={res['selected_k']}), Customers={res['customer_count']:,}")

    if not args.skip_paper_assets:
        print(f"\n========================================================")
        print(f"  GENERATING TRACEABLE RESEARCH TABLES & 300 DPI FIGURES")
        print(f"========================================================")
        try:
            generate_paper_assets()
            print("[run_all] Paper assets generated successfully in results/paper/")
        except Exception as exc:
            print(f"[run_all] Warning: Paper assets generation encountered an issue: {exc}", file=sys.stderr)

    print("\n[run_all] Pipeline execution finished successfully.")


if __name__ == "__main__":
    main()
