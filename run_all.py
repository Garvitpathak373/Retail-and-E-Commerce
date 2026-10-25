"""Command-line entry point for independent real-world dataset pipelines."""

from __future__ import annotations

import argparse

from train_model import train_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["online_retail_ii", "olist", "all"], required=True)
    dataset = parser.parse_args().dataset
    for key in ([dataset] if dataset != "all" else ["online_retail_ii", "olist"]):
        result = train_dataset(key)
        print(f"{key}: selected K={result['selected_k']}, customers={result['customer_count']}")


if __name__ == "__main__":
    main()
