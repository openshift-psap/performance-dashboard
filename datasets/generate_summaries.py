#!/usr/bin/env python3
"""Generate lightweight CSV summaries from RHAIIS dataset files.

This script processes the original dataset files (pickle, parquet, npy) and extracts
only the token length information into lightweight CSV files suitable for the
dashboard's Dataset Representation section.

Setup:
    1. Place original dataset files in their respective subdirectories under datasets/
    2. Original files are NOT version controlled (.gitignore) due to their large size

Usage:
    Run from the project root directory:

    cd /path/to/performance-dashboard
    python datasets/generate_summaries.py

    Output summaries will be saved to: datasets/summaries/
"""

import os
import pickle  # nosec B403 - Used only for trusted benchmark datasets
import sys


def process_deepseek_r1():
    """Process deepseek-r1 dataset (pickle format with tok_input_len and tok_ref_output_len)."""
    print("Processing deepseek-r1...")
    input_path = "datasets/deepseek-r1/mlperf_deepseek_r1_dataset_4388_fp8_eval.pkl"
    output_path = "datasets/summaries/deepseek-r1.csv"

    if not os.path.exists(input_path):
        print(f"  Input file not found: {input_path}")
        return

    try:
        import pandas as pd

        with open(input_path, "rb") as f:
            data = pickle.load(f)  # nosec B301 - Loading trusted benchmark dataset

        if not isinstance(data, pd.DataFrame):
            print(f"  Unexpected data type: {type(data)}")
            return

        if "tok_input_len" in data.columns and "tok_ref_output_len" in data.columns:
            summary = pd.DataFrame(
                {
                    "input_length": data["tok_input_len"],
                    "output_length": data["tok_ref_output_len"],
                }
            )
        elif "input_length" in data.columns and "output_length" in data.columns:
            summary = data[["input_length", "output_length"]].copy()
        else:
            print(f"  Expected columns not found. Available: {list(data.columns)}")
            return

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        summary.to_csv(output_path, index=False)
        print(f"  Generated: {output_path} ({len(summary):,} samples)")
        print(
            f"  Input tokens  - min: {summary['input_length'].min()}, "
            f"max: {summary['input_length'].max()}, "
            f"mean: {summary['input_length'].mean():.1f}"
        )
        print(
            f"  Output tokens - min: {summary['output_length'].min()}, "
            f"max: {summary['output_length'].max()}, "
            f"mean: {summary['output_length'].mean():.1f}"
        )

    except ImportError:
        print("  Requires pandas. Install with: pip install pandas")
    except Exception as e:
        print(f"  Error: {e}")


def process_gpt_oss():
    """Process gpt-oss_data perf evaluation dataset (parquet format).

    Note: This dataset only contains input token lengths (num_tokens).
    Output token lengths are not available in the source data.
    """
    print("Processing gpt-oss (perf eval)...")
    input_path = "datasets/gpt-oss_data/perf/perf_eval_ref.parquet"
    output_path = "datasets/summaries/gpt-oss.csv"

    if not os.path.exists(input_path):
        print(f"  Input file not found: {input_path}")
        return

    try:
        import pandas as pd

        data = pd.read_parquet(input_path)

        if "num_tokens" not in data.columns:
            print(
                f"  Expected 'num_tokens' column not found. Available: {list(data.columns)}"
            )
            return

        summary = pd.DataFrame(
            {
                "input_length": data["num_tokens"],
            }
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        summary.to_csv(output_path, index=False)
        print(f"  Generated: {output_path} ({len(summary):,} samples)")
        print(
            f"  Input tokens - min: {summary['input_length'].min()}, "
            f"max: {summary['input_length'].max()}, "
            f"mean: {summary['input_length'].mean():.1f}"
        )
        print("  Note: Output token lengths not available in source data")

    except ImportError as e:
        print(
            f"  Requires pandas and pyarrow. Install with: pip install pandas pyarrow. Error: {e}"
        )
    except Exception as e:
        print(f"  Error: {e}")


def process_sharegpt():
    """Process ShareGPT Vicuna dataset from HuggingFace.

    Downloads the cleaned/split JSON from anon8231489123/ShareGPT_Vicuna_unfiltered,
    tokenizes each conversation's human turns (input) and gpt turns (output) using
    tiktoken, and writes input_length/output_length to a CSV summary.

    Requires: pip install tiktoken huggingface_hub
    """
    print("Processing ShareGPT Vicuna...")
    output_path = "datasets/summaries/sharegpt-vicuna.csv"

    try:
        import json

        import pandas as pd
        import tiktoken
        from huggingface_hub import hf_hub_download

        print(
            "  Downloading ShareGPT_V3_unfiltered_cleaned_split.json from HuggingFace..."
        )
        json_path = hf_hub_download(
            repo_id="anon8231489123/ShareGPT_Vicuna_unfiltered",
            filename="ShareGPT_V3_unfiltered_cleaned_split.json",
            repo_type="dataset",
        )

        with open(json_path, encoding="utf-8") as f:
            conversations_list = json.load(f)

        enc = tiktoken.get_encoding("cl100k_base")
        input_lengths = []
        output_lengths = []
        skipped = 0

        print(f"  Tokenizing {len(conversations_list):,} conversations...")
        for item in conversations_list:
            conversations = item.get("conversations")
            if not conversations:
                skipped += 1
                continue

            human_text = " ".join(
                turn["value"]
                for turn in conversations
                if turn.get("from") == "human" and turn.get("value")
            )
            gpt_text = " ".join(
                turn["value"]
                for turn in conversations
                if turn.get("from") == "gpt" and turn.get("value")
            )

            if not human_text and not gpt_text:
                skipped += 1
                continue

            input_lengths.append(
                len(enc.encode(human_text, disallowed_special=())) if human_text else 0
            )
            output_lengths.append(
                len(enc.encode(gpt_text, disallowed_special=())) if gpt_text else 0
            )

        summary = pd.DataFrame(
            {
                "input_length": input_lengths,
                "output_length": output_lengths,
            }
        )

        before_count = len(summary)
        for col in ["input_length", "output_length"]:
            q1 = summary[col].quantile(0.25)
            q3 = summary[col].quantile(0.75)
            iqr = q3 - q1
            upper_bound = q3 + 1.5 * iqr
            summary = summary[summary[col] <= upper_bound]
        outliers_removed = before_count - len(summary)
        print(f"  Removed {outliers_removed:,} outlier rows (IQR method)")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        summary.to_csv(output_path, index=False)
        print(
            f"  Generated: {output_path} ({len(summary):,} samples, {skipped} skipped, {outliers_removed} outliers removed)"
        )
        print(
            f"  Input tokens  - min: {summary['input_length'].min()}, "
            f"max: {summary['input_length'].max()}, "
            f"mean: {summary['input_length'].mean():.1f}"
        )
        print(
            f"  Output tokens - min: {summary['output_length'].min()}, "
            f"max: {summary['output_length'].max()}, "
            f"mean: {summary['output_length'].mean():.1f}"
        )

    except ImportError as e:
        print(
            f"  Requires tiktoken and huggingface_hub. Install with: pip install tiktoken huggingface_hub. Error: {e}"
        )
    except Exception as e:
        print(f"  Error: {e}")


def main():
    """Process all datasets and generate CSV summaries."""
    print("=" * 60)
    print("RHAIIS Dataset Summary Generator")
    print("=" * 60)
    print()

    if not os.path.exists("datasets"):
        print("Error: 'datasets' directory not found.")
        print("Please run this script from the project root directory.")
        sys.exit(1)

    process_deepseek_r1()
    print()
    process_gpt_oss()
    print()
    process_sharegpt()

    print()
    print("=" * 60)
    print("Done! Summary files saved to datasets/summaries/")
    print("=" * 60)


if __name__ == "__main__":
    main()
