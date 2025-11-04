#!/usr/bin/env python3
"""Helper script to generate CSV summary files from MLPerf dataset files.

This script processes the original dataset files (pickle, JSON, etc.) and extracts
only the input_length and output_length information into lightweight CSV files.

Setup:
    1. Place original dataset files in: mlperf-data/original/
    2. Original files are NOT version controlled (.gitignore)

Usage:
    Run from the project root directory:

    cd /path/to/performance-dashboard
    python mlperf-data/original/generate_dataset_summaries.py

    Output summaries will be saved to: mlperf-data/summaries/
"""

import csv
import gzip
import json
import os
import pickle  # nosec B403 - Used only for trusted MLCommons official datasets


def process_deepseek_r1():
    """Process deepseek-r1 dataset (pickle format)."""
    print("Processing deepseek-r1...")
    input_path = "mlperf-data/original/mlperf_deepseek_r1_dataset_4388_fp8_eval.pkl"
    output_path = "mlperf-data/summaries/deepseek-r1.csv"

    if not os.path.exists(input_path):
        print(f"  ⚠️  Input file not found: {input_path}")
        return

    try:
        # This requires pandas - user should run in their Streamlit environment
        import pandas as pd

        with open(input_path, "rb") as f:
            data = pickle.load(f)  # nosec B301 - Loading trusted MLCommons dataset

        # Extract token length columns (deepseek uses tok_input_len and tok_ref_output_len)
        if isinstance(data, pd.DataFrame):
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
                print(
                    f"  ⚠️  Expected columns not found. Available: {list(data.columns)}"
                )
                return
        else:
            print(f"  ⚠️  Unexpected data type: {type(data)}")
            return

        # Save to CSV
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        summary.to_csv(output_path, index=False)
        print(f"  ✅ Generated: {output_path} ({len(summary):,} samples)")

    except ImportError:
        print("  ⚠️  Requires pandas. Please run with: python3 -m pip install pandas")
    except Exception as e:
        print(f"  ❌ Error: {e}")


def process_llama31_8b():
    """Process llama3.1-8b-datacenter dataset (JSON format)."""
    print("Processing llama3.1-8b-datacenter...")
    input_path = "mlperf-data/original/cnn_eval.json"
    output_path = "mlperf-data/summaries/llama3-1-8b-datacenter.csv"

    if not os.path.exists(input_path):
        print(f"  ⚠️  Input file not found: {input_path}")
        return

    try:
        with open(input_path) as f:
            json_data = json.load(f)

        # Save to CSV
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["input_length", "output_length"])

            for item in json_data:
                # Input length from tokenized input
                input_length = len(item.get("tok_input", []))

                # Output length: estimate from character count (rough approximation: ~4 chars per token)
                output_text = item.get("output", "")
                output_length = len(output_text) // 4  # Rough approximation

                writer.writerow([input_length, output_length])

        print(f"  ✅ Generated: {output_path} ({len(json_data):,} samples)")

    except Exception as e:
        print(f"  ❌ Error: {e}")


def process_llama2_70b():
    """Process llama2-70b dataset (compressed pickle format)."""
    print("Processing llama2-70b-99...")
    input_path = (
        "mlperf-data/original/open_orca_gpt4_tokenized_llama.sampled_24576.pkl.gz"
    )
    output_path = "mlperf-data/summaries/llama2-70b-99.csv"

    if not os.path.exists(input_path):
        print(f"  ⚠️  Input file not found: {input_path}")
        return

    try:
        # This requires pandas - user should run in their Streamlit environment
        import pandas as pd

        with gzip.open(input_path, "rb") as f:
            data = pickle.load(f)  # nosec B301 - Loading trusted MLCommons dataset

        # Extract token length columns (llama2-70b uses tok_input_length and tok_output_length)
        if isinstance(data, pd.DataFrame):
            if (
                "tok_input_length" in data.columns
                and "tok_output_length" in data.columns
            ):
                summary = pd.DataFrame(
                    {
                        "input_length": data["tok_input_length"],
                        "output_length": data["tok_output_length"],
                    }
                )
            elif "input_length" in data.columns and "output_length" in data.columns:
                summary = data[["input_length", "output_length"]].copy()
            else:
                print(
                    f"  ⚠️  Expected columns not found. Available: {list(data.columns)}"
                )
                return
        else:
            print(f"  ⚠️  Unexpected data type: {type(data)}")
            return

        # Save to CSV
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        summary.to_csv(output_path, index=False)
        print(f"  ✅ Generated: {output_path} ({len(summary):,} samples)")

    except ImportError:
        print("  ⚠️  Requires pandas. Please run with: python3 -m pip install pandas")
    except Exception as e:
        print(f"  ❌ Error: {e}")


def main():
    """Process all datasets and generate CSV summaries."""
    print("=" * 60)
    print("MLPerf Dataset Summary Generator")
    print("=" * 60)
    print()

    # Process each dataset
    process_deepseek_r1()
    process_llama31_8b()
    process_llama2_70b()

    print()
    print("=" * 60)
    print("Done! Summary files saved to mlperf-data/summaries/")
    print("=" * 60)


if __name__ == "__main__":
    main()
