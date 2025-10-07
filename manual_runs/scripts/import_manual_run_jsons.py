"""Import Manual Run JSONs Script.

Script to process and import benchmark results from guidellm JSON files
into the performance dashboard data format.
"""

import argparse
import json
import os

import pandas as pd


def process_benchmark_section(
    benchmark_run,
    accelerator,
    model_name,
    version,
    tp_size,
    runtime_args,
    benchmark_index,
):
    """Process a single benchmark section and extract performance metrics.

    Args:
        benchmark_run: Benchmark run data from JSON.
        accelerator: Accelerator type (e.g., H200, MI300X).
        model_name: Name of the AI model.
        version: Version of the inference server.
        tp_size: Tensor parallelism size.
        runtime_args: Runtime configuration arguments.
        benchmark_index: Index of the benchmark run.

    Returns:
        dict: Processed benchmark metrics.
    """
    full_model_name = f"{accelerator}-{model_name}-{tp_size}"

    profile_args = benchmark_run.get("args", {}).get("profile", {})
    uuid = benchmark_run.get("run_id")

    request_loader = benchmark_run.get("request_loader", {})
    request_data_str = request_loader.get("data", "{}")

    try:
        request_config = json.loads(request_data_str)
        config_prompt_tokens = request_config.get("prompt_tokens", 0)
        config_output_tokens = request_config.get("output_tokens", 0)
    except (json.JSONDecodeError, KeyError):
        config_prompt_tokens = 0
        config_output_tokens = 0

    streams = profile_args.get("streams", [])
    measured_rates = profile_args.get("measured_rates", [])
    measured_concurrencies = profile_args.get("measured_concurrencies", [])

    if (
        benchmark_index < len(streams)
        and benchmark_index < len(measured_rates)
        and benchmark_index < len(measured_concurrencies)
    ):
        intended_concurrency = streams[benchmark_index]
        measured_rps = measured_rates[benchmark_index]
        measured_concurrency = measured_concurrencies[benchmark_index]
    else:
        intended_concurrency = streams[0] if streams else None
        measured_rps = measured_rates[0] if measured_rates else None
        measured_concurrency = (
            measured_concurrencies[0] if measured_concurrencies else None
        )

    run_stats = benchmark_run.get("run_stats", {})
    requests_made = run_stats.get("requests_made", {})
    successful_reqs = requests_made.get("successful", 0)  # nosec B113
    errored_reqs = requests_made.get("errored", 0)  # nosec B113

    metrics = benchmark_run.get("metrics", {})

    output_tps_metrics = metrics.get("output_tokens_per_second", {}).get(
        "successful", {}
    )
    output_tok_per_sec = output_tps_metrics.get("mean", 0)

    total_tps_metrics = metrics.get("tokens_per_second", {}).get("successful", {})
    total_tok_per_sec = total_tps_metrics.get("mean", 0)

    prompt_tok_metrics = metrics.get("prompt_token_count", {}).get("successful", {})
    output_tok_metrics = metrics.get("output_token_count", {}).get("successful", {})

    ttft_metrics = metrics.get("time_to_first_token_ms", {}).get("successful", {})
    tpot_metrics = metrics.get("time_per_output_token_ms", {}).get("successful", {})
    itl_metrics = metrics.get("inter_token_latency_ms", {}).get("successful", {})
    request_latency_metrics = metrics.get("request_latency", {}).get("successful", {})

    row = {
        "run": full_model_name,
        "accelerator": accelerator,
        "model": model_name,
        "version": version,
        "prompt toks": config_prompt_tokens,
        "output toks": config_output_tokens,
        "TP": tp_size,
        "measured concurrency": measured_concurrency,
        "intended concurrency": intended_concurrency,
        "measured rps": measured_rps,
        "output_tok/sec": output_tok_per_sec,
        "total_tok/sec": total_tok_per_sec,
        "prompt_token_count_mean": prompt_tok_metrics.get("mean"),
        "prompt_token_count_p99": prompt_tok_metrics.get("percentiles", {}).get("p99"),
        "output_token_count_mean": output_tok_metrics.get("mean"),
        "output_token_count_p99": output_tok_metrics.get("percentiles", {}).get("p99"),
        "ttft_median": ttft_metrics.get("median"),
        "ttft_p95": ttft_metrics.get("percentiles", {}).get("p95"),
        "ttft_p1": ttft_metrics.get("percentiles", {}).get("p01"),
        "ttft_p999": ttft_metrics.get("percentiles", {}).get("p999"),
        "tpot_median": tpot_metrics.get("median"),
        "tpot_p95": tpot_metrics.get("percentiles", {}).get("p95"),
        "tpot_p99": tpot_metrics.get("percentiles", {}).get("p99"),
        "tpot_p999": tpot_metrics.get("percentiles", {}).get("p999"),
        "tpot_p1": tpot_metrics.get("percentiles", {}).get("p01"),
        "itl_median": itl_metrics.get("median"),
        "itl_p95": itl_metrics.get("percentiles", {}).get("p95"),
        "itl_p999": itl_metrics.get("percentiles", {}).get("p999"),
        "itl_p1": itl_metrics.get("percentiles", {}).get("p01"),
        "request_latency_median": request_latency_metrics.get("median"),
        "request_latency_min": request_latency_metrics.get("min"),
        "request_latency_max": request_latency_metrics.get("max"),
        "successful_requests": successful_reqs,
        "errored_requests": errored_reqs,
        "uuid": uuid,
        "ttft_mean": ttft_metrics.get("mean"),
        "ttft_p99": ttft_metrics.get("percentiles", {}).get("p99"),
        "itl_mean": itl_metrics.get("mean"),
        "itl_p99": itl_metrics.get("percentiles", {}).get("p99"),
        "runtime_args": runtime_args,
    }

    return row


def parse_guidellm_json(
    json_path, accelerator, model_name, version, tp_size, runtime_args
):
    """Parse GuideLL JSON benchmark results.

    Args:
        json_path: Path to the JSON file.
        accelerator: Accelerator type.
        model_name: Name of the AI model.
        version: Version of the inference server.
        tp_size: Tensor parallelism size.
        runtime_args: Runtime configuration arguments.

    Returns:
        list: List of processed benchmark results.
    """
    try:
        with open(json_path) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}")
        return None

    all_run_data = []

    if not data.get("benchmarks"):
        print("Error: JSON file does not contain a 'benchmarks' key.")
        return None

    benchmarks = data["benchmarks"]

    if len(benchmarks) > 1:
        print(f"Processing {len(benchmarks)} separate benchmark sections...")
    else:
        print("Processing single benchmark...")

    for i, benchmark_run in enumerate(benchmarks):
        row_data = process_benchmark_section(
            benchmark_run, accelerator, model_name, version, tp_size, runtime_args, i
        )
        if row_data:
            all_run_data.append(row_data)

    if all_run_data:
        return pd.DataFrame(all_run_data)
    else:
        print("No valid data extracted from benchmark sections.")
        return None


def main():
    """Main function to process benchmark JSON files.

    Processes command line arguments and imports benchmark results
    from JSON files into the consolidated CSV format.
    """
    parser = argparse.ArgumentParser(
        description="Import guidellm JSON results into the consolidated benchmark CSV.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "json_file", help="Path to the guidellm JSON output file to import"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name (e.g., 'meta-llama/Llama-2-7b-chat-hf')",
    )
    parser.add_argument(
        "--version",
        required=True,
        help="Version/framework identifier (e.g., 'vLLM-0.6.1')",
    )
    parser.add_argument("--tp", type=int, required=True, help="Tensor parallelism size")
    parser.add_argument(
        "--accelerator", required=True, help="Accelerator type (e.g., 'H100', 'A100')"
    )
    parser.add_argument(
        "--runtime-args",
        required=True,
        help="Runtime arguments used for the inference server (e.g., 'tensor-parallel-size: 8; max-model-len: 8192; trust-remote-code: True')",
    )
    parser.add_argument(
        "--csv-file",
        default="new_benchmarks.csv",
        help="Path to the consolidated CSV file to append data to.",
    )
    args = parser.parse_args()

    print(f"Processing {args.json_file}...")

    new_data_df = parse_guidellm_json(
        args.json_file,
        args.accelerator,
        args.model,
        args.version,
        args.tp,
        args.runtime_args,
    )

    if new_data_df is not None and not new_data_df.empty:
        if os.path.exists(args.csv_file):
            print(f"Appending {len(new_data_df)} new rows to {args.csv_file}...")
            existing_df = pd.read_csv(args.csv_file)
            combined_df = pd.concat([existing_df, new_data_df], ignore_index=True)
        else:
            print(
                f"Creating new CSV file at {args.csv_file} with {len(new_data_df)} rows..."
            )
            combined_df = new_data_df

        fieldnames = [
            "run",
            "accelerator",
            "model",
            "version",
            "prompt toks",
            "output toks",
            "TP",
            "measured concurrency",
            "intended concurrency",
            "measured rps",
            "output_tok/sec",
            "total_tok/sec",
            "prompt_token_count_mean",
            "prompt_token_count_p99",
            "output_token_count_mean",
            "output_token_count_p99",
            "ttft_median",
            "ttft_p95",
            "ttft_p1",
            "ttft_p999",
            "tpot_median",
            "tpot_p95",
            "tpot_p99",
            "tpot_p999",
            "tpot_p1",
            "itl_median",
            "itl_p95",
            "itl_p999",
            "itl_p1",
            "request_latency_median",
            "request_latency_min",
            "request_latency_max",
            "successful_requests",
            "errored_requests",
            "uuid",
            "ttft_mean",
            "ttft_p99",
            "itl_mean",
            "itl_p99",
            "runtime_args",
        ]

        for col in fieldnames:
            if col not in combined_df.columns:
                combined_df[col] = None

        combined_df = combined_df[fieldnames]
        combined_df.to_csv(args.csv_file, index=False)
        print("Successfully updated the CSV file.")
    else:
        print(
            "No valid benchmark data was loaded. Exiting without creating a CSV file."
        )


if __name__ == "__main__":
    main()
