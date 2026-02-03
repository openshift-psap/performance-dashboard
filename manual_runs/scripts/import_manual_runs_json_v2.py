"""Import Manual Run JSONs Script for guidellm 0.5.x.

Script to process and import benchmark results from guidellm 0.5.x and above JSON files
into the performance dashboard data format.

Updated for guidellm 0.5.x JSON structure changes.
"""

import argparse
import json
import os

import pandas as pd


def process_benchmark_section(
    benchmark,
    accelerator,
    model_name,
    version,
    tp_size,
    runtime_args,
    global_data_config,
    image_tag,
    guidellm_version,
    guidellm_start_time_ms,
    guidellm_end_time_ms,
):
    """Process a single benchmark section and extract performance metrics.

    Args:
        benchmark: Benchmark data from JSON (guidellm 0.5.x format).
        accelerator: Accelerator type (e.g., H200, MI300X).
        model_name: Name of the AI model.
        version: Version of the inference server.
        tp_size: Tensor parallelism size.
        runtime_args: Runtime configuration arguments.
        global_data_config: Global data configuration from top-level args.
        image_tag: Container image tag used for the run.
        guidellm_version: Version of guidellm used to run the benchmark.
        guidellm_start_time_ms: Aggregated start time in milliseconds.
        guidellm_end_time_ms: Aggregated end time in milliseconds.

    Returns:
        dict: Processed benchmark metrics.
    """
    full_model_name = f"{accelerator}-{model_name}-{tp_size}"

    config = benchmark.get("config", {})
    uuid = config.get("run_id")

    # Get strategy info (streams/concurrency)
    strategy = config.get("strategy", {})
    intended_concurrency = strategy.get("streams") or strategy.get("max_concurrency", 0)

    # Parse data config for prompt/output tokens
    # Format can be either JSON or key=value pairs like "prompt_tokens=1000,output_tokens=1000"
    config_prompt_tokens = 0
    config_output_tokens = 0
    try:
        if global_data_config and len(global_data_config) > 0:
            data_str = global_data_config[0]
            # Try JSON format first
            try:
                request_config = json.loads(data_str)
                config_prompt_tokens = request_config.get("prompt_tokens", 0)
                config_output_tokens = request_config.get("output_tokens", 0)
            except json.JSONDecodeError:
                # Try key=value format: "prompt_tokens=1000,output_tokens=1000"
                for item in data_str.split(","):
                    if "=" in item:
                        key, value = item.strip().split("=", 1)
                        if key == "prompt_tokens":
                            config_prompt_tokens = int(value)
                        elif key == "output_tokens":
                            config_output_tokens = int(value)
    except (KeyError, TypeError, ValueError):
        config_prompt_tokens = 0
        config_output_tokens = 0

    # Get request stats from scheduler_metrics
    scheduler_metrics = benchmark.get("scheduler_metrics", {})
    requests_made = scheduler_metrics.get("requests_made", {})
    successful_reqs = requests_made.get("successful", 0)
    errored_reqs = requests_made.get("errored", 0)

    # Get metrics
    metrics = benchmark.get("metrics", {})

    # Output tokens per second
    output_tps_metrics = metrics.get("output_tokens_per_second", {}).get("total", {})
    output_tok_per_sec = output_tps_metrics.get("mean", 0)

    # Total tokens per second
    total_tps_metrics = metrics.get("tokens_per_second", {}).get("total", {})
    total_tok_per_sec = total_tps_metrics.get("mean", 0)

    # Token counts
    prompt_tok_metrics = metrics.get("prompt_token_count", {}).get("successful", {})
    output_tok_metrics = metrics.get("output_token_count", {}).get("successful", {})

    # Latency metrics
    ttft_metrics = metrics.get("time_to_first_token_ms", {}).get("successful", {})
    tpot_metrics = metrics.get("time_per_output_token_ms", {}).get("successful", {})
    itl_metrics = metrics.get("inter_token_latency_ms", {}).get("successful", {})
    request_latency_metrics = metrics.get("request_latency", {}).get("successful", {})

    # Request concurrency
    request_concurrency = metrics.get("request_concurrency", {}).get("successful", {})
    measured_concurrency = request_concurrency.get("mean", intended_concurrency)

    # Requests per second
    rps_metrics = metrics.get("requests_per_second", {}).get("successful", {})
    measured_rps = rps_metrics.get("mean", 0)

    # Helper to get percentiles (0.5.x uses p01, p05, p10, p25, p50, p75, p90, p95, p99, p999)
    def get_percentile(metrics_dict, key):
        percentiles = metrics_dict.get("percentiles", {})
        return percentiles.get(key)

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
        "prompt_token_count_p99": get_percentile(prompt_tok_metrics, "p99"),
        "output_token_count_mean": output_tok_metrics.get("mean"),
        "output_token_count_p99": get_percentile(output_tok_metrics, "p99"),
        "ttft_median": ttft_metrics.get("median"),
        "ttft_p95": get_percentile(ttft_metrics, "p95"),
        "ttft_p1": get_percentile(ttft_metrics, "p01"),
        "ttft_p999": get_percentile(ttft_metrics, "p999"),
        "tpot_median": tpot_metrics.get("median"),
        "tpot_p95": get_percentile(tpot_metrics, "p95"),
        "tpot_p99": get_percentile(tpot_metrics, "p99"),
        "tpot_p999": get_percentile(tpot_metrics, "p999"),
        "tpot_p1": get_percentile(tpot_metrics, "p01"),
        "itl_median": itl_metrics.get("median"),
        "itl_p95": get_percentile(itl_metrics, "p95"),
        "itl_p999": get_percentile(itl_metrics, "p999"),
        "itl_p1": get_percentile(itl_metrics, "p01"),
        "request_latency_median": request_latency_metrics.get("median"),
        "request_latency_min": request_latency_metrics.get("min"),
        "request_latency_max": request_latency_metrics.get("max"),
        "successful_requests": successful_reqs,
        "errored_requests": errored_reqs,
        "uuid": uuid,
        "ttft_mean": ttft_metrics.get("mean"),
        "ttft_p99": get_percentile(ttft_metrics, "p99"),
        "itl_mean": itl_metrics.get("mean"),
        "itl_p99": get_percentile(itl_metrics, "p99"),
        "runtime_args": runtime_args,
        "guidellm_start_time_ms": guidellm_start_time_ms,
        "guidellm_end_time_ms": guidellm_end_time_ms,
        "image_tag": image_tag,
        "guidellm_version": guidellm_version,
    }

    return row


def parse_guidellm_json(
    json_path,
    accelerator,
    model_name,
    version,
    tp_size,
    runtime_args,
    image_tag,
    guidellm_version,
):
    """Parse guidellm 0.5.x JSON benchmark results.

    Args:
        json_path: Path to the JSON file.
        accelerator: Accelerator type.
        model_name: Name of the AI model.
        version: Version of the inference server.
        tp_size: Tensor parallelism size.
        runtime_args: Runtime configuration arguments.
        image_tag: Container image tag used for the run.
        guidellm_version: Version of guidellm used to run the benchmark.

    Returns:
        DataFrame: Processed benchmark results.
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

    # Check guidellm version
    metadata = data.get("metadata", {})
    guidellm_version = metadata.get("guidellm_version", "unknown")
    print(f"Detected guidellm version: {guidellm_version}")

    all_run_data = []

    if not data.get("benchmarks"):
        print("Error: JSON file does not contain a 'benchmarks' key.")
        return None

    benchmarks = data["benchmarks"]

    # Get global data config (prompt_tokens, output_tokens)
    global_args = data.get("args", {})
    global_data_config = global_args.get("data", [])

    # Extract aggregated guidellm start and end times from scheduler_metrics
    start_times = []
    end_times = []
    for benchmark in benchmarks:
        scheduler_metrics = benchmark.get("scheduler_metrics", {})
        if "start_time" in scheduler_metrics:
            start_times.append(scheduler_metrics["start_time"])
        if "end_time" in scheduler_metrics:
            end_times.append(scheduler_metrics["end_time"])

    # Get min start_time and max end_time, convert to milliseconds
    guidellm_start_time_ms = int(min(start_times) * 1000) if start_times else ""
    guidellm_end_time_ms = int(max(end_times) * 1000) if end_times else ""

    print(f"Processing {len(benchmarks)} benchmark sections...")

    for i, benchmark in enumerate(benchmarks):
        row_data = process_benchmark_section(
            benchmark,
            accelerator,
            model_name,
            version,
            tp_size,
            runtime_args,
            global_data_config,
            image_tag,
            guidellm_version,
            guidellm_start_time_ms,
            guidellm_end_time_ms,
        )
        if row_data:
            all_run_data.append(row_data)
            streams = (
                benchmark.get("config", {}).get("strategy", {}).get("streams", "?")
            )
            print(
                f"  Processed benchmark {i + 1}/{len(benchmarks)} (streams={streams})"
            )

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
        description="Import guidellm 0.5.x JSON results into the consolidated benchmark CSV.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "json_file", help="Path to the guidellm JSON output file to import"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name (e.g., 'meta-llama/Llama-3.3-70B-Instruct')",
    )
    parser.add_argument(
        "--version",
        required=True,
        help="Version/framework identifier (e.g., 'vLLM-0.13.0')",
    )
    parser.add_argument("--tp", type=int, required=True, help="Tensor parallelism size")
    parser.add_argument(
        "--accelerator", required=True, help="Accelerator type (e.g., 'H200', 'MI300X')"
    )
    parser.add_argument(
        "--runtime-args",
        required=True,
        help="Runtime arguments used for the inference server",
    )
    parser.add_argument(
        "--image-tag",
        required=True,
        help="Container image tag used for the run (e.g., 'vllm/vllm-openai:v0.13.0')",
    )
    parser.add_argument(
        "--guidellm-version",
        required=True,
        help="Version of guidellm used to run the benchmark (e.g., 'v0.5.x', 'v0.3.0')",
    )
    parser.add_argument(
        "--csv-file",
        default="new_benchmarks.csv",
        help="Path to the output CSV file (default: new_benchmarks.csv)",
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
        args.image_tag,
        args.guidellm_version,
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
            "guidellm_start_time_ms",
            "guidellm_end_time_ms",
            "image_tag",
            "guidellm_version",
        ]

        for col in fieldnames:
            if col not in combined_df.columns:
                combined_df[col] = None

        combined_df = combined_df[fieldnames]
        combined_df.to_csv(args.csv_file, index=False)
        print(f"Successfully saved to {args.csv_file}")
    else:
        print(
            "No valid benchmark data was loaded. Exiting without creating a CSV file."
        )


if __name__ == "__main__":
    main()
