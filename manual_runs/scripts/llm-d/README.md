# Import Manual Runs JSON — LLM-D

Script to process and import benchmark results from **guidellm v0.5.0** and above JSON files into the LLM-D performance dashboard CSV format. This script adds LLM-D-specific deployment metadata columns (DP, EP, replicas, pod counts, router config) on top of the standard benchmark metrics.

## Prerequisites

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install pandas
```

## Usage

```bash
python import_manual_runs_llm_d.py <json_file> \
  --model <model_name> \
  --version <framework_version> \
  --tp <tensor_parallelism> \
  --accelerator <gpu_type> \
  --runtime-args <server_args> \
  --llm-d \
  --dp <data_parallelism> \
  --ep <expert_parallelism> \
  --replicas <replica_count> \
  --prefill-pod-count <prefill_pods> \
  --decode-pod-count <decode_pods> \
  --router-config <router_yaml_or_json> \
  --csv-file <output_csv>
```

## Arguments

| Argument              | Required    | Description                                          | Example                                 |
| --------------------- | ----------- | ---------------------------------------------------- | --------------------------------------- |
| `json_file`           | Yes         | Path to guidellm JSON output file                    | `mixtral-llmd.json`                     |
| `--model`             | Yes         | Model name (HuggingFace format)                      | `mistralai/Mixtral-8x22B-Instruct-v0.1` |
| `--version`           | Yes         | Framework/version identifier                         | `LLM-D-0.1.0`                           |
| `--tp`                | Yes         | Tensor parallelism size                              | `4`, `8`                                |
| `--accelerator`       | Yes         | GPU/accelerator type                                 | `H200`, `MI300X`                        |
| `--runtime-args`      | Yes         | Server runtime configuration                         | See examples below                      |
| `--llm-d`             | Yes         | Enable LLM-D mode (adds deployment metadata columns) | _(flag, no value)_                      |
| `--dp`                | Yes (LLM-D) | Data parallelism size                                | `2`, `4`                                |
| `--ep`                | Yes (LLM-D) | Expert parallelism size                              | `1`, `2`                                |
| `--replicas`          | Yes (LLM-D) | Number of replicas                                   | `1`, `4`                                |
| `--router-config`     | Yes (LLM-D) | Router/endpoint picker configuration (YAML or JSON)  | `"round-robin"`, `config.yaml`          |
| `--prefill-pod-count` | No          | Number of prefill pods (default: 0)                  | `2`, `4`                                |
| `--decode-pod-count`  | No          | Number of decode pods (default: 0)                   | `4`, `8`                                |
| `--image-tag`         | No (LLM-D)  | Container image tag (optional in LLM-D mode)         | `llm-d:v0.1.0`                          |
| `--guidellm-version`  | No (LLM-D)  | guidellm version (optional in LLM-D mode)            | `v0.5.1`                                |
| `--notes`             | No          | Free-form note applied to every row                  | `"baseline run"`                        |
| `--csv-file`          | No          | Output CSV path (default: `new_benchmarks.csv`)      | `mixtral-llmd.csv`                      |

## Examples

### LLM-D — Mixtral 8x22B with Disaggregated Serving

```bash
python import_manual_runs_llm_d.py \
  mixtral-llmd.json \
  --model "mistralai/Mixtral-8x22B-Instruct-v0.1" \
  --version "LLM-D-0.1.0" \
  --tp 4 \
  --accelerator "H200" \
  --runtime-args "tp: 4; ep: 2; max-model-len: 131072" \
  --llm-d \
  --dp 2 \
  --ep 2 \
  --replicas 4 \
  --prefill-pod-count 2 \
  --decode-pod-count 4 \
  --router-config "round-robin" \
  --csv-file "mixtral-llmd.csv"
```

### LLM-D — With Notes

```bash
python import_manual_runs_llm_d.py \
  gpt-oss-llmd.json \
  --model "openai/gpt-oss-120b" \
  --version "LLM-D-0.1.0" \
  --tp 8 \
  --accelerator "H200" \
  --runtime-args "tp: 8; max-model-len: 131072; gpu-memory-utilization: 0.9" \
  --llm-d \
  --dp 1 \
  --ep 1 \
  --replicas 1 \
  --prefill-pod-count 0 \
  --decode-pod-count 0 \
  --router-config "least-load" \
  --notes "baseline run" \
  --csv-file "gpt-oss-llmd.csv"
```

## Appending to LLM-D Dashboard CSV

After generating a CSV file, append it to the LLM-D dashboard data (skip the header):

```bash
tail -n +2 my-benchmark.csv >> ../../../llmd-dashboard.csv
```

## Output CSV Columns

The LLM-D mode outputs 51 columns including deployment metadata:

| #   | Column                    | Description                                          |
| --- | ------------------------- | ---------------------------------------------------- |
| 1   | `run`                     | Unique run identifier (`{accelerator}-{model}-{tp}`) |
| 2   | `accelerator`             | GPU type (H200, MI300X, etc.)                        |
| 3   | `model`                   | Model name                                           |
| 4   | `version`                 | Framework version                                    |
| 5   | `prompt toks`             | Configured prompt token count                        |
| 6   | `output toks`             | Configured output token count                        |
| 7   | `TP`                      | Tensor parallelism size                              |
| 8   | `DP`                      | Data parallelism size                                |
| 9   | `EP`                      | Expert parallelism size                              |
| 10  | `replicas`                | Number of replicas                                   |
| 11  | `prefill_pod_count`       | Number of prefill pods                               |
| 12  | `decode_pod_count`        | Number of decode pods                                |
| 13  | `router_config`           | Router/endpoint picker configuration                 |
| 14  | `measured concurrency`    | Actual measured concurrency                          |
| 15  | `intended concurrency`    | Requested concurrency (streams)                      |
| 16  | `measured rps`            | Requests per second                                  |
| 17  | `output_tok/sec`          | Output tokens per second                             |
| 18  | `total_tok/sec`           | Total tokens per second                              |
| 19  | `prompt_token_count_mean` | Mean prompt token count                              |
| 20  | `prompt_token_count_p99`  | P99 prompt token count                               |
| 21  | `output_token_count_mean` | Mean output token count                              |
| 22  | `output_token_count_p99`  | P99 output token count                               |
| 23  | `ttft_median`             | Time to first token - median (ms)                    |
| 24  | `ttft_p95`                | Time to first token - P95 (ms)                       |
| 25  | `ttft_p1`                 | Time to first token - P1 (ms)                        |
| 26  | `ttft_p999`               | Time to first token - P99.9 (ms)                     |
| 27  | `tpot_median`             | Time per output token - median (ms)                  |
| 28  | `tpot_p95`                | Time per output token - P95 (ms)                     |
| 29  | `tpot_p99`                | Time per output token - P99 (ms)                     |
| 30  | `tpot_p999`               | Time per output token - P99.9 (ms)                   |
| 31  | `tpot_p1`                 | Time per output token - P1 (ms)                      |
| 32  | `itl_median`              | Inter-token latency - median (ms)                    |
| 33  | `itl_p95`                 | Inter-token latency - P95 (ms)                       |
| 34  | `itl_p999`                | Inter-token latency - P99.9 (ms)                     |
| 35  | `itl_p1`                  | Inter-token latency - P1 (ms)                        |
| 36  | `request_latency_median`  | End-to-end request latency - median (s)              |
| 37  | `request_latency_min`     | End-to-end request latency - minimum (s)             |
| 38  | `request_latency_max`     | End-to-end request latency - maximum (s)             |
| 39  | `successful_requests`     | Number of successful requests                        |
| 40  | `errored_requests`        | Number of errored requests                           |
| 41  | `uuid`                    | Unique benchmark run ID                              |
| 42  | `ttft_mean`               | Time to first token - mean (ms)                      |
| 43  | `ttft_p99`                | Time to first token - P99 (ms)                       |
| 44  | `itl_mean`                | Inter-token latency - mean (ms)                      |
| 45  | `itl_p99`                 | Inter-token latency - P99 (ms)                       |
| 46  | `runtime_args`            | Server configuration arguments                       |
| 47  | `guidellm_start_time_ms`  | Benchmark start time (epoch ms)                      |
| 48  | `guidellm_end_time_ms`    | Benchmark end time (epoch ms)                        |
| 49  | `image_tag`               | Container image used                                 |
| 50  | `guidellm_version`        | guidellm version used                                |
| 51  | `notes`                   | Free-form notes                                      |

## Notes

- This script is designed for **guidellm v0.5.0** and above JSON format
- The `--llm-d` flag is required to enable LLM-D mode with deployment metadata columns
- When `--llm-d` is set, `--image-tag` and `--guidellm-version` become optional (auto-detected from JSON metadata when available)
- Required LLM-D flags: `--dp`, `--ep`, `--replicas`, `--router-config`
