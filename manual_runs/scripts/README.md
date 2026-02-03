# Import Manual Runs JSON v2

Script to process and import benchmark results from **guidellm v0.5.0** and above JSON files into the performance dashboard CSV format.

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
python import_manual_runs_json_v2.py <json_file> \
  --model <model_name> \
  --version <framework_version> \
  --tp <tensor_parallelism> \
  --accelerator <gpu_type> \
  --runtime-args <server_args> \
  --image-tag <container_image> \
  --guidellm-version <guidellm_version> \
  --csv-file <output_csv>
```

## Arguments

| Argument             | Required | Description                                     | Example                                 |
| -------------------- | -------- | ----------------------------------------------- | --------------------------------------- |
| `json_file`          | Yes      | Path to guidellm JSON output file               | `llama-70b-vllm.json`                   |
| `--model`            | Yes      | Model name (HuggingFace format)                 | `meta-llama/Llama-3.3-70B-Instruct`     |
| `--version`          | Yes      | Framework/version identifier                    | `vLLM-0.13.0`, `TRT-LLM-1.2.0rc2.post1` |
| `--tp`               | Yes      | Tensor parallelism size                         | `4`, `8`                                |
| `--accelerator`      | Yes      | GPU/accelerator type                            | `H200`, `MI300X`, `TPU`                 |
| `--runtime-args`     | Yes      | Server runtime configuration                    | See examples below                      |
| `--image-tag`        | Yes      | Container image tag                             | `vllm/vllm-openai:v0.13.0`              |
| `--guidellm-version` | Yes      | guidellm version used                           | `v0.5.1`, `v0.3.0`                      |
| `--csv-file`         | No       | Output CSV path (default: `new_benchmarks.csv`) | `llama-70b-vllm.csv`                    |

## Examples

### vLLM - Llama 3.3 70B

```bash
python import_manual_runs_json_v2.py \
  llama-70b-vllm.json \
  --model "meta-llama/Llama-3.3-70B-Instruct" \
  --version "vLLM-0.13.0" \
  --tp 4 \
  --accelerator "H200" \
  --runtime-args "tensor-parallel-size: 4; max-model-len: 2248; gpu-memory-utilization: 0.9; kv-cache-dtype: fp8; async-scheduling: true; no-enable-prefix-caching: true; max-num-batched-tokens: 8192; dtype: auto" \
  --image-tag "vllm/vllm-openai:v0.13.0" \
  --guidellm-version "v0.5.1" \
  --csv-file "llama-70b-vllm.csv"
```

### TRT-LLM - Llama 3.3 70B

```bash
python import_manual_runs_json_v2.py \
  llama-70b-trtllm.json \
  --model "meta-llama/Llama-3.3-70B-Instruct" \
  --version "TRT-LLM-1.2.0rc2.post1" \
  --tp 8 \
  --accelerator "H200" \
  --runtime-args "tp_size: 8; max_batch_size: 1024; max_num_tokens: 16384; max_seq_len: 2248; kv_cache_config.dtype: fp8" \
  --image-tag "nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc2.post1" \
  --guidellm-version "v0.5.1" \
  --csv-file "llama-70b-trtllm.csv"
```

## Appending to Consolidated Dashboard

After generating a CSV file, append it to the main dashboard (skip the header):

```bash
tail -n +2 my-benchmark.csv >> ../consolidated_dashboard.csv
```

## Output CSV Columns

The script outputs 44 columns compatible with the performance dashboard:

| #   | Column                    | Description                                          |
| --- | ------------------------- | ---------------------------------------------------- |
| 1   | `run`                     | Unique run identifier (`{accelerator}-{model}-{tp}`) |
| 2   | `accelerator`             | GPU type (H200, MI300X, etc.)                        |
| 3   | `model`                   | Model name                                           |
| 4   | `version`                 | Framework version                                    |
| 5   | `prompt toks`             | Configured prompt token count                        |
| 6   | `output toks`             | Configured output token count                        |
| 7   | `TP`                      | Tensor parallelism size                              |
| 8   | `measured concurrency`    | Actual measured concurrency                          |
| 9   | `intended concurrency`    | Requested concurrency (streams)                      |
| 10  | `measured rps`            | Requests per second                                  |
| 11  | `output_tok/sec`          | Output tokens per second                             |
| 12  | `total_tok/sec`           | Total tokens per second                              |
| 13  | `prompt_token_count_mean` | Mean prompt token count                              |
| 14  | `prompt_token_count_p99`  | P99 prompt token count                               |
| 15  | `output_token_count_mean` | Mean output token count                              |
| 16  | `output_token_count_p99`  | P99 output token count                               |
| 17  | `ttft_median`             | Time to first token - median (ms)                    |
| 18  | `ttft_p95`                | Time to first token - P95 (ms)                       |
| 19  | `ttft_p1`                 | Time to first token - P1 (ms)                        |
| 20  | `ttft_p999`               | Time to first token - P99.9 (ms)                     |
| 21  | `tpot_median`             | Time per output token - median (ms)                  |
| 22  | `tpot_p95`                | Time per output token - P95 (ms)                     |
| 23  | `tpot_p99`                | Time per output token - P99 (ms)                     |
| 24  | `tpot_p999`               | Time per output token - P99.9 (ms)                   |
| 25  | `tpot_p1`                 | Time per output token - P1 (ms)                      |
| 26  | `itl_median`              | Inter-token latency - median (ms)                    |
| 27  | `itl_p95`                 | Inter-token latency - P95 (ms)                       |
| 28  | `itl_p999`                | Inter-token latency - P99.9 (ms)                     |
| 29  | `itl_p1`                  | Inter-token latency - P1 (ms)                        |
| 30  | `request_latency_median`  | End-to-end request latency - median (s)              |
| 31  | `request_latency_min`     | End-to-end request latency - minimum (s)             |
| 32  | `request_latency_max`     | End-to-end request latency - maximum (s)             |
| 33  | `successful_requests`     | Number of successful requests                        |
| 34  | `errored_requests`        | Number of errored requests                           |
| 35  | `uuid`                    | Unique benchmark run ID                              |
| 36  | `ttft_mean`               | Time to first token - mean (ms)                      |
| 37  | `ttft_p99`                | Time to first token - P99 (ms)                       |
| 38  | `itl_mean`                | Inter-token latency - mean (ms)                      |
| 39  | `itl_p99`                 | Inter-token latency - P99 (ms)                       |
| 40  | `runtime_args`            | Server configuration arguments                       |
| 41  | `guidellm_start_time_ms`  | Benchmark start time (epoch ms)                      |
| 42  | `guidellm_end_time_ms`    | Benchmark end time (epoch ms)                        |
| 43  | `image_tag`               | Container image used                                 |
| 44  | `guidellm_version`        | guidellm version used                                |

## Notes

- This script is designed for **guidellm v0.5.0** and above JSON format
- For older guidellm v0.3.0 results, use `import_manual_runs_json.py`
- If the output CSV already exists, new rows are appended to it
