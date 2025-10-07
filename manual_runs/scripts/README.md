# Manual Runs Documentation

This directory contains data processing scripts for the RHAIIS Performance Dashboard.

## import_manual_run_jsons.py

A script to import guidellm JSON benchmark results into the consolidated CSV format used by the performance dashboard.

### Purpose

This script processes JSON output files from guidellm benchmarks and converts them into CSV format that can be consumed by the dashboard. It's designed for importing manual benchmark runs or results from external benchmark tools.

### Prerequisites

- Python 3.9+
- Required packages: `pandas`, `numpy`, `json` (from requirements.txt)
- Access to guidellm JSON output files

### Usage

```bash
python import_manual_run_jsons.py <json_file> [options]
```

### Parameters

| Parameter        | Required                              | Description                                    |
| ---------------- | ------------------------------------- | ---------------------------------------------- |
| `json_file`      | Path to the guidellm JSON output file | `benchmark_results.json`                       |
| `--model`        | Model name/identifier                 | `RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic`  |
| `--version`      | Version/framework identifier          | `vLLM-0.10.1`, `RHAIIS-3.2` etc                |
| `--tp`           | Tensor parallelism size               | `8`                                            |
| `--accelerator`  | Accelerator type                      | `H200`, `MI300X`, or `TPU`                     |
| `--runtime-args` | Runtime server arguments              | `tensor-parallel-size: 8; max-model-len: 8192` |
| `--csv-file`     | Output CSV file path                  | `new_benchmarks.csv` (default)                 |

### Examples

#### Basic Usage

```bash
python import_manual_run_jsons.py benchmark_results.json \
  --model "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic" \
  --version "vLLM-0.10.1" \
  --tp 8 \
  --accelerator "H200" \
  --runtime-args "tensor-parallel-size: 8; max-model-len: 8192; trust-remote-code: True"
```

#### Multiple Benchmark Results

```bash
# Process multiple JSON files and append to the same CSV
python import_manual_run_jsons.py h200_results.json \
  --model "RedHatAI/Llama-4-Maverick-17B-128E-Instruct-FP8" \
  --version "vLLM-0.10.1" \
  --tp 8 \
  --accelerator "H200" \
  --runtime-args "tensor-parallel-size: 8; max-model-len: 8192; gpu-memory-utilization: 0.92" \
  --csv-file "consolidated_benchmarks.csv"

python import_manual_run_jsons.py mi300x_results.json \
  --model "RedHatAI/Llama-4-Maverick-17B-128E-Instruct-FP8" \
  --version "vLLM-0.10.1" \
  --tp 8 \
  --accelerator "MI300X" \
  --runtime-args "tensor-parallel-size: 8; max-model-len: 8192; gpu-memory-utilization: 0.92" \
  --csv-file "consolidated_benchmarks.csv"
```

#### Different Configurations

```bash
# Different TP sizes
python import_manual_run_jsons.py tp4_results.json \
  --model "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic" \
  --version "RHAIIS-3.2.1" \
  --tp 4 \
  --accelerator "H200" \
  --runtime-args "tensor-parallel-size: 4; max-model-len: 8192"

python import_manual_run_jsons.py tp2_results.json \
  --model "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic" \
  --version "RHAIIS-3.2.1" \
  --tp 2 \
  --accelerator "H200" \
  --runtime-args "tensor-parallel-size: 2; max-model-len: 8192"
```

### Input File Format

The script expects guidellm JSON output files.

### Output Format

The script generates CSV files with the following columns:

- `run`: Combined accelerator-model-TP identifier
- `accelerator`: Hardware accelerator type
- `model`: Model name
- `version`: Version/framework identifier
- `prompt toks`, `output toks`: Token counts from request configuration
- `TP`: Tensor parallelism size
- `measured concurrency`, `intended concurrency`: Concurrency metrics
- `output_tok/sec`, `total_tok/sec`: Throughput metrics
- Various latency percentiles (`ttft_p95`, `itl_p95`, etc.)
- `successful_requests`, `errored_requests`: Request counts
- `runtime_args`: Formatted runtime arguments string
- Additional performance metrics...

### Runtime Arguments Format

The `--runtime-args` parameter should be a semicolon-separated string of key-value pairs:

```bash
--runtime-args "tensor-parallel-size: 8; max-model-len: 8192; trust-remote-code: True; gpu-memory-utilization: 0.92; disable-log-requests: True"
```

### Integration with Dashboard

After processing JSON files:

1. **Merge with existing data:**

   ```bash
   # Append new results to consolidated dashboard CSV
   cat new_benchmarks.csv >> consolidated_dashboard.csv
   ```

2. **Update dashboard data:**
   - Copy the consolidated CSV to your dashboard directory
   - Rebuild the container image if using containerized deployment
   - Restart the dashboard application

3. **Verify in dashboard:**
   - Check that new data appears in the filters
   - Validate metrics and visualizations
   - Test performance comparisons

For more information about the overall dashboard and deployment, see the main [README.md](../README.md).
