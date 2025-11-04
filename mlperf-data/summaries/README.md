# MLPerf Dataset Summaries

## Overview

This directory contains lightweight CSV summaries of MLPerf Inference dataset token lengths. These summaries are used by the dashboard's "Dataset Representation" section to display input/output token distribution statistics.

## Summary Files

Each CSV file contains two columns:

- `input_length`: Number of tokens in the prompt/input
- `output_length`: Number of tokens in the completion/output

### Available Datasets

- `deepseek-r1.csv` - DeepSeek-R1 model evaluation dataset
- `llama3-1-8b-datacenter.csv` - LLaMA 3.1 8B datacenter dataset
- `llama2-70b-99.csv` - LLaMA 2 70B dataset (99% and 99.9% variants)

## Generating Summaries

If you need to add a new dataset summary:

1. Place the original dataset file (`.pkl`, `.pkl.gz`, or `.json`) in the `mlperf-data/original/` directory
2. Update the script `mlperf-data/original/generate_dataset_summaries.py` to add a new processor function
3. Run the generation script from the project root:
   ```bash
   cd /path/to/performance-dashboard
   python mlperf-data/original/generate_dataset_summaries.py
   ```
4. The script will create/update CSV summaries in this directory

**Note:** The `mlperf-data/original/` folder is not version controlled (in `.gitignore`) due to large file sizes.

## File Size

Summary CSV files are typically 40-180 KB compared to original datasets (1-20 MB), making them suitable for version control and Docker images.

## Usage in Dashboard

To add a new model to the dashboard's Dataset Representation section, update the `dataset_map` in `mlperf_datacenter.py`:

```python
dataset_map = {
    "model-name": "mlperf-data/summaries/model-name.csv",
}
```
