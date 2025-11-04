# MLPerf Original Dataset Files

## Purpose

This directory contains the original MLPerf Inference dataset files used to generate the lightweight CSV summaries. These files are **NOT version controlled** (see `.gitignore`) due to their large size.

## Downloading Datasets

All datasets can be downloaded from the [MLCommons Inference Benchmark Data Download](https://inference.mlcommons-storage.org/index.html) page.

### Example Download Command

Run from the `mlperf-data/original/` directory:

```bash
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) -d ./ https://inference.mlcommons-storage.org/metadata/deepseek-r1-datasets-fp8-eval.uri
```

Other datasets (Llama 3.1 8B, Llama 2 70B, etc.) can be downloaded similarly using their respective URLs from the MLCommons download page.

## Expected Files

After downloading and extracting, you should have:

- `mlperf_deepseek_r1_dataset_4388_fp8_eval.pkl` - DeepSeek-R1 dataset

## Generating Summaries

After placing files here, run the generation script located in this folder:

```bash
cd /path/to/performance-dashboard
python mlperf-data/original/generate_dataset_summaries.py
```

This will extract token length statistics into lightweight CSV files in `../summaries/` that ARE version controlled.

## Note

- Original files can be several MB to 20+ MB
- Generated summaries are only 40-180 KB
- Only summaries are needed for the dashboard to run
- Keep original files locally for regenerating summaries if needed
