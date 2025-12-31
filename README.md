# AI Inference Performance Dashboard

A comprehensive performance analysis dashboard for AI inference benchmarks including RHAIIS (Red Hat AI Inference Server), LLM-D (disaggregated LLM inference), and MLPerf submissions. This dashboard provides interactive visualizations and analysis of AI model performance across different accelerators, versions, and configurations.

## Features

### RHAIIS Dashboard

- **Interactive Performance Plots**: Compare throughput, latency, and efficiency metrics
- **Cost Analysis**: Calculate cost per million tokens with cloud provider pricing
- **Performance Rankings**: Identify top performers by throughput and latency
- **Regression Analysis**: Track performance changes between versions
- **Pareto Tradeoff Analysis**: Visualize performance trade-offs between competing objectives
- **Runtime Configuration Tracking**: View inference server arguments used
- **Multi-Accelerator Support**: Compare H200, MI300X, and TPU performance

### LLM-D Dashboard

- **Disaggregated Architecture Analysis**: Analyze LLM-D benchmark results with separated prefill/decode pods
- **Compare with RHAIIS**: Side-by-side performance comparison of LLM-D vs traditional RHAIIS architecture
  - Throughput vs Concurrency comparison
  - Latency analysis (TTFT, TPOT)
  - Detailed metrics and summary statistics
- **Performance Plots**: Interactive visualizations with multiple Y-axis metric options
- **Runtime Configuration Tracking**: View server configurations and deployment parameters

### MLPerf Dashboard

- **Multi-Version Support**: Compare MLPerf v5.0 and v5.1 submissions
- **Benchmark Comparisons**: Analyze performance across different models and scenarios
- **Normalized Result Analysis**: Compare systems with different accelerator counts
- **Dataset Representation**: View token length distributions for evaluation datasets
- **Offline vs Server Comparison**: Analyze performance degradation between scenarios
- **Cross-Version Analysis**: Track how systems perform across MLPerf versions

## Key Metrics Analyzed

- **Throughput**: Output tokens per second, total tokens per second
- **Latency**: Time to First Token (TTFT), Time Per Output Token (TPOT), Inter-Token Latency (ITL)
- **Efficiency**: Throughput per tensor parallelism unit
- **Cost Efficiency**: Cost per million tokens across cloud providers
- **Error Rates**: Request success/failure analysis
- **Concurrency Performance**: Performance at different load levels
- **Disaggregated Architecture** (LLM-D): Prefill/decode pod configurations, replica scaling

## Directory Structure

```
performance-dashboard/
├── dashboard.py                    # Main dashboard application
├── dashboard_styles.py             # CSS styling file
├── mlperf_datacenter.py            # MLPerf dashboard module
├── llmd_dashboard.py               # LLM-D dashboard module
├── pyproject.toml                  # Project metadata and dependencies
├── requirements.txt                # Python dependencies
├── Dockerfile.openshift            # Container build configuration
├── .pre-commit-config.yaml         # Pre-commit hooks configuration
├── Makefile                        # Development commands
├── consolidated_dashboard.csv      # RHAIIS benchmark data. Get the latest csv data from the AWS S3 bucket.
├── llmd-dashboard.csv              # LLM-D benchmark data. Get the latest csv data from the AWS S3 bucket.
├── mlperf-data/                    # MLPerf data files
│   ├── mlperf-5.1.csv              # MLPerf v5.1 submission data
│   ├── mlperf-5.0.csv              # MLPerf v5.0 submission data
│   ├── summaries/                  # Dataset summaries (version controlled)
│   │   ├── README.md               # Dataset summary documentation
│   │   ├── deepseek-r1.csv         # DeepSeek-R1 token length summary
│   │   ├── llama3-1-8b-datacenter.csv  # Llama 3.1 8B token length summary
│   │   └── llama2-70b-99.csv       # Llama 2 70B token length summary
│   └── original/                   # Original datasets (NOT version controlled)
│       ├── README.md               # Download and usage instructions
│       └── generate_dataset_summaries.py  # Script to generate CSV summaries
├── manual_runs/scripts/            # Data processing scripts
│   └── import_manual_run_jsons.py  # Import manual benchmark results
├── deploy/                         # OpenShift deployment files
│   ├── openshift-deployment-staging.yaml  # Staging deployment (with S3)
│   ├── openshift-service-staging.yaml     # Staging service
│   └── openshift-route-staging.yaml       # Staging route
├── tests/                          # Test suite
│   ├── test_data_processing.py     # Data processing unit tests
│   ├── test_import_script.py       # Import script tests
│   ├── test_integration.py         # Integration tests
│   ├── test_mlperf_datacenter.py   # MLPerf module tests
│   ├── conftest.py                 # Shared fixtures
│   └── README.md                   # Test documentation
└── docs/                           # Documentation
    └── CODE_QUALITY.md             # Code quality guidelines
```

## Quick Start

### Local Development

1. **Clone the repository**:

   ```bash
   git clone https://github.com/openshift-psap/performance-dashboard.git
   cd performance-dashboard
   ```

2. **Set up Python environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

   > **Note**: `boto3` is included for optional S3 integration. See [S3 Configuration](#s3-configuration-optional) for details.

3. **Add your data**:
   - **RHAIIS data**: Place your `consolidated_dashboard.csv` in the root directory
   - **LLM-D data**: Place your `llmd-dashboard.csv` in the root directory
   - **MLPerf data**: MLPerf CSV files are included in `mlperf-data/` directory
   - Use the utilities in `manual_runs/scripts/` to process new benchmark data

4. **Run the dashboard**:

   ```bash
   streamlit run dashboard.py
   ```

5. **Access**: Open http://localhost:8501 in your browser
   - Use the sidebar to switch between "RHAIIS Dashboard", "LLM-D Dashboard", and "MLPerf Dashboard" views

### Development Environment Setup

For a complete development environment with linting, formatting, and code quality tools:

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies from pyproject.toml
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

**Available development commands:**

- `make format` - Auto-format code (Black, Ruff)
- `make lint` - Run linting checks
- `make type-check` - Run static type checking
- `make test` - Run tests with coverage
- `make ci-local` - Run all CI checks locally
- `make clean` - Clean temporary files

**Code Quality:**

- All code is checked with ruff, black, mypy
- Pre-commit hooks enforce code standards
- Tests must pass before merging
- Documentation required for public functions

See [Code Quality Documentation](docs/CODE_QUALITY.md) for detailed information.

### Container Deployment

1. **Build the container**:

   ```bash
   podman build -f Dockerfile.openshift -t performance-dashboard .
   ```

2. **Run locally**:
   ```bash
   podman run -p 8501:8501 performance-dashboard
   ```

### OpenShift Deployment

#### Prerequisites

- OpenShift CLI (`oc`) installed and configured
- Access to an OpenShift cluster with permissions to create projects
- Container registry access (quay.io or internal registry)
- Latest CSV data file in the project directory

#### Step-by-Step Deployment

1. **Create the namespace/project**:

   ```bash
   oc new-project rhaiis-dashboard --display-name="RHAIIS Performance Dashboard"
   ```

2. **Prepare your data**:

   ```bash
   # Ensure you have the latest consolidated_dashboard.csv (RHAIIS) in the root directory
   # You can download it from the AWS S3 bucket or generate it using the scripts

   # Ensure you have llmd-dashboard.csv (LLM-D) in the root directory

   # MLPerf data files are included in mlperf-data/ directory
   # Dataset summaries are in mlperf-data/summaries/
   ```

3. **Build and push the container image**:

   ```bash
   # Build the container image with your data
   podman build -f Dockerfile.openshift -t quay.io/your-username/rhaiis-dashboard:latest .

   # Push to your container registry
   podman push quay.io/your-username/rhaiis-dashboard:latest
   ```

4. **Update the image reference in deployment**:

   # Edit deploy/openshift-deployment.yaml to use your image

5. **Deploy all components**:

   ```bash
   # Deploy the application, service, and route
   oc apply -f deploy/openshift-deployment.yaml
   oc apply -f deploy/openshift-service.yaml
   oc apply -f deploy/openshift-route.yaml
   ```

6. **Access the dashboard**:
   ```bash
   # Get the dashboard URL
   echo "Dashboard URL: http://$(oc get route rhaiis-dashboard -n rhaiis-dashboard -o jsonpath='{.spec.host}')"
   ```

#### Updating the Dashboard

**Option 1: Update Data via S3 (Recommended for data-only changes)**

If S3 is configured, simply upload new CSV files to your S3 bucket:

The dashboard will automatically pick up the new data within 5 minutes (cache TTL).

**Option 2: Rebuild Container (Required for code changes)**

When you have code changes:

1. **Rebuild the image** with updated code:

   ```bash
   podman build -f Dockerfile.openshift -t quay.io/your-username/rhaiis-dashboard:latest .
   podman push quay.io/your-username/rhaiis-dashboard:latest
   ```

2. **Restart the deployment** to use the new image:
   ```bash
   oc rollout restart deployment/rhaiis-dashboard -n rhaiis-dashboard
   ```

## Data Processing

### Processing New Benchmark Data from manual runs

1. **From manual JSON results**:

   ```bash
   python scripts/import_manual_run_jsons.py benchmark.json \
     --model "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic" \
     --version "vLLM-0.10.1" \
     --tp 8 \
     --accelerator "H200" \
     --runtime-args "tensor-parallel-size: 8; max-model-len: 8192"
   ```

2. **Consolidate data**: Merge new results with existing CSV file

## MLPerf Data Management

### MLPerf CSV Files

The dashboard supports multiple MLPerf Inference versions:

- **v5.1**: Latest submission results (`mlperf-data/mlperf-5.1.csv`)
- **v5.0**: Previous version results (`mlperf-data/mlperf-5.0.csv`)

These files are version controlled and included in the repository.

### MLPerf Dataset Summaries

The "Dataset Representation" section uses lightweight CSV summaries of token length distributions:

**Available summaries** (in `mlperf-data/summaries/`):

- `deepseek-r1.csv` - DeepSeek-R1 evaluation dataset
- `llama3-1-8b-datacenter.csv` - Llama 3.1 8B CNN dataset
- `llama2-70b-99.csv` - Llama 2 70B Open Orca dataset

### Managing Original Datasets

Original dataset files are stored in `mlperf-data/original/` and are **NOT version controlled** due to their size.

**To download and add a new dataset:**

1. Download the dataset to `mlperf-data/original/`
2. Update `generate_dataset_summaries.py` with a new processor function
3. Run the script to generate the summary
4. Update `mlperf_datacenter.py` to map the model name to the summary file

See `mlperf-data/original/README.md` and `mlperf-data/summaries/README.md` for detailed instructions.

## LLM-D Data Management

### LLM-D CSV Format

The `llmd-dashboard.csv` file contains benchmark results for disaggregated LLM inference with the following key columns:

- **Configuration**: `accelerator`, `model`, `version`, `TP`, `replicas`, `prefill_pod_count`, `decode_pod_count`
- **Workload**: `prompt toks` (ISL), `output toks` (OSL), `intended concurrency`
- **Performance Metrics**: `output_tok/sec`, `total_tok/sec`, `ttft_median`, `ttft_p95`, `tpot_median`, `itl_median`, `request_latency_median`
- **Success Metrics**: `successful_requests`, `errored_requests`
- **Metadata**: `uuid`, `runtime_args`, `router_config`

### Compare with RHAIIS

The LLM-D dashboard includes a comparison feature that:

- Loads RHAIIS data from `consolidated_dashboard.csv`
- Filters to matching accelerators and models
- Compares LLM-D (1 replica only) vs RHAIIS performance
- Supports ISL/OSL profile matching for fair comparisons

### Testing

The project includes a comprehensive test suite with unit and integration tests.

**Run all tests:**

```bash
pytest tests/
```

**Run with coverage:**

```bash
pytest tests/ --cov=. --cov-report=html
```

**Quick test command:**

```bash
make test
```

**Test Categories:**

- **Data Processing Tests** - Core data manipulation functions
- **Import Script Tests** - JSON import and parsing
- **Integration Tests** - End-to-end workflows

See [tests/README.md](tests/README.md) for detailed test documentation.

## 🔧 Configuration

### Environment Variables

#### Streamlit Configuration

- `STREAMLIT_SERVER_HEADLESS=true`: Headless mode for production
- `STREAMLIT_SERVER_PORT=8501`: Server port
- `STREAMLIT_SERVER_ADDRESS=0.0.0.0`: Listen address

#### S3 Configuration (Optional)

The dashboard can load CSV data directly from an AWS S3 bucket instead of local files. This is useful for production deployments where data is updated externally.

| Variable                | Description                               | Default                      |
| ----------------------- | ----------------------------------------- | ---------------------------- |
| `S3_BUCKET`             | S3 bucket name (enables S3 mode when set) | _(none)_                     |
| `S3_KEY`                | Path to RHAIIS CSV in bucket              | `consolidated_dashboard.csv` |
| `S3_KEY_LLMD`           | Path to LLM-D CSV in bucket               | `llmd-dashboard.csv`         |
| `S3_REGION`             | AWS region                                | `us-east-1`                  |
| `AWS_ACCESS_KEY_ID`     | AWS access key (for private buckets)      | _(none)_                     |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key (for private buckets)      | _(none)_                     |

**Behavior:**

- If `S3_BUCKET` is set, data is loaded from S3 with a 5-minute cache
- If S3 fails, falls back to local CSV files
- If `S3_BUCKET` is not set, uses local files only

#### Local Testing with S3

```bash
# Set environment variables
export AWS_ACCESS_KEY_ID='your-key'
export AWS_SECRET_ACCESS_KEY='your-secret'
export S3_BUCKET='your-s3-bucket'
export S3_KEY='location of the consolidated-dashboard.csv file in the bucket'
export S3_KEY_LLMD='location of the llmd-dashboard.csv file in the bucket'
export S3_REGION='us-east-1'

# Run the dashboard
streamlit run dashboard.py
```

### Data Requirements

- **CSV Format**: Must include columns for model, version, accelerator, TP, metrics
- **Runtime Args**: Semicolon-separated key-value pairs
- **Benchmark Profiles**: Support for different prompt/output token configurations

## Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Set up development environment**: `pip install -e ".[dev]"`
4. **Install pre-commit hooks**: `pre-commit install`
5. **Make changes and test locally**: `pytest tests/`
6. **Run code quality checks**: `make ci-local`
7. **Update documentation as needed**
8. **Submit a merge request**

**Development Workflow:**

```bash
# 1. Create feature branch
git checkout -b feature/my-feature

# 2. Make changes
# ... edit code ...

# 3. Run tests
pytest tests/

# 4. Format and lint
make format
make lint

# 5. Commit (pre-commit hooks will run)
git add .
git commit -m "Add feature"

# 6. Push and create a Pull request against main
git push origin feature/my-feature
```

---

**⚠️ CONFIDENTIAL**: This dashboard displays performance data for internal use only.
