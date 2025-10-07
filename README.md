# RHAIIS Performance Dashboard

A comprehensive performance analysis dashboard for RHAIIS (Red Hat AI Inference server) benchmarks. This dashboard provides interactive visualizations and analysis of AI model performance across different accelerators, versions, and configurations.

## Features

- **Interactive Performance Plots**: Compare throughput, latency, and efficiency metrics
- **Cost Analysis**: Calculate cost per million tokens with cloud provider pricing
- **Performance Rankings**: Identify top performers by throughput and latency
- **Regression Analysis**: Track performance changes between versions
- **Runtime Configuration Tracking**: View inference server arguments used
- **Multi-Accelerator Support**: Compare H200, MI300X, and TPU performance

## Key Metrics Analyzed

- **Throughput**: Output tokens per second
- **Latency**: Time to First Token (TTFT) and Inter-Token Latency (ITL)
- **Efficiency**: Throughput per tensor parallelism unit
- **Cost Efficiency**: Cost per million tokens across cloud providers
- **Error Rates**: Request success/failure analysis
- **Concurrency Performance**: Performance at different load levels

## Directory Structure

```
performance-dashboard/
├── dashboard.py                    # Main dashboard application
├── dashboard_styles.py             # CSS styling file
├── pyproject.toml                  # Project metadata and dependencies
├── requirements.txt                # Python dependencies
├── Dockerfile.openshift            # Container build configuration
├── .pre-commit-config.yaml         # Pre-commit hooks configuration
├── Makefile                        # Development commands
├── manual_runs/scripts/            # Data processing scripts
│   └── import_manual_run_jsons.py  # Import manual benchmark results
├── deploy/                         # OpenShift deployment files
│   ├── openshift-deployment.yaml   # Application deployment
│   ├── openshift-service.yaml      # Service configuration
│   └── openshift-route.yaml        # Route/ingress configuration
├── tests/                          # Test suite
│   ├── test_data_processing.py     # Data processing unit tests
│   ├── test_import_script.py       # Import script tests
│   ├── test_integration.py         # Integration tests
│   ├── conftest.py                 # Shared fixtures
│   └── README.md                   # Test documentation
├── docs/                           # Documentation
│   └── CODE_QUALITY.md             # Code quality guidelines
└── data/                           # Data files (excluded from git)
    └── consolidated_dashboard.csv  # Benchmark data, Get the latest csv data from the AWS S3 bucket.
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

3. **Add your data**:
   - Place your `consolidated_dashboard.csv` in the root directory
   - Use the utilities in `manual_runs/scripts/` to process new benchmark data

4. **Run the dashboard**:

   ```bash
   streamlit run dashboard.py
   ```

5. **Access**: Open http://localhost:8501 in your browser

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
   # Ensure you have the latest consolidated_dashboard.csv in the root directory
   # You can download it from the AWS S3 bucket or generate it using the scripts
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

When you have new data or code changes:

1. **Rebuild the image** with updated data:

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

- `STREAMLIT_SERVER_HEADLESS=true`: Headless mode for production
- `STREAMLIT_SERVER_PORT=8501`: Server port
- `STREAMLIT_SERVER_ADDRESS=0.0.0.0`: Listen address

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
