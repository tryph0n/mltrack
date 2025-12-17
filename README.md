```txt
! DISCLAIMER !

This project is provided as a resource for Jedha students. Made with lots of coffee and love.
This is an opinionated approach. I strongly advise against using it in production as test coverage is insufficient, CI/CD is virtually non-existent, and no load testing has been performed: use it at your own risk.
```

# MLtrack

Local-first MLOps stack for model metrics comparison. Train models on any compute platform (local, Colab, AWS, Lambda AI), track experiments with MLflow, and compare results in a unified dashboard.

## Architecture

### System Overview

```mermaid
graph TB
    subgraph "Training Pipeline"
        main[main.py] --> pipelines[pipelines.py]
        pipelines --> loaders[loaders.py]
        pipelines --> models[models.py]
        pipelines --> storage[storage.py]
    end

    subgraph "Config & Utils"
        config[config.py]
        logger[logger.py]
    end

    subgraph "MLflow Stack"
        mlflow[MLflow Server<br/>:5000]
        s3[S3 Storage<br/>mltrack-mlflow]
        postgres[(PostgreSQL<br/>:5432)]
    end

    subgraph "Dashboard"
        streamlit[streamlit/app.py]
        data[streamlit/data.py]
    end

    pipelines --> mlflow
    mlflow --> postgres
    mlflow --> s3
    storage --> s3
    data --> mlflow
    streamlit --> data

    main -.-> config
    main -.-> logger
    pipelines -.-> logger
```

### Module Structure

| Module | Purpose |
|--------|---------|
| `loaders.py` | Data loading abstraction (DataLoader base class, IrisLoader) |
| `models.py` | Model registry for sklearn classifiers (LogisticRegression, RandomForest, SVM) |
| `pipelines.py` | Training pipeline orchestration with MLflow tracking |
| `storage.py` | S3 storage handler for datasets and trained models |
| `logger.py` | Logging configuration (local/production mode) |
| `config.py` | MLflow and AWS configuration from environment |
| `train.py` | Standalone training script (demo with Iris dataset) |

## Prerequisites

- **Docker** and Docker Compose v2
- **uv** (Python package manager) - [Install](https://github.com/astral-sh/uv)
- **AWS Account** with S3 access
- **Python 3.10+**

## Quick Start (Local Development)

1. Clone the repository
2. Copy and configure environment variables:

```bash
cp .env.example .env
# Edit .env with your AWS credentials (see AWS Setup below)
```

3. Start the infrastructure:

```bash
docker compose up -d
```

This starts:
- PostgreSQL (database for MLflow metadata)
- MLflow server at http://localhost:5000
- Streamlit dashboard at http://localhost:8501

4. Install Python dependencies:

```bash
uv sync
```

5. Run the demo training script:

```bash
uv run python src/mltrack/main.py --help
```

6. View results in the dashboard at http://localhost:8501

## AWS Setup

### 1. Create S3 Buckets

Create two S3 buckets in your AWS account:

```bash
aws s3 mb s3://mltrack-mlflow --region eu-west-3
aws s3 mb s3://mltrack-data --region eu-west-3
```

**Bucket folder structure:**

- `mltrack-mlflow/artifacts/` - MLflow model artifacts and experiment data (populated automatically)
- `mltrack-data/training/` - Training datasets (upload manually or via code)
- `mltrack-data/inference/` - Inference data and predictions (upload manually or via code)

**Note:** The demo training script (`src/mltrack/train.py`) uses sklearn's built-in Iris dataset, so `mltrack-data` will remain empty unless you upload your own datasets. The `mltrack-mlflow` bucket is automatically populated with model artifacts when you run training.

### 2. Create IAM User

Create an IAM user for programmatic access:

1. Go to AWS Console > IAM > Users > Add User
2. User name: `mltrack-service`
3. Access type: **Programmatic access**
4. Click "Next: Permissions"

### 3. Create IAM Policy

Create a custom policy with minimal required permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "MLtrackS3Access",
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::mltrack-mlflow",
                "arn:aws:s3:::mltrack-mlflow/*",
                "arn:aws:s3:::mltrack-data",
                "arn:aws:s3:::mltrack-data/*"
            ]
        }
    ]
}
```

**Steps:**
1. IAM > Policies > Create Policy
2. Select JSON tab and paste the policy above
3. Name: `MLtrackS3Policy`
4. Create policy
5. Attach policy to `mltrack-service` user

### 4. Get Credentials

After creating the user:

1. Download the CSV file with credentials (only shown once!)
2. Note the `Access Key ID` and `Secret Access Key`

### 5. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_REGION=eu-west-3
S3_BUCKET_MLFLOW=mltrack-mlflow
S3_BUCKET_DATA=mltrack-data
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_SERVER_ALLOWED_HOSTS=*
```

**Configuration notes:**
- For local, `MLFLOW_SERVER_ALLOWED_HOSTS=*` allows Streamlit to access MLflow from Docker (prevents "Invalid Host header" errors)
- For production, restrict allowed hosts: `MLFLOW_SERVER_ALLOWED_HOSTS=yourdomain.com,localhost`

**Security notes:**
- Rotate credentials regularly
- Use different IAM users for production and development

## Running Training from External Compute

You can run training scripts from any compute platform (Google Colab, AWS SageMaker, Lambda AI, etc.) by setting environment variables:

```python
import os

# Set environment variables
os.environ["MLFLOW_TRACKING_URI"] = "http://your-mlflow-server:5000"
os.environ["AWS_ACCESS_KEY_ID"] = "your-key"
os.environ["AWS_SECRET_ACCESS_KEY"] = "your-secret"
os.environ["AWS_REGION"] = "eu-west-3"
os.environ["S3_BUCKET_MLFLOW"] = "mltrack-mlflow"
os.environ["S3_BUCKET_DATA"] = "mltrack-data"

# Install mltrack
!pip install git+https://github.com/yourusername/mltrack.git

# Run training
from mltrack.config import setup_mlflow
setup_mlflow()

# Your training code here
import mlflow
with mlflow.start_run():
    # Train your model
    pass
```

All runs will be tracked to your MLflow server and visible in the dashboard.

## Production Deployment

For production deployment:

1. **Remove local PostgreSQL override**:
   ```bash
   # Rename or delete docker-compose.override.yml
   mv docker-compose.override.yml docker-compose.override.yml.bak
   ```

2. **Update DATABASE_URL** in `.env` to point to NeonDB or your hosted PostgreSQL:
   ```bash
   DATABASE_URL=postgresql://user:pass@your-neon-host/mltrack
   ```

3. **Deploy to your server**:
   ```bash
   docker compose up -d
   ```

4. **Expose MLflow server** (use nginx reverse proxy, add HTTPS, etc.)

## Usage Patterns

### Modular Pipeline (main.py)

Use the refactored modules (loaders, models, pipelines):

```bash
# Train all available models
uv run python -m mltrack.main

# Train specific models
uv run python -m mltrack.main --models logistic_regression svm_rbf

# Custom experiment
uv run python -m mltrack.main --experiment-name "Experiment v2" --test-size 0.3
```

**Best for**: Experimentation, composable workflows, extensibility

### Programmatic Usage

```python
from mltrack.config import setup_mlflow
from mltrack.pipelines import TrainingPipeline, run_pipeline

# Setup MLflow
setup_mlflow()

# Configure pipeline
pipeline = TrainingPipeline(
    loader_name="iris",
    model_names=["random_forest", "svm_rbf"],
    test_size=0.2,
)

# Run pipeline
results = run_pipeline(pipeline, experiment_name="My Experiment")
```

**Best for**: Integration into larger workflows, production systems

## Development

### Running Tests

```bash
uv sync --extra dev  # Install dev dependencies first
uv run pytest
```

### Linting and Formatting

```bash
uv sync --dev  # Install dev dependencies first
uv run ruff check .
uv run ruff format .
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `uv run pytest`
5. Check linting: `uv run ruff check .`
6. Submit a pull request

## License

MIT
