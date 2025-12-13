# MLtrack Architecture Documentation

This document provides detailed architecture diagrams for the MLtrack system.

## Component Diagram

```mermaid
graph TB
    subgraph "User Interface"
        Dashboard[Streamlit Dashboard<br/>Port 8501]
    end

    subgraph "Tracking & Storage"
        MLflow[MLflow Server<br/>Port 5000]
        Postgres[(PostgreSQL<br/>Metadata Store)]
        S3_Artifacts[(S3: mltrack-mlflow<br/>Model Artifacts)]
        S3_Data[(S3: mltrack-data<br/>Training Data)]
    end

    subgraph "Compute Options"
        Local[Local Training<br/>Python Script]
        Colab[Google Colab<br/>Notebook]
        AWS[AWS SageMaker<br/>Training Jobs]
        Lambda[Lambda AI<br/>GPU Instances]
    end

    Dashboard -->|Query Experiments| MLflow
    MLflow -->|Store Metadata| Postgres
    MLflow -->|Store Artifacts| S3_Artifacts

    Local -->|Log Runs| MLflow
    Local -->|Load Data| S3_Data
    Colab -->|Log Runs| MLflow
    Colab -->|Load Data| S3_Data
    AWS -->|Log Runs| MLflow
    AWS -->|Load Data| S3_Data
    Lambda -->|Log Runs| MLflow
    Lambda -->|Load Data| S3_Data
```

## Data Flow Diagram

```mermaid
flowchart LR
    subgraph "Training Phase"
        Dataset[Training Dataset]
        Script[Training Script]
        Model[Trained Model]
    end

    subgraph "MLflow Tracking"
        Params[Parameters]
        Metrics[Metrics]
        Artifacts[Model Artifacts]
    end

    subgraph "Storage Layer"
        PG[(PostgreSQL)]
        S3[(S3 Buckets)]
    end

    subgraph "Visualization"
        UI[Streamlit Dashboard]
    end

    Dataset --> Script
    Script --> Model
    Model --> Artifacts

    Script --> Params
    Script --> Metrics

    Params --> PG
    Metrics --> PG
    Artifacts --> S3

    PG --> UI
    S3 --> UI
```

## Training Run Sequence Diagram

```mermaid
sequenceDiagram
    participant User
    participant Script as Training Script
    participant MLflow as MLflow Server
    participant PG as PostgreSQL
    participant S3 as S3 Bucket

    User->>Script: Execute train.py
    Script->>Script: setup_mlflow()
    Script->>MLflow: Set tracking URI

    Script->>MLflow: start_run()
    MLflow->>PG: Create run entry

    Script->>Script: Train model
    Script->>MLflow: log_params()
    MLflow->>PG: Store parameters

    Script->>MLflow: log_metrics()
    MLflow->>PG: Store metrics

    Script->>MLflow: log_model()
    MLflow->>S3: Upload artifacts
    MLflow->>PG: Store artifact URI

    Script->>MLflow: end_run()
    MLflow->>PG: Mark run complete

    User->>Script: Check results
    Script-->>User: Training complete
```

## External Compute Integration Pattern

```mermaid
graph TB
    subgraph "External Platform"
        Colab[Google Colab Notebook]
        Env[Environment Setup]
        Training[Training Code]
    end

    subgraph "MLtrack Infrastructure"
        MLflow[MLflow Server<br/>Public Endpoint]
        Auth[Authentication Layer<br/>Future]
    end

    subgraph "Persistent Storage"
        Database[(PostgreSQL<br/>Metadata)]
        S3[(S3<br/>Artifacts & Data)]
    end

    Colab --> Env
    Env -->|Set ENV vars| Training
    Training -->|HTTPS| MLflow
    MLflow -->|Store| Database
    MLflow -->|Upload| S3

    style Auth stroke-dasharray: 5 5
    note1[Future: Add API keys<br/>for secure access]

    Auth -.-> MLflow
    note1 -.-> Auth
```

## Deployment Patterns

### Local Development

```mermaid
graph LR
    subgraph "Docker Compose"
        MLflow[MLflow]
        Streamlit[Streamlit]
        Postgres[PostgreSQL]
    end

    subgraph "Local Machine"
        Browser[Web Browser]
        Terminal[Terminal]
    end

    Browser -->|:8501| Streamlit
    Browser -->|:5000| MLflow
    Terminal -->|Train| MLflow
```

### Production Deployment

```mermaid
graph TB
    subgraph "Cloud Infrastructure"
        LB[Load Balancer<br/>HTTPS]
        MLflow[MLflow Server<br/>Container]
        Streamlit[Streamlit Dashboard<br/>Container]
    end

    subgraph "Managed Services"
        NeonDB[(NeonDB<br/>PostgreSQL)]
        S3[(S3 Buckets)]
    end

    subgraph "External Clients"
        Researchers[Researchers<br/>Training Jobs]
        Users[Data Scientists<br/>Web UI]
    end

    Users -->|HTTPS| LB
    LB --> Streamlit
    LB --> MLflow

    Researchers -->|API| MLflow

    MLflow --> NeonDB
    MLflow --> S3
    Streamlit --> MLflow
```

## Security Considerations

### Current State (MVP)

- No authentication on MLflow server
- Basic S3 IAM credentials
- Local development only

### Future Enhancements

```mermaid
graph TB
    Client[External Client]
    Auth[Auth Service<br/>API Keys / OAuth]
    MLflow[MLflow Server]
    DB[(Encrypted DB)]
    S3[(Encrypted S3)]

    Client -->|API Key| Auth
    Auth -->|Valid| MLflow
    MLflow --> DB
    MLflow --> S3

    style Auth fill:#f9f
    style DB fill:#9ff
    style S3 fill:#9ff
```

Planned security features:
- API key authentication for MLflow
- OAuth integration for Streamlit dashboard
- Encrypted database connections
- S3 bucket encryption at rest
- VPC isolation for production
- Audit logging

## Scaling Considerations

### Current Capacity
- Single MLflow server instance
- Single Streamlit instance
- PostgreSQL container (local) or NeonDB (production)

### Future Scaling Options

```mermaid
graph TB
    subgraph "Scaled Infrastructure"
        LB[Load Balancer]
        MLflow1[MLflow Instance 1]
        MLflow2[MLflow Instance 2]
        MLflow3[MLflow Instance N]
        Stream1[Streamlit Instance 1]
        Stream2[Streamlit Instance 2]
    end

    subgraph "Shared Storage"
        PG[(PostgreSQL<br/>Connection Pool)]
        S3[(S3<br/>Unlimited Scale)]
    end

    LB --> MLflow1
    LB --> MLflow2
    LB --> MLflow3
    LB --> Stream1
    LB --> Stream2

    MLflow1 --> PG
    MLflow2 --> PG
    MLflow3 --> PG
    MLflow1 --> S3
    MLflow2 --> S3
    MLflow3 --> S3

    Stream1 --> MLflow1
    Stream2 --> MLflow2
```

## Technology Stack Details

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Experiment Tracking | MLflow | Track parameters, metrics, models |
| Backend Storage | PostgreSQL | Store experiment metadata |
| Artifact Storage | AWS S3 | Store model artifacts and datasets |
| Dashboard | Streamlit | Visualize and compare experiments |
| Containerization | Docker Compose | Local development environment |
| Package Manager | uv | Fast Python dependency management |
| Linting | Ruff | Code quality and formatting |
| Testing | pytest | Unit and integration tests |
| CI/CD | GitHub Actions | Automated testing and validation |

## Directory Structure Rationale

```
mltrack/
├── src/mltrack/                    # Installable Python package (src layout)
│   ├── __init__.py
│   ├── config.py                   # Centralized configuration management
│   ├── logger.py                   # Logging utilities
│   ├── loaders.py                  # Data loading abstraction (DataLoader base, IrisLoader)
│   ├── models.py                   # Model registry (get_model, list_models)
│   ├── pipelines.py                # Pipeline orchestration (TrainingPipeline, run_pipeline)
│   ├── storage.py                  # S3 storage handler (S3Storage, get_storage)
│   ├── train.py                    # Standalone training script (demo)
│   └── main.py                     # Main entry point (if available)
├── streamlit/                      # Separate service, own dependencies
│   ├── app.py                      # Main dashboard application
│   ├── data.py                     # Data retrieval utilities
│   ├── Dockerfile                  # Isolated container
│   └── __init__.py
├── mlflow/                         # Separate service
│   ├── Dockerfile                  # MLflow server container
│   └── entrypoint.sh               # Container entrypoint
├── tests/                          # Test suite
│   └── test_imports.py             # Package import tests
├── docs/                           # Architecture documentation
│   └── architecture.md             # Detailed diagrams and design rationale
├── pyproject.toml                  # Project configuration (uv)
└── docker-compose.yml              # Local development stack
```

### Module Details

#### Core Modules

**`loaders.py`** - Data loading abstraction
- `DataLoader` (ABC): Base class for all loaders
- `IrisLoader`: Built-in Iris dataset loader
- `get_loader(name)`: Factory function for loader instances

**`models.py`** - Model registry
- `MODELS`: Registry dict with model classes and default parameters
- `get_model(name, **kwargs)`: Get configured model instance
- `list_models()`: List available models

**`pipelines.py`** - Pipeline orchestration
- `TrainingPipeline`: Configuration dataclass
- `train_single_model()`: Train one model with MLflow logging
- `run_pipeline()`: Execute complete training pipeline

**`storage.py`** - S3 storage handler
- `S3Storage`: Handler for save/load operations
- `save_preprocessed()`, `load_preprocessed()`: Data serialization
- `save_model()`, `load_model()`: Model persistence
- `get_storage()`: Factory function

#### Utility Modules

**`config.py`** - Configuration management
- `MLtrackConfig`: Dataclass for environment variables
- `setup_mlflow()`: Initialize MLflow server connection

**`logger.py`** - Logging utilities
- `get_logger(name)`: Get configured logger (respects ENV variable)

#### Entry Points

**`train.py`** - Standalone demo
- Direct model training with MLflow autolog
- Iris dataset + three classifiers (LogisticRegression, RandomForest, SVM)
- Works offline if needed

**`main.py`** - Modular entry point (optional)
- Uses pipelines module for orchestrated training
- Demonstrates composable architecture

**Design decisions:**
- **Src layout**: Prevents accidental imports from project root
- **Module separation**: Each concern (loading, modeling, pipeline) isolated
- **Registry pattern**: Flexible model/loader registration without hardcoding
- **Abstract base classes**: Extensible architecture for custom loaders/models
- **Standalone compatibility**: train.py works without external setup
- **MLflow integration**: Autolog for automatic parameter/metric tracking

## Extending the Architecture

### Adding a Custom Data Loader

Create a new loader by implementing the `DataLoader` interface:

```python
# In loaders.py
class CustomDataLoader(DataLoader):
    """Loader for custom dataset."""

    def load(self) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
        """Load your data."""
        # Load X, y, feature_names, target_names
        return X, y, feature_names, target_names

# Register in loaders.py get_loader()
loaders = {
    "iris": IrisLoader,
    "custom": CustomDataLoader,  # Add here
}
```

Usage:
```bash
uv run python -m mltrack.main --loader custom
```

### Adding a Custom Model

Register new sklearn models in the models registry:

```python
# In models.py
MODELS: dict[str, tuple[type[ClassifierMixin], dict[str, Any]]] = {
    "logistic_regression": (...),
    "random_forest": (...),
    "custom_model": (
        CustomClassifier,
        {"param1": value1, "param2": value2}
    ),
}
```

Usage:
```bash
uv run python -m mltrack.main --models custom_model logistic_regression
```

### Integration Points

**No code changes needed for:**
- Adding MLflow tracking parameters → autolog handles it
- Changing hyperparameters → use `--models custom_model --extra-params` (if implemented)
- Different S3 buckets → environment variables in `.env`
- Custom logging levels → `ENV=local` for debug logging

**Requires code changes for:**
- New data sources → New DataLoader implementation
- New model types → Register in MODELS dict
- Custom pipeline logic → Extend TrainingPipeline class or create new pipeline function
