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
├── src/mltrack/           # Installable Python package (src layout)
│   ├── config.py          # Centralized configuration management
│   └── train.py           # Standalone training script
├── streamlit/             # Separate service, own dependencies
│   ├── Dockerfile         # Isolated container
│   └── app.py            # Dashboard application
├── mlflow/                # Separate service
│   └── Dockerfile        # MLflow server container
├── requirements/          # Split by service for minimal images
│   ├── base.txt          # Common dependencies
│   ├── mlflow.txt        # MLflow server only
│   ├── streamlit.txt     # Dashboard only
│   └── dev.txt           # Development tools
├── tests/                 # Test suite
└── docs/                  # Architecture documentation
```

**Design decisions:**
- **Src layout**: Prevents accidental imports from project root
- **Split requirements**: Minimize Docker image sizes
- **Service isolation**: Each service has independent dependencies
- **Standalone training**: Training script works anywhere with config
