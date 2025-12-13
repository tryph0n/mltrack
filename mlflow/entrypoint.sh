#!/bin/sh
set -e

exec mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri "${DATABASE_URL}" \
    --default-artifact-root "s3://${S3_BUCKET_MLFLOW}/artifacts" \
    --allowed-hosts "${MLFLOW_SERVER_ALLOWED_HOSTS}"
