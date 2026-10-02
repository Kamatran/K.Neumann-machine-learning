# The official image lacks the S3 client (boto3) needed to store artifacts in MinIO,
# and the auth and PostgreSQL extras used by the shared-server variant.
FROM ghcr.io/mlflow/mlflow:v3.16.0
RUN pip install --no-cache-dir "mlflow[auth]==3.16.0" boto3==1.43.93 psycopg2-binary==2.9.13
