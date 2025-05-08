FROM ghcr.io/mlflow/mlflow:v2.21.3

# Устанавливаем зависимости для PostgreSQL
RUN pip install --no-cache-dir psycopg2-binary boto3

# Дополнительные инструменты для отладки (можно удалить в production)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
