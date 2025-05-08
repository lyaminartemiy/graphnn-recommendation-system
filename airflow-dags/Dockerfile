FROM apache/airflow:2.8.1-python3.11

# Установка системных зависимостей
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        python3-dev \
        libopenblas-dev \
        gfortran \
        libopenmpi-dev && \
    rm -rf /var/lib/apt/lists/*

# Установка Python-зависимостей
USER airflow
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    requests==2.31.0 \
    redis==5.2.1 \
    PyYAML==6.0.2 \
    torch==2.3.0 \
    torch-geometric==2.5.3 \
    pandas==2.2.3 \
    tqdm==4.67.1 \
    numpy==2.0.0 \
    scikit-learn==1.6.1 \
    boto3==1.37.34 \
    mlflow==2.21.3 \
    fastparquet==2024.11.0 \
    pyarrow==19.0.1 \
    psycopg2-binary==2.9.10 \
    marshmallow-sqlalchemy==0.28.2

# Копируем весь проект
COPY ./src /opt/airflow/src
COPY ./dags /opt/airflow/dags

# Устанавливаем PYTHONPATH
ENV PYTHONPATH=/opt/airflow:/opt/airflow/src:${PYTHONPATH}
