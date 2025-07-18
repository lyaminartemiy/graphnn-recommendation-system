from contextlib import asynccontextmanager
from typing import AsyncIterator

import redis.asyncio as redis
import boto3
from botocore.client import Config
import yaml
from fastapi import FastAPI, Request
from src.schemas.schemas import RecommendationServiceConfig
from mlflow import MlflowClient

import mlflow


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[dict]:
    """
    Lifespan для управления ресурсами приложения
    """
    # Загрузка конфигурации сервиса
    with open("./config/config.yaml", "r") as f:
        config_data = yaml.safe_load(f)

    # Валидация Pydantic модели
    app.state.service_config = RecommendationServiceConfig.model_validate(config_data)

    # Инициализация асинхронного клиента Redis
    redis_url = app.state.service_config.infrastructure.redis_url
    redis_client = redis.from_url(url=redis_url, decode_responses=True)
    app.state.redis_client = redis_client

    # Инициализация клиента S3
    app.state.s3_client = boto3.client(
        "s3",
        endpoint_url=app.state.service_config.infrastructure.s3_endpoint_url,
        aws_access_key_id="minio",
        aws_secret_access_key="minio123",
        config=Config(signature_version="s3v4"),
    )

    # Проверка подключения к БД
    try:
        await redis_client.ping()
        print("Подключение к Redis установлено")
    except Exception as e:
        print(f"Ошибка подключения к Redis: {e}")
        raise

    # Настройка MLflow и загрузка модели
    try:
        mlflow.set_tracking_uri(app.state.service_config.infrastructure.mlflow_uri)
        
        client = MlflowClient()
        latest_version = client.get_latest_versions("lightgcn_recommender")[0]
        print("latest_version:", latest_version)
        model_uri = f"models:/lightgcn_recommender/{latest_version.version}"
        
        # Загружаем модель
        pyfunc_model = mlflow.pyfunc.load_model(model_uri)
        app.state.graph_nn_recommender = pyfunc_model.unwrap_python_model()
        
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        raise

    yield

    # Закрытие соединения
    await redis_client.close()


async def get_redis_client(request: Request) -> redis.Redis:
    """
    Зависимость для получения Redis клиента
    """

    return request.app.state.redis_client


async def get_s3_client(request: Request) -> redis.Redis:
    """
    Зависимость для получения клиента S3
    """

    return request.app.state.s3_client


async def get_graph_nn_recommender(request: Request):
    """
    Зависимость для получения модели
    """

    return request.app.state.graph_nn_recommender
