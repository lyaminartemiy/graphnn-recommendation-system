from contextlib import asynccontextmanager
from typing import AsyncIterator

import redis.asyncio as redis
import yaml
from fastapi import FastAPI, Request
from src.schemas.schemas import RecommendationServiceConfig


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
    redis_client = redis.from_url(
        "redis://:redis123@redis:6379/0", decode_responses=True  # Укажите ваш URL
    )

    # Сохраняем в state
    app.state.redis_client = redis_client

    # Проверка подключения к БД
    try:
        await redis_client.ping()
        print("Подключение к Redis установлено")
    except Exception as e:
        print(f"Ошибка подключения к Redis: {e}")
        raise

    yield

    # Закрытие соединения
    await redis_client.close()
    print("Подключение к Redis закрыто")


async def get_redis(request: Request) -> redis.Redis:
    """Зависимость для получения Redis клиента"""
    return request.app.state.redis_client
