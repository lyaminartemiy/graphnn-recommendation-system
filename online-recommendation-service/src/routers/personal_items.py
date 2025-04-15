from typing import Dict, List

import redis
from fastapi import APIRouter, Depends
from src.lifespan import get_redis
from src.schemas.schemas import RecommendationResponse
from src.services.personal_items import get_recommendations

router = APIRouter(prefix="/recommendations")


@router.get("/personal_items", response_model=RecommendationResponse)
async def get_personal_recommendations(
    user_id: str,
    redis_client: redis.Redis = Depends(get_redis),
) -> Dict[str, List[Dict[str, float]]]:
    """
    Возвращает персонализированные рекомендации для пользователя.

    Формат ответа:
    {
        "recommendations": [
            {"item_id": "123", "score": 0.95},
            {"item_id": "456", "score": 0.82}
        ]
    }
    """
    recommendations = await get_recommendations(
        user_id=user_id,
        redis_client=redis_client,
    )

    return {
        "recommendations": [
            {"item_id": item_id, "score": score} for item_id, score in recommendations
        ]
    }
