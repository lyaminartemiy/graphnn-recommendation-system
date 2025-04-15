from typing import List, Tuple

import redis
from fastapi import HTTPException


async def get_recommendations(
    user_id: str, redis_client: redis.Redis
) -> List[Tuple[str, float]]:
    """Возвращает список рекомендаций в формате [(item_id, score), ...]"""
    try:
        user_data = await redis_client.get(f"user:{user_id}")

        if not user_data:
            return []

        recommendations = [
            ("123", 0.5),
            ("123", 0.5),
            ("123", 0.5),
        ]

        return recommendations

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")
