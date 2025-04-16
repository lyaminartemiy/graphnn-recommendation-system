from typing import List, Tuple

import redis
import torch
from fastapi import HTTPException
from src.modules.graph_nn.inference import get_recommendations_from_graph_nn


async def get_recommendations(
    user_id: str, redis_client: redis.Redis, model: torch.nn.Module
) -> List[Tuple[str, float]]:
    """
    Возвращает список рекомендаций в формате [(item_id, score), ...]
    """

    try:
        user_data = await redis_client.get(f"user:{user_id}")
        print(user_data)

        if not user_data:
            return []

        recommendations = get_recommendations_from_graph_nn()

        return {
            "recommendations": [
                {"item_id": item_id, "score": score}
                for item_id, score in recommendations
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")
