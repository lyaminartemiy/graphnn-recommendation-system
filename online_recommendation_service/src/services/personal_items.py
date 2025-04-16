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
            return {
                "recommendations": []
            }

        recommendations = get_recommendations_from_graph_nn(
            model=model,
            input_data=user_data,
            count=10,
            return_scores=True,
        )
        print("recommendations:", recommendations)

        response =  {
            "recommendations": [
                {"item_id": str(item_id), "score": float(score)}
                for item_id, score in recommendations
            ]
        }
        print("response:", response)
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")
