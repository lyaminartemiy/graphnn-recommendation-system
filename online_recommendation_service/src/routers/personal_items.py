from typing import Dict, List

import redis
from fastapi import APIRouter, Depends
from src.lifespan import get_graph_nn_recommender, get_redis
from src.modules.graph_nn.model import GNNRecommender
from src.schemas.schemas import RecommendationResponse
from src.services.personal_items import get_recommendations

router = APIRouter(prefix="/recommendations")


@router.get("/personal_items/", response_model=RecommendationResponse)
async def get_personal_recommendations(
    user_id: str,
    redis_client: redis.Redis = Depends(get_redis),
    model: GNNRecommender = Depends(get_graph_nn_recommender),
) -> Dict[str, List[Dict[str, float]]]:
    """
    Возвращает персонализированные рекомендации для пользователя.
    """

    return await get_recommendations(
        user_id=user_id,
        redis_client=redis_client,
        model=model,
    )
