from typing import Dict, List

import ast
import redis
from fastapi import APIRouter, Depends
from src.lifespan import get_redis_client
from src.schemas.schemas import EventsResponse

router = APIRouter(prefix="/recommendations")


@router.get("/events/", response_model=EventsResponse)
async def get_events(
    user_id: str,
    redis_client: redis.Redis = Depends(get_redis_client),
) -> Dict[str, List[Dict[str, float]]]:
    """
    Возвращает историю транзакций пользователя.
    """

    events = await redis_client.get(f"user:{user_id}")
    if not events:
        return {
            "events": []
        }
    events = ast.literal_eval(events)
    print(events)
    return {"events": [str(event) for event in events]}
