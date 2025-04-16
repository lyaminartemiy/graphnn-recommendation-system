from typing import List

from pydantic import BaseModel


class RecommendationItem(BaseModel):
    item_id: str
    score: float


class RecommendationResponse(BaseModel):
    recommendations: List[RecommendationItem]


class InfrastructureConfig(BaseModel):
    redis_url: str
    mlflow_uri: str
    model_run_id: str


class RecommendationServiceConfig(BaseModel):
    infrastructure: InfrastructureConfig
