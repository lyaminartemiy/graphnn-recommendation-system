from typing import List, Dict
import json
import redis


def get_recommendations_from_graph_nn(
    model,
    input_data: dict,
    count: int,
    return_scores=True,
):
    # Ensure input_data is properly formatted
    if isinstance(input_data, str):
        try:
            input_data = json.loads(input_data)
        except json.JSONDecodeError:
            input_data = {"items": []}
    
    # Получаем рекомендации из модели
    recommendations_result = model.predict(
        context={},
        model_input=input_data,
        params={
            "top_k": count,
            "return_scores": return_scores,
        },
    )
    print("recommendations_result:", recommendations_result)
    
    # Ensure we always return a list of tuples
    if not recommendations_result:
        return []
    if isinstance(recommendations_result[0], str):  # Only item IDs returned
        return [(item, 0.0) for item in recommendations_result]
    
    return recommendations_result


async def get_recommendations(
    user_id: str, 
    redis_client: redis.Redis, 
    model,
) -> Dict[str, List[Dict[str, float]]]:
    """
    Возвращает рекомендации с учетом дизлайков и истории
    """
    # Получаем дизлайки пользователя
    dislikes = await redis_client.smembers(f"user_dislikes:{user_id}")
    dislikes = set(dislikes) if dislikes else set()
    
    # Получаем историю пользователя
    user_data = await redis_client.get(f"user:{user_id}")
    
    if not user_data:
        # Холодный пользователь
        return {"recommendations": []}
    
    recommendations = get_recommendations_from_graph_nn(
        model=model,
        input_data=user_data,
        count=20,  # Берем больше, чтобы отфильтровать
        return_scores=True,
    )
    
    # Фильтруем рекомендации
    filtered_recommendations = []
    for item_id, score in recommendations:
        str_item_id = str(item_id)
        if str_item_id not in dislikes:
            filtered_recommendations.append((item_id, score))
    
    return {
        "recommendations": [
            {"item_id": str(item_id), "score": float(score)}
            for item_id, score in filtered_recommendations[:10]  # Возвращаем топ-10
        ]
    }
