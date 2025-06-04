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


def get_lightgcn_recommendations(
    model,
    user_id: str,
    count: int = 10,
    return_scores: bool = True
) -> List[tuple]:
    """Получение рекомендаций от LightGCN модели"""
    try:
        # Получаем рекомендации из модели
        recommendations = model.predict(
            context={},
            model_input={"user_ids": [user_id]},
            params={
                "top_k": count,
                "return_scores": return_scores,
            },
        )
        
        # Форматируем результат
        if return_scores and recommendations and isinstance(recommendations[0], (list, tuple)):
            return [(item_id, float(score)) for item_id, score in recommendations]
        elif recommendations:
            return [(item_id, 0.0) for item_id in recommendations]
        
        return []
    except Exception as e:
        print(f"Ошибка при получении рекомендаций: {e}")
        return []


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
        # Если истории нет - возвращаем популярные товары
        popular_items = await redis_client.get("popular_recommendations")
        if popular_items:
            return {
                "recommendations": [
                    {"item_id": item_id, "score": 1.0}  # У популярных score=1
                    for item_id in json.loads(popular_items)[:10]  # Берем топ-10
                ]
            }
        return {"recommendations": []}
    
    # Получаем рекомендации от модели
    recommendations = get_lightgcn_recommendations(
        model=model,
        user_id=user_id,
        count=20  # Берем больше для последующей фильтрации
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
