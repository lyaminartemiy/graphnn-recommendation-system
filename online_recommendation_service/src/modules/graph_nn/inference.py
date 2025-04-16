from src.modules.graph_nn.model import GNNRecommender


async def get_recommendations_from_graph_nn(
    model: GNNRecommender,
    input_data: dict,
    count: int,
    return_scores=True,
):
    return model.recommend(
        items=input_data,
        top_k=count,
        return_scores=return_scores,
    )
