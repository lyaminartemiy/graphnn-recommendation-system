import torch

import mlflow


def load_mlflow_model(run_id: str, model_name: str = "model") -> torch.nn.Module:
    """
    Загружает модель и артефакты из MLflow
    Args:
        run_id: ID запуска в MLflow
        model_name: Название модели в артефактах
    Returns:
        Загруженная модель PyTorch
    """
    try:
        # Формируем URI модели
        model_uri = f"runs:/{run_id}/{model_name}"

        print(f"Загрузка модели из MLflow: {model_uri}")
        model = mlflow.pytorch.load_model(model_uri)
        item_encoder = mlflow.artifacts.download_artifacts(
            f"runs:/{run_id}/encoders/item_encoder.pkl"
        )

        artifacts = {
            "model": model,
            "item_encoder": item_encoder,
        }
        print("Модель успешно загружена")
        return artifacts
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        raise
