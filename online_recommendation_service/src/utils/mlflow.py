import mlflow
import joblib


def load_mlflow_model(run_id: str, model_name: str = "model") -> dict:
    """
    Загружает модель и артефакты из MLflow
    Args:
        run_id: ID запуска в MLflow
        model_name: Название модели в артефактах
    Returns:
        Словарь с загруженными артефактами:
        {
            "model": PyTorch модель,
            "item_encoder": загруженный энкодер
        }
    """
    artifacts = {}
    
    try:
        # 1. Загрузка PyTorch модели
        model_uri = f"runs:/{run_id}/{model_name}"
        print(f"Загрузка модели из MLflow: {model_uri}")
        artifacts["model"] = mlflow.pytorch.load_model(model_uri)
        
        # 2. Загрузка энкодера
        encoder_path = mlflow.artifacts.download_artifacts(
            f"runs:/{run_id}/encoders/item_encoder.pkl"
        )
        
        artifacts["item_encoder"] = joblib.load(encoder_path)
        
        print(f"Тип загруженного энкодера: {type(artifacts['item_encoder'])}")
        print("Модель и артефакты успешно загружены")
        return artifacts
        
    except Exception as e:
        print(f"Ошибка загрузки модели: {str(e)}")
        raise
