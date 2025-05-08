from datetime import datetime
from airflow.decorators import dag, task
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
import mlflow
import joblib
import os
import pickle
from pathlib import Path
import redis
import json
from typing import List, Dict


# Константы для путей сохранения
DATA_DIR = Path("/opt/airflow/data/tmp_graph_nn")
DATA_DIR.mkdir(parents=True, exist_ok=True)

@dag(
    dag_id="graph_nn_training_pipeline",
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["ml", "graph_nn"],
)
def graph_nn_pipeline():
    
    class Constants:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        BATCH_SIZE = 32
        HIDDEN_DIM = 256
        NUM_LAYERS = 5
        NUM_HEADS = 4
        SHUFFLE = False
        TRAIN_SPLIT = 0.8
        TRANSACTIONS_TABLE = "recsys.transactions"
        DB_HOST = "postgres-data"
        DB_PORT = 5432
        DB_NAME = "appdata"
        DB_USER = "appdata"
        DB_PASSWORD = "appdata123"
        USER_ID = "user_id"
        ITEMS_IDS = "items_ids"
        REDIS_HOST = "redis"
        REDIS_PORT = 6379
        REDIS_PASSWORD = "redis123"

    def _save_obj(obj, path):
        """Сохранение объекта на диск"""
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    def _load_obj(path):
        """Загрузка объекта с диска"""
        with open(path, 'rb') as f:
            return pickle.load(f)

    def load_data_to_redis(user_sequences: List[Dict], r: redis.Redis) -> None:
        """Загрузка данных в Redis"""
        for user_sequence in user_sequences:
            user_id = user_sequence.get(Constants.USER_ID)
            items_ids = user_sequence.get(Constants.ITEMS_IDS)

            r.set(f"user:{user_id}", json.dumps(items_ids))

        print("Данные успешно загружены в Redis")

    @task
    def prepare_data():
        """Загрузка и подготовка данных из PostgreSQL"""
        from src.modules.graph_nn.dataset import TransactionDataset
        
        try:
            import psycopg2
            with psycopg2.connect(
                host=Constants.DB_HOST,
                port=Constants.DB_PORT,
                dbname=Constants.DB_NAME,
                user=Constants.DB_USER,
                password=Constants.DB_PASSWORD
            ) as conn:
                query = f"""
                    SELECT user_id, item_id, time, price, sales_channel_id 
                    FROM {Constants.TRANSACTIONS_TABLE} 
                    ORDER BY time, user_id
                """
                transactions = pd.read_sql(query, conn)
            
            # Сохраняем сырые транзакции для Redis
            raw_data_path = DATA_DIR / "raw_transactions.pkl"
            _save_obj(transactions, raw_data_path)
            
            graph_dataset = TransactionDataset(transactions)
            
            # Сохраняем датасет на диск
            dataset_path = DATA_DIR / "graph_dataset.pkl"
            _save_obj(graph_dataset, dataset_path)
            
            # Разделение данных
            graph_train_size = int(Constants.TRAIN_SPLIT * len(graph_dataset))
            train_indices = list(range(graph_train_size))
            test_indices = list(range(graph_train_size, len(graph_dataset)))
            
            return {
                "dataset_path": str(dataset_path),
                "raw_data_path": str(raw_data_path),
                "train_indices": train_indices,
                "test_indices": test_indices,
                "num_items": graph_dataset.num_items
            }
            
        except Exception as e:
            print(f"Ошибка при работе с PostgreSQL: {str(e)}")
            raise

    @task
    def load_to_redis(data_dict):
        """Загрузка данных пользователей в Redis"""
        # Загружаем сырые транзакции
        transactions = _load_obj(data_dict["raw_data_path"])
        
        # Подготовка данных для Redis
        user_sequences = transactions.groupby('user_id')['item_id'].apply(list).reset_index()
        user_sequences = user_sequences.rename(columns={
            'user_id': Constants.USER_ID,
            'item_id': Constants.ITEMS_IDS
        }).to_dict('records')
        
        # Подключение к Redis
        r = redis.Redis(
            host=Constants.REDIS_HOST,
            password=Constants.REDIS_PASSWORD,
            port=Constants.REDIS_PORT,
            decode_responses=True,
        )
        
        # Загрузка данных
        load_data_to_redis(user_sequences, r)
        
        return "Данные успешно загружены в Redis"

    @task
    def create_data_loaders(data_dict):
        """Создание DataLoader'ов"""
        # Загружаем датасет с диска
        graph_dataset = _load_obj(data_dict["dataset_path"])
        
        # Создаем подмножества
        graph_train_dataset = torch.utils.data.Subset(
            graph_dataset, 
            data_dict["train_indices"]
        )
        graph_test_dataset = torch.utils.data.Subset(
            graph_dataset,
            data_dict["test_indices"]
        )
        
        # Создаем загрузчики
        train_loader = DataLoader(
            graph_train_dataset, 
            batch_size=Constants.BATCH_SIZE, 
            shuffle=Constants.SHUFFLE
        )
        test_loader = DataLoader(
            graph_test_dataset, 
            batch_size=Constants.BATCH_SIZE, 
            shuffle=Constants.SHUFFLE
        )
        
        # Сохраняем загрузчики на диск
        loaders_path = DATA_DIR / "data_loaders.pkl"
        _save_obj({
            "train_loader": train_loader,
            "test_loader": test_loader,
            "num_items": data_dict["num_items"],
            "dataset_path": data_dict["dataset_path"]
        }, loaders_path)
        
        return {
            "loaders_path": str(loaders_path)
        }

    @task
    def train_model(data_dict):
        """Обучение модели"""
        from src.modules.graph_nn.model import TransactionGNN
        from src.modules.graph_nn.train import train_epoch, evaluate_epoch
        
        # Загружаем данные
        loaders_data = _load_obj(data_dict["loaders_path"])
        
        # Инициализируем модель
        graph_model = TransactionGNN(
            num_items=loaders_data["num_items"],
            hidden_dim=Constants.HIDDEN_DIM,
            num_layers=Constants.NUM_LAYERS,
            num_heads=Constants.NUM_HEADS,
        ).to(Constants.DEVICE)
        graph_optimizer = torch.optim.Adam(graph_model.parameters(), lr=0.001)

        # Обучение и оценка
        for epoch in range(5):
            train_loss = train_epoch(
                model=graph_model,
                loader=loaders_data["train_loader"],
                optimizer=graph_optimizer,
                device=Constants.DEVICE,
            )
            test_acc, test_loss = evaluate_epoch(
                model=graph_model,
                loader=loaders_data["test_loader"],
                device=Constants.DEVICE,
            )
            print(
                f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Test Acc: {test_acc:.4f}, Test Loss: {test_loss:.4f}"
            )
        
        # Сохраняем модель и датасет
        model_path = DATA_DIR / "trained_model.pkl"
        _save_obj({
            "model": graph_model,
            "dataset_path": loaders_data["dataset_path"]
        }, model_path)
        
        return {
            "model_path": str(model_path)
        }

    @task
    def save_to_mlflow(data_dict):
        """Сохранение модели в MLflow"""

        from src.mlflow_models.graph_nn.model import GNNRecommenderWrapper

        # Загружаем модель и датасет
        model_data = _load_obj(data_dict["model_path"])
        graph_dataset = _load_obj(model_data["dataset_path"])
        
        # Настройка окружения MLflow
        os.environ["AWS_ACCESS_KEY_ID"] = "minio"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"

        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("graph-nn-model")

        # Создаем временную директорию для артефактов
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)
        
        # Сохраняем необходимые артефакты
        item_encoder_path = artifacts_dir / "item_encoder.pkl"
        joblib.dump(graph_dataset.item_encoder, item_encoder_path)
                
        # Сохраняем состояние модели
        model_state_path = artifacts_dir / "model_state.pt"
        torch.save({
            "state_dict": model_data["model"].state_dict(),
            "hidden_dim": Constants.HIDDEN_DIM,
            "num_layers": Constants.NUM_LAYERS,
            "num_heads": Constants.NUM_HEADS,
            "dropout": getattr(Constants, "DROPOUT", 0.1),
            "max_seq_length": getattr(Constants, "MAX_SEQ_LENGTH", 50)
        }, model_state_path)

        with mlflow.start_run(run_name=f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Логируем параметры модели
            mlflow.log_params({
                "num_items": graph_dataset.num_items,
                "hidden_dim": Constants.HIDDEN_DIM,
                "num_layers": Constants.NUM_LAYERS,
                "num_heads": Constants.NUM_HEADS,
                "dropout": getattr(Constants, "DROPOUT", 0.1),
                "max_seq_length": getattr(Constants, "MAX_SEQ_LENGTH", 50)
            })

            # Логируем артефакты
            mlflow.log_artifact(item_encoder_path, "artifacts")
            mlflow.log_artifact(model_state_path, "artifacts")

            # Создаем словарь с путями к артефактам для PythonModel
            artifacts = {
                "item_encoder": str(item_encoder_path),
                "model_state": str(model_state_path)
            }

            # Логируем pyfunc модель
            model_info = mlflow.pyfunc.log_model(
                artifact_path="gnn_recommender",
                python_model=GNNRecommenderWrapper(),
                artifacts=artifacts,
                registered_model_name="gnn_recommender",
                code_path=["/opt/airflow/src/mlflow_models"],
                pip_requirements=[
                    "PyYAML==6.0.2",
                    "torch==2.3.0",
                    "torch-geometric==2.5.3",
                    "pandas==2.2.3",
                    "tqdm==4.67.1",
                    "numpy==2.0.0",
                    "scikit-learn==1.6.1",
                    "boto3==1.37.34",
                    "mlflow==2.21.3",
                    "fastparquet==2024.11.0",
                    "pyarrow==19.0.1",
                    "psycopg2-binary==2.9.10",
                    "marshmallow-sqlalchemy==0.28.2",
                ],
            )
        
        # Очистка временных файлов
        for file in artifacts_dir.glob("*"):
            file.unlink()
        artifacts_dir.rmdir()
        
        return f"Model saved to MLflow with run_id: {model_info.run_id}"
    
    # Определяем порядок выполнения задач
    data = prepare_data()
    redis_task = load_to_redis(data)  # Загрузка в Redis
    loaders = create_data_loaders(data)
    model = train_model(loaders)
    save_to_mlflow(model)

    # Зависимости между задачами
    data >> redis_task  # Сначала загрузка данных, потом в Redis
    data >> loaders     # Параллельно с загрузкой в Redis


graph_nn_pipeline_dag = graph_nn_pipeline()
