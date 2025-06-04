from datetime import datetime
from airflow.decorators import dag, task
import pandas as pd
import torch
import mlflow
import joblib
import os
import pickle
from pathlib import Path
import redis
import json
from typing import List, Dict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch_geometric.data import Data

# Константы для путей сохранения
DATA_DIR = Path("/opt/airflow/data/tmp_graph_nn")
DATA_DIR.mkdir(parents=True, exist_ok=True)

@dag(
    dag_id="enhanced_lightgcn_training_pipeline",
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["ml", "graph_nn"],
)
def enhanced_lightgcn_pipeline():
    
    class Constants:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        BATCH_SIZE = 512
        HIDDEN_DIM = 256
        NUM_LAYERS = 10
        NUM_HEADS = 4
        SHUFFLE = False
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
        MAX_SEQ_LENGTH = 50
        DROPOUT = 0.1

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
            items_ids = list(dict.fromkeys(user_sequence.get(Constants.ITEMS_IDS)))  # Удаление дубликатов

            r.set(f"user:{user_id}", json.dumps(items_ids))

        print("Данные успешно загружены в Redis")

    @task
    def prepare_data():
        """Загрузка и подготовка данных из PostgreSQL"""
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
            
            # Создаем энкодеры для пользователей и товаров
            user_encoder = LabelEncoder()
            item_encoder = LabelEncoder()
            
            transactions['user_idx'] = user_encoder.fit_transform(transactions['user_id'])
            transactions['item_idx'] = item_encoder.fit_transform(transactions['item_id'])
            
            # Создаем фиктивные признаки товаров (в реальной системе нужно заменить на реальные)
            item_features = pd.get_dummies(transactions['item_id'].astype(str)).T
            item_features = item_features.reset_index().rename(columns={'index': 'item_id'})
            
            # Нормализация признаков
            scaler = StandardScaler()
            item_feature_values = scaler.fit_transform(item_features.iloc[:, 1:].values)
            item_features_normalized = pd.DataFrame(
                item_feature_values, 
                index=item_features['item_id'],
                columns=item_features.columns[1:]
            )
            
            # Сохраняем подготовленные данные
            data_dict = {
                "transactions": transactions,
                "user_encoder": user_encoder,
                "item_encoder": item_encoder,
                "item_features": item_features_normalized,
                "raw_data_path": str(raw_data_path),
                "num_users": len(user_encoder.classes_),
                "num_items": len(item_encoder.classes_)
            }
            
            dataset_path = DATA_DIR / "prepared_data.pkl"
            _save_obj(data_dict, dataset_path)
            
            return {
                "dataset_path": str(dataset_path),
                "raw_data_path": str(raw_data_path),
                "num_users": len(user_encoder.classes_),
                "num_items": len(item_encoder.classes_)
            }
            
        except Exception as e:
            print(f"Ошибка при подготовке данных: {str(e)}")
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
    def create_graph_data(data_dict):
        """Создание графовых данных для обучения"""
        data = _load_obj(data_dict["dataset_path"])
        transactions = data["transactions"]
        item_features = data["item_features"]
        
        # Разделение на train/test с временным учетом
        train_df = transactions.iloc[:int(len(transactions)*0.6)]
        test_df = transactions.iloc[int(len(transactions)*0.6):]
        
        # Создание взвешенного графа
        def create_weighted_graph_edges(df, user_map, item_map):
            freq = df.groupby(['user_idx', 'item_idx']).size().reset_index(name='freq')
            last_time = df.groupby(['user_idx', 'item_idx'])['time'].max().reset_index()
            
            merged = pd.merge(freq, last_time, on=['user_idx', 'item_idx'])
            merged['weight'] = merged['freq'] * (1 + pd.to_datetime(merged['time']).dt.year - 2018)
            
            src = [user_map[u] for u in merged['user_idx']]
            dst = [item_map[i] for i in merged['item_idx']]
            weights = torch.tensor(merged['weight'].values, dtype=torch.float)
            
            return torch.tensor([src, dst], dtype=torch.long), weights
        
        train_edge_index, train_weights = create_weighted_graph_edges(
            train_df, 
            {u: i for i, u in enumerate(transactions['user_idx'].unique())}, 
            {i: i for i in transactions['item_idx'].unique()}
        )
        
        test_edge_index, test_weights = create_weighted_graph_edges(
            test_df,
            {u: i for i, u in enumerate(transactions['user_idx'].unique())}, 
            {i: i for i in transactions['item_idx'].unique()}
        )
        
        # Создание PyG Data объекта с весами
        num_users = data_dict["num_users"]
        num_items = data_dict["num_items"]
        
        data_pyg = Data(
            edge_index=torch.cat([train_edge_index, test_edge_index], dim=1),
            edge_attr=torch.cat([train_weights, test_weights]),
            num_nodes=num_users + num_items,
            train_mask=torch.cat([
                torch.ones(train_edge_index.shape[1], dtype=torch.bool),
                torch.zeros(test_edge_index.shape[1], dtype=torch.bool)
            ]),
            test_mask=torch.cat([
                torch.zeros(train_edge_index.shape[1], dtype=torch.bool),
                torch.ones(test_edge_index.shape[1], dtype=torch.bool)
            ])
        )
        
        # Подготовка тензора дополнительных признаков
        item_feature_tensor = torch.FloatTensor(
            item_features.loc[transactions['item_id'].unique()].values
        ).to(Constants.DEVICE)
        
        # Сохраняем данные
        graph_data_path = DATA_DIR / "graph_data.pkl"
        _save_obj({
            "graph_data": data_pyg,
            "item_feature_tensor": item_feature_tensor,
            "num_users": num_users,
            "num_items": num_items,
            "user_encoder": data["user_encoder"],
            "item_encoder": data["item_encoder"]
        }, graph_data_path)
        
        return {
            "graph_data_path": str(graph_data_path)
        }

    @task
    def train_model(data_dict):
        """Обучение модели EnhancedLightGCN"""
        from torch_geometric.nn import LGConv
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.optim import Adam
        
        # Загружаем данные
        data = _load_obj(data_dict["graph_data_path"])
        graph_data = data["graph_data"]
        item_feature_tensor = data["item_feature_tensor"]
        
        # Определяем модель EnhancedLightGCN
        class EnhancedLightGCN(nn.Module):
            def __init__(self, num_users, num_items, embedding_dim=128, num_layers=3, 
                         dropout=0.1, item_features=None):
                super().__init__()
                self.num_users = num_users
                self.num_items = num_items
                self.dropout = dropout
                
                self.user_emb = nn.Embedding(num_users, embedding_dim)
                self.item_emb = nn.Embedding(num_items, embedding_dim)
                nn.init.xavier_uniform_(self.user_emb.weight)
                nn.init.xavier_uniform_(self.item_emb.weight)
                
                if item_features is not None:
                    self.item_feature_proj = nn.Linear(item_features.shape[1], embedding_dim)
                    self.item_features = item_features
                else:
                    self.item_feature_proj = None
                
                self.convs = nn.ModuleList([LGConv() for _ in range(num_layers)])
                self.dropout = nn.Dropout(p=dropout)
                self.attention = nn.Parameter(torch.ones(num_layers + 1, 1, 1))

            def forward(self, edge_index):
                user_emb = self.dropout(self.user_emb.weight)
                item_emb = self.dropout(self.item_emb.weight)
                
                if self.item_feature_proj is not None:
                    item_emb = item_emb + self.item_feature_proj(self.item_features)
                
                all_emb = torch.cat([user_emb, item_emb])
                
                embs = [all_emb]
                for conv in self.convs:
                    new_emb = conv(all_emb, edge_index)
                    new_emb = self.dropout(new_emb)
                    new_emb = new_emb + embs[-1] if len(embs) > 0 else new_emb
                    embs.append(new_emb)
                
                stacked_embs = torch.stack(embs, dim=0)
                attention_weights = torch.softmax(self.attention, dim=0)
                final_emb = (stacked_embs * attention_weights).sum(dim=0)
                
                return final_emb[:self.num_users], final_emb[self.num_users:]
        
        # Функция потерь
        def enhanced_loss(user_emb, item_emb, edge_index, weights, num_items, device, lambda_reg=0.01):
            src, dst = edge_index
            
            neg_items = torch.randint(0, num_items, (len(dst),), device=device)
            
            pos_scores = (user_emb[src] * item_emb[dst]).sum(dim=1)
            neg_scores = (user_emb[src] * item_emb[neg_items]).sum(dim=1)
            bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)) * weights.to(device)
            bpr_loss = bpr_loss.mean()
            
            user_norm = F.normalize(user_emb[src], p=2, dim=1)
            item_pos_norm = F.normalize(item_emb[dst], p=2, dim=1)
            item_neg_norm = F.normalize(item_emb[neg_items], p=2, dim=1)
            
            contrast_loss = -torch.log(
                torch.exp((user_norm * item_pos_norm).sum(1) / 0.1) / (
                torch.exp((user_norm * item_pos_norm).sum(1) / 0.1) + 
                torch.exp((user_norm * item_neg_norm).sum(1) / 0.1)
            )).mean()
            
            l2_reg = lambda_reg * (user_emb.norm(2).pow(2) + item_emb.norm(2).pow(2))
            
            return bpr_loss + 0.5 * contrast_loss + l2_reg
        
        # Инициализация модели
        model = EnhancedLightGCN(
            num_users=data["num_users"],
            num_items=data["num_items"],
            embedding_dim=Constants.HIDDEN_DIM,
            num_layers=Constants.NUM_LAYERS,
            dropout=Constants.DROPOUT,
            item_features=item_feature_tensor
        ).to(Constants.DEVICE)
        
        optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        
        # Подготовка данных
        train_edge_index = graph_data.edge_index[:, graph_data.train_mask].to(Constants.DEVICE)
        train_weights = graph_data.edge_attr[graph_data.train_mask].to(Constants.DEVICE)
        
        # Обучение модели
        best_loss = float('inf')
        patience = 30
        early_stop_counter = 0
        
        for epoch in range(500):
            model.train()
            optimizer.zero_grad()
            
            user_emb, item_emb = model(train_edge_index)
            loss = enhanced_loss(
                user_emb, item_emb, 
                train_edge_index, 
                train_weights,
                data["num_items"], 
                Constants.DEVICE,
                lambda_reg=0.5,
            )
            
            loss.backward()
            optimizer.step()
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
                
                # Ранняя остановка
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    early_stop_counter = 0
                    torch.save(model.state_dict(), DATA_DIR / "best_model_state.pt")
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= patience:
                        print("Early stopping triggered")
                        break
        
        # Сохраняем обученную модель
        model_path = DATA_DIR / "trained_model.pkl"
        _save_obj({
            "model_state": DATA_DIR / "best_model_state.pt",
            "user_encoder": data["user_encoder"],
            "item_encoder": data["item_encoder"],
            "item_features": item_feature_tensor,
            "hidden_dim": Constants.HIDDEN_DIM,
            "num_layers": Constants.NUM_LAYERS,
            "num_heads": 1,  # Для LightGCN не используется
            "dropout": Constants.DROPOUT,
            "max_seq_length": Constants.MAX_SEQ_LENGTH
        }, model_path)
        
        return {
            "model_path": str(model_path)
        }

    @task
    def save_to_mlflow(data_dict):
        """Сохранение модели в MLflow"""
        from src.mlflow_models.lightgcn.model import LightGCNRecommenderWrapper

        # Загружаем модель и данные
        model_data = _load_obj(data_dict["model_path"])
        
        # Настройка окружения MLflow
        os.environ["AWS_ACCESS_KEY_ID"] = "minio"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"

        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("lightgcn-model")

        # Создаем временную директорию для артефактов
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)
        
        # Сохраняем необходимые артефакты
        item_encoder_path = artifacts_dir / "item_encoder.pkl"
        joblib.dump(model_data["item_encoder"], item_encoder_path)
        
        user_encoder_path = artifacts_dir / "user_encoder.pkl"
        joblib.dump(model_data["user_encoder"], user_encoder_path)
        
        item_features_path = artifacts_dir / "item_features.pt"
        torch.save(model_data["item_features"], item_features_path)
        
        # Сохраняем состояние модели
        model_state_path = artifacts_dir / "model_state.pt"
        torch.save({
            "state_dict": torch.load(model_data["model_state"]),
            "hidden_dim": model_data["hidden_dim"],
            "num_layers": model_data["num_layers"],
            "dropout": model_data["dropout"],
            "max_seq_length": model_data["max_seq_length"]
        }, model_state_path)

        with mlflow.start_run(run_name=f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Логируем параметры модели
            mlflow.log_params({
                "num_users": model_data["user_encoder"].classes_.shape[0],
                "num_items": model_data["item_encoder"].classes_.shape[0],
                "hidden_dim": model_data["hidden_dim"],
                "num_layers": model_data["num_layers"],
                "dropout": model_data["dropout"],
                "max_seq_length": model_data["max_seq_length"]
            })

            # Логируем артефакты
            mlflow.log_artifact(item_encoder_path, "artifacts")
            mlflow.log_artifact(user_encoder_path, "artifacts")
            mlflow.log_artifact(item_features_path, "artifacts")
            mlflow.log_artifact(model_state_path, "artifacts")

            # Создаем словарь с путями к артефактам для PythonModel
            artifacts = {
                "item_encoder": str(item_encoder_path),
                "user_encoder": str(user_encoder_path),
                "item_features": str(item_features_path),
                "model_state": str(model_state_path)
            }

            # Логируем pyfunc модель
            mlflow.pyfunc.log_model(
                artifact_path="lightgcn_recommender",
                python_model=LightGCNRecommenderWrapper(),
                artifacts=artifacts,
                registered_model_name="lightgcn_recommender",
                code_path=["/opt/airflow/src/mlflow_models/lightgcn"],
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

    @task
    def create_popular_recommendations(data_dict):
        """Создание популярных рекомендаций на основе частоты покупок"""
        try:
            # Загружаем подготовленные данные
            data = _load_obj(data_dict["dataset_path"])
            transactions = data["transactions"]
            
            # Вычисляем популярные товары (топ-N по частоте покупок)
            popular_items = (
                transactions["item_id"]
                .value_counts()
                .head(Constants.MAX_SEQ_LENGTH)
                .index.tolist()
            )
            
            # Подключаемся к Redis
            r = redis.Redis(
                host=Constants.REDIS_HOST,
                password=Constants.REDIS_PASSWORD,
                port=Constants.REDIS_PORT,
                decode_responses=True,
            )
            
            # Сохраняем популярные товары в Redis
            r.set("popular_recommendations", json.dumps(popular_items))
            
            return "Популярные рекомендации успешно сохранены в Redis"
        except Exception as e:
            print(f"Ошибка при создании популярных рекомендаций: {str(e)}")
            raise

    # Обновленный порядок выполнения задач
    data = prepare_data()
    redis_task = load_to_redis(data)
    popular_recommendations_task = create_popular_recommendations(data)  # Новая задача
    graph_data = create_graph_data(data)
    model = train_model(graph_data)
    save_to_mlflow(model)

    # Обновленные зависимости
    data >> redis_task
    data >> popular_recommendations_task  # Зависит от данных
    data >> graph_data


enhanced_lightgcn_pipeline_dag = enhanced_lightgcn_pipeline()
