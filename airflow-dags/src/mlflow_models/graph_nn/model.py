import mlflow.pyfunc
import torch
import joblib
from typing import Dict, Any, List, Union, Optional

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from torch_geometric.nn import GATConv


class TransactionGNN(nn.Module):
    def __init__(
        self,
        num_items: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout

        # Эмбеддинг товаров
        self.item_embedding = nn.Embedding(num_items, hidden_dim)

        # GAT-слои
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_channels = hidden_dim * num_heads if i > 0 else hidden_dim
            out_channels = hidden_dim
            self.gat_layers.append(
                GATConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=num_heads,
                    concat=True,
                    dropout=dropout,
                    add_self_loops=True,
                )
            )

        # Классификатор
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * num_heads, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_items),
        )

    def forward(self, data):
        # Эмбеддинги товаров
        x = self.item_embedding(data.y)

        # Применяем GAT-слои
        for layer in self.gat_layers:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(layer(x, data.edge_index))
            x = F.normalize(x, p=2, dim=-1)  # Нормализация

        # Предсказание следующего товара
        if x.size(0) == 0:
            return torch.zeros(0, self.predictor[-1].out_features)

        # Берем последний узел как контекст
        context = x[-1].unsqueeze(0)
        logits = self.predictor(context)
        return logits.squeeze(0)

    def predict_next_item(self, data, top_k=5):
        """Предсказание top-k следующих товаров"""
        with torch.no_grad():
            if data.num_nodes == 0:
                return torch.zeros(0, top_k, dtype=torch.long), torch.zeros(0, top_k)

            logits = self.forward(data)
            
            # Обработка случая, когда logits пустые
            if logits.numel() == 0:
                return torch.zeros(0, top_k, dtype=torch.long), torch.zeros(0, top_k)

            probs = F.softmax(logits, dim=-1)
            top_probs, top_items = torch.topk(probs, k=min(top_k, probs.size(-1)))
            
            return top_items, top_probs


class GNNRecommender:
    def __init__(
        self,
        model: TransactionGNN,
        item_encoder: LabelEncoder,
        user_encoder: Optional[LabelEncoder] = None,
        device: str = "cpu",
        max_seq_length: int = 50,
    ):
        """
        Упрощенная обертка для графовой модели рекомендаций (без временных меток)

        Args:
            model: Обученная модель TransactionGNN
            item_encoder: Энкодер для item_id
            user_encoder: Энкодер для user_id (опционально)
            device: Устройство для вычислений ('cpu' или 'cuda')
            max_seq_length: Максимальная длина последовательности
        """
        self.model = model.to(device)
        self.model.eval()
        self.item_encoder = item_encoder
        self.user_encoder = user_encoder
        self.device = device
        self.max_seq_length = max_seq_length

    def _create_graph(self, items: List[str]) -> Data:
        """
        Создает граф из последовательности товаров

        Args:
            items: Список item_id

        Returns:
            Объект Data (pyg) с графом последовательности
        """
        if not items:
            return Data(
                x=torch.zeros((0, 1), dtype=torch.float),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                y=torch.zeros(0, dtype=torch.long),
                num_nodes=0,
            )

        # Преобразуем item_id в индексы
        try:
            item_indices = self.item_encoder.transform(items)
        except ValueError:
            # Для неизвестных товаров используем нулевой индекс
            item_indices = np.zeros(len(items), dtype=int)

        # Обрезаем последовательность если нужно
        if len(items) > self.max_seq_length:
            items = items[-self.max_seq_length :]
            item_indices = item_indices[-self.max_seq_length :]

        # Создаем узлы графа (позиции в последовательности как признаки)
        positions = torch.arange(len(items), dtype=torch.float).unsqueeze(1) / len(
            items
        )

        # Создаем ребра (направленные связи между последовательными товарами)
        edge_index = (
            torch.tensor([[i, i + 1] for i in range(len(items) - 1)], dtype=torch.long)
            .t()
            .contiguous()
        )

        return Data(
            x=positions,
            edge_index=edge_index,
            y=torch.tensor(item_indices, dtype=torch.long),
            num_nodes=len(items),
        )

    def recommend(
        self,
        items: Union[List[str], Dict[str, List[str]]],
        top_k: int = 5,
        return_scores: bool = False,
    ) -> Union[List[str], List[tuple]]:
        """
        Генерирует рекомендации на основе истории товаров

        Args:
            items: Список item_id ИЛИ словарь с ключом 'items'
            top_k: Количество возвращаемых рекомендаций
            return_scores: Если True, возвращает (item_id, score)

        Returns:
            Рекомендации в указанном формате
        """
        # Поддержка как списка, так и словаря с ключом 'items'
        if isinstance(items, dict):
            items = items.get("items", [])

        if not items:
            return [] if not return_scores else []

        # Создаем граф
        graph = self._create_graph(items)
        graph = graph.to(self.device)

        # Получаем предсказания
        with torch.no_grad():
            top_items, top_probs = self.model.predict_next_item(graph, top_k=top_k)

        # Если нет предсказаний (слишком короткая последовательность)
        if top_items.numel() == 0:
            return [] if not return_scores else []

        # Преобразуем индексы обратно в item_id
        recommended_items = self.item_encoder.inverse_transform(
            top_items.cpu().numpy().flatten()
        )

        if return_scores:
            scores = top_probs.cpu().numpy().flatten()
            return list(zip(recommended_items, scores))

        return recommended_items.tolist()


class GNNRecommenderWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model: 'GNNRecommender' = None):
        self.model = model
    
    def load_context(self, context):
        """
        Загружает артефакты модели из контекста MLflow
        """

        item_encoder = joblib.load(context.artifacts["item_encoder"])
        print(f"Item encoder type: {type(item_encoder)}")
        
        # Загрузка состояния модели
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_state = torch.load(context.artifacts["model_state"], map_location=device)
        
        # Создание модели
        gnn_model = TransactionGNN(
            num_items=len(item_encoder.classes_),
            hidden_dim=model_state["hidden_dim"],
            num_layers=model_state["num_layers"],
            num_heads=model_state["num_heads"],
            dropout=model_state.get("dropout", 0.1)
        )
        gnn_model.load_state_dict(model_state["state_dict"])
        
        # Инициализация враппера
        self.model = GNNRecommender(
            model=gnn_model,
            item_encoder=item_encoder,
            user_encoder=None,
            device=device,
            max_seq_length=model_state.get("max_seq_length", 50)
        )
    
    def predict(self, context, model_input: Union[List[str], Dict[str, Any]], params: Dict[str, Any] = None):
        """
        Генерирует предсказания для входных данных
        
        Args:
            model_input: Может быть:
                - Список item_id
                - Словарь с ключом 'items' (список item_id)
            params: Дополнительные параметры:
                - top_k: количество рекомендаций (по умолчанию 5)
                - return_scores: возвращать ли оценки (по умолчанию False)
        """
        if params is None:
            params = {}
        
        top_k = params.get("top_k", 5)
        return_scores = params.get("return_scores", False)
        
        return self.model.recommend(model_input, top_k=top_k, return_scores=return_scores)
