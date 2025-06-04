import mlflow.pyfunc
import torch
import joblib
from typing import Dict, Any, List, Union
from torch_geometric.nn import LGConv
from sklearn.preprocessing import LabelEncoder


# Инициализация модели
class EnhancedLightGCN(torch.nn.Module):
    # Копия архитектуры модели
    def __init__(self, num_users, num_items, embedding_dim=128, num_layers=3, 
                    dropout=0.1, item_features=None):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.dropout = dropout
        self.user_emb = torch.nn.Embedding(num_users, embedding_dim)
        self.item_emb = torch.nn.Embedding(num_items, embedding_dim)
        
        if item_features is not None:
            self.item_feature_proj = torch.nn.Linear(item_features.shape[1], embedding_dim)
            self.item_features = item_features
        else:
            self.item_feature_proj = None
        
        self.convs = torch.nn.ModuleList([LGConv() for _ in range(num_layers)])
        self.dropout = torch.nn.Dropout(p=dropout)
        self.attention = torch.nn.Parameter(torch.ones(num_layers + 1, 1, 1))

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


class LightGCNRecommender:
    def __init__(
        self,
        model: 'EnhancedLightGCN',
        item_encoder: LabelEncoder,
        user_encoder: LabelEncoder,
        item_features: torch.Tensor = None,
        device: str = "cpu"
    ):
        self.model = model.to(device)
        self.model.eval()
        self.item_encoder = item_encoder
        self.user_encoder = user_encoder
        self.item_features = item_features
        self.device = device
        self.edge_index = None  # Будет установлено при загрузке

    def _prepare_input(self, user_id: str) -> torch.Tensor:
        """Подготовка входных данных для модели"""
        user_idx = self.user_encoder.transform([user_id])[0]
        return torch.tensor([user_idx], device=self.device)

    def recommend(
        self,
        user_id: Union[str, List[str]],
        k: int = 50,
        return_scores: bool = False
    ) -> Union[List[str], List[tuple]]:
        """Генерация рекомендаций для пользователя"""
        if isinstance(user_id, list):
            user_id = user_id[0]  # Берем первого пользователя если передан список
            
        user_idx = self._prepare_input(user_id)
        
        with torch.no_grad():
            user_emb, item_emb = self.model(self.edge_index)
            scores = user_emb[user_idx] @ item_emb.T
            
            top_scores, top_indices = torch.topk(scores, k=k)
            recommended_items = self.item_encoder.inverse_transform(top_indices.cpu().numpy().flatten())
            
            if return_scores:
                return list(zip(recommended_items, top_scores.cpu().numpy().flatten()))
            return recommended_items.tolist()


class LightGCNWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self):
        self.recommender = None
    
    def load_context(self, context):
        """Загрузка модели и артефактов"""
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Загрузка энкодеров
        item_encoder = joblib.load(context.artifacts["item_encoder"])
        user_encoder = joblib.load(context.artifacts["user_encoder"])
        
        # Загрузка состояния модели
        model_state = torch.load(context.artifacts["model_state"], map_location=device)
        edge_index = torch.load(context.artifacts["edge_index"], map_location=device)
        
        # Загрузка item features если есть
        item_features = None
        if "item_features" in context.artifacts:
            item_features = torch.load(context.artifacts["item_features"], map_location=device)
        
        # Создание модели
        model = EnhancedLightGCN(
            num_users=len(user_encoder.classes_),
            num_items=len(item_encoder.classes_),
            embedding_dim=model_state["embedding_dim"],
            num_layers=model_state["num_layers"],
            dropout=model_state["dropout"],
            item_features=item_features
        )
        model.load_state_dict(model_state["state_dict"])
        
        # Инициализация рекомендательной системы
        self.recommender = LightGCNRecommender(
            model=model,
            item_encoder=item_encoder,
            user_encoder=user_encoder,
            item_features=item_features,
            device=device
        )
        self.recommender.edge_index = edge_index
    
    def predict(self, context, model_input: Dict[str, Any], params: Dict[str, Any] = None):
        """Генерация предсказаний"""
        if params is None:
            params = {}
        
        k = params.get("k", 10)
        return_scores = params.get("return_scores", False)
        
        if isinstance(model_input, dict):
            user_id = model_input.get("user_id")
        else:
            user_id = str(model_input)
        
        return self.recommender.recommend(
            user_id=user_id,
            k=k,
            return_scores=return_scores
        )
