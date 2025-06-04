import mlflow.pyfunc
import torch
import joblib
from typing import Dict, Any, List, Union
from sklearn.preprocessing import LabelEncoder
from torch_geometric.nn import LGConv
import torch.nn as nn


class LightGCN(nn.Module):
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

    def predict_next_items(self, user_indices, item_indices, top_k=5):
        """Предсказание top-k товаров для пользователей"""
        with torch.no_grad():
            user_emb, item_emb = self.forward(torch.tensor([[], []], dtype=torch.long))
            scores = user_emb[user_indices] @ item_emb.T
            top_probs, top_items = torch.topk(scores, k=top_k, dim=1)
            return top_items, top_probs

class LightGCNRecommender:
    def __init__(
        self,
        model: LightGCN,
        item_encoder: LabelEncoder,
        user_encoder: LabelEncoder,
        item_features: torch.Tensor,
        device: str = "cpu",
        max_seq_length: int = 50,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.item_encoder = item_encoder
        self.user_encoder = user_encoder
        self.item_features = item_features
        self.device = device
        self.max_seq_length = max_seq_length

    def recommend(
        self,
        user_ids: Union[List[str], Dict[str, List[str]]],
        top_k: int = 5,
        return_scores: bool = False,
    ) -> Union[List[str], List[tuple]]:
        if isinstance(user_ids, dict):
            user_ids = user_ids.get("user_ids", [])
        
        if not user_ids:
            return [] if not return_scores else []
        
        try:
            user_indices = self.user_encoder.transform(user_ids)
        except ValueError:
            return [] if not return_scores else []
        
        with torch.no_grad():
            user_emb, item_emb = self.model(torch.tensor([[], []], dtype=torch.long).to(self.device))
            scores = user_emb[user_indices] @ item_emb.T
            top_probs, top_items = torch.topk(scores, k=top_k, dim=1)
        
        recommended_items = self.item_encoder.inverse_transform(top_items.cpu().numpy().flatten())
        
        if return_scores:
            scores = top_probs.cpu().numpy().flatten()
            return list(zip(recommended_items, scores))
        
        return recommended_items.tolist()

class LightGCNRecommenderWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self):
        self.model = None
    
    def load_context(self, context):
        item_encoder = joblib.load(context.artifacts["item_encoder"])
        user_encoder = joblib.load(context.artifacts["user_encoder"])
        item_features = torch.load(context.artifacts["item_features"])
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_state = torch.load(context.artifacts["model_state"], map_location=device)
        
        # Сначала создаем LightGCN модель
        gnn_model = LightGCN(
            num_users=len(user_encoder.classes_),
            num_items=len(item_encoder.classes_),
            embedding_dim=model_state["hidden_dim"],
            num_layers=model_state["num_layers"],
            dropout=model_state.get("dropout", 0.1),
            item_features=item_features
        )
        gnn_model.load_state_dict(model_state["state_dict"])
        
        # Затем создаем LightGCNRecommender обертку
        self.model = LightGCNRecommender(
            model=gnn_model,
            item_encoder=item_encoder,
            user_encoder=user_encoder,
            item_features=item_features,
            device=device,
            max_seq_length=model_state.get("max_seq_length", 50)
        )
    
    def predict(self, context, model_input: Union[List[str], Dict[str, Any]], params: Dict[str, Any] = None):
        if params is None:
            params = {}
        
        top_k = params.get("top_k", 5)
        return_scores = params.get("return_scores", False)
        
        return self.model.recommend(model_input, top_k=top_k, return_scores=return_scores)
