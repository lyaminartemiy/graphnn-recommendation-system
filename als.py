import torch
import torch.nn as nn
import torch.nn.functional as F
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
