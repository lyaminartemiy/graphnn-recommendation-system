import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data


class TransactionDataset:
    def __init__(
        self,
        data: pd.DataFrame,
        item_encoder: LabelEncoder = None,
        min_sequence_length: int = 2,
    ):
        """
        Датасет для последовательностей товаров (без user_id)

        Args:
            data: DataFrame с колонкой 'item_id'
            item_encoder: Предобученный LabelEncoder для товаров
            min_sequence_length: Минимальная длина последовательности
        """
        self.df = data.copy()
        self.min_seq_len = min_sequence_length

        # Инициализация энкодера товаров
        self.item_encoder = item_encoder if item_encoder else LabelEncoder()
        if item_encoder is None:
            self.item_encoder.fit(self.df["item_id"].unique())

        self.df["item_idx"] = self.item_encoder.transform(self.df["item_id"])
        self.num_items = len(self.item_encoder.classes_)

        # Группируем все товары в одну последовательность
        self.full_sequence = self.df["item_idx"].tolist()

    def __len__(self):
        return max(0, len(self.full_sequence) - self.min_seq_len + 1)

    def __getitem__(self, idx):
        """
        Возвращает граф для последовательности товаров
        items[idx:idx+min_seq_len+1]
        """
        end_idx = idx + self.min_seq_len + 1
        items = self.full_sequence[idx:end_idx]

        if len(items) < 2:
            return self._create_empty_graph()

        # Узлы - все товары кроме последнего
        nodes = torch.tensor(items[:-1], dtype=torch.long)

        # Ребра - последовательные переходы
        edge_index = (
            torch.tensor(
                [[i, i + 1] for i in range(len(items) - 2)],  # Связи между узлами
                dtype=torch.long,
            )
            .t()
            .contiguous()
        )

        # Целевой товар - следующий в последовательности
        target = torch.tensor(items[-1], dtype=torch.long)

        return Data(
            x=torch.zeros(len(nodes), 1),  # Фиктивные признаки
            edge_index=edge_index,
            y=nodes,  # Исходные товары
            target=target,  # Следующий товар
        )

    def _create_empty_graph(self):
        return Data(
            x=torch.zeros(0, 1),
            edge_index=torch.zeros(2, 0),
            y=torch.zeros(0),
            target=torch.tensor(-1),  # Индикатор отсутствия данных
        )
