import torch
import torch.nn.functional as F
from src.modules.graph_nn.model import TransactionGNN
from torch_geometric.loader import DataLoader
from tqdm import tqdm


def train_epoch(
    model: TransactionGNN, loader: DataLoader, optimizer: torch.optim.Adam, device: str
) -> float:
    model.train()
    total_loss = 0
    total_samples = 0

    for data in tqdm(loader, desc="Train Epoch:"):
        data = data.to(device)
        optimizer.zero_grad()

        if data.num_nodes == 0:
            continue

        targets = data.y[1:]
        logits = model(data)
        loss = F.cross_entropy(logits.unsqueeze(0).expand(len(targets), -1), targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(targets)
        total_samples += len(targets)

    return total_loss / total_samples if total_samples > 0 else 0


def evaluate_epoch(model: TransactionGNN, loader: DataLoader, device: str) -> tuple:
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for data in tqdm(loader, desc="Evaluate Epoch:"):
            data = data.to(device)

            if data.num_nodes == 0:
                continue

            targets = data.y[1:]
            logits = model(data)

            preds = logits.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_samples += len(targets)

            loss = F.cross_entropy(
                logits.unsqueeze(0).expand(len(targets), -1), targets
            )
            total_loss += loss.item() * len(targets)

    acc = total_correct / total_samples if total_samples > 0 else 0
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    return acc, avg_loss
