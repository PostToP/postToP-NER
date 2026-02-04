from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch import nn
from config.config import TABLE_BACK, VOCAB_SIZE


class SequenceTagger(nn.Module):
    def __init__(
        self,
        token_feature_dim: int,
        global_feature_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        embed_dim = 2**4
        rnn_hidden_size = 2**8
        self.embedding = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=0)

        concat_dim = embed_dim + token_feature_dim + global_feature_dim

        self.layernorm1 = nn.LayerNorm(concat_dim)
        self.bilstm = nn.GRU(
            input_size=concat_dim,
            hidden_size=rnn_hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.layernorm2 = nn.LayerNorm(rnn_hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(rnn_hidden_size * 2, len(TABLE_BACK))

    def forward(
        self,
        tokens: torch.Tensor,
        token_features: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        embedded = self.embedding(tokens)

        global_expanded = global_features.unsqueeze(1).expand(-1, tokens.size(1), -1)

        x = torch.cat([embedded, token_features, global_expanded], dim=-1)
        x = self.layernorm1(x)
        x, _ = self.bilstm(x)
        x = self.layernorm2(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits


def build_model(token_feature_dim: int, global_feature_dim: int) -> SequenceTagger:
    return SequenceTagger(
        token_feature_dim=token_feature_dim,
        global_feature_dim=global_feature_dim,
    )


def _compute_f1(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    if len(y_true) == 0:
        return {
            "f1_micro": 0.0,
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
            "f1_per_class": {},
        }

    labels = list(range(len(TABLE_BACK)))
    f1_micro = f1_score(y_true, y_pred, average="micro", labels=labels, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    f1_weighted = f1_score(
        y_true,
        y_pred,
        average="weighted",
        labels=labels,
        zero_division=0,
    )
    f1_per_class_scores = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    return {
        "f1_micro": float(f1_micro),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "f1_per_class": {
            TABLE_BACK[label]: float(score)
            for label, score in zip(labels, f1_per_class_scores)
        },
    }


def evaluate_model(
    model: nn.Module,
    data_loader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    loss_sum = 0.0
    total_tokens = 0
    correct_tokens = 0
    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.no_grad():
        for tokens, token_features, global_features, labels in data_loader:
            tokens = tokens.to(device)
            token_features = token_features.to(device)
            global_features = global_features.to(device)
            labels = labels.to(device)

            logits = model(tokens, token_features, global_features)
            mask = tokens != 0

            if mask.any():
                valid_logits = logits[mask]
                valid_labels = labels[mask]
                batch_loss = F.cross_entropy(
                    valid_logits, valid_labels, reduction="sum"
                )
                loss_sum += batch_loss.item()
                total_tokens += valid_labels.numel()
            else:
                continue

            predictions = torch.argmax(logits, dim=-1)
            correct_tokens += ((predictions == labels) & mask).sum().item()

            tokens_np = tokens.cpu().numpy()
            labels_np = labels.cpu().numpy()
            preds_np = predictions.cpu().numpy()

            for i in range(tokens_np.shape[0]):
                seq_mask = tokens_np[i] != 0
                if not np.any(seq_mask):
                    continue
                y_true.extend(labels_np[i][seq_mask].tolist())
                y_pred.extend(preds_np[i][seq_mask].tolist())

    f1_scores = _compute_f1(y_true, y_pred)
    average_loss = loss_sum / total_tokens if total_tokens > 0 else 0.0
    accuracy = (correct_tokens / total_tokens) if total_tokens > 0 else 0.0

    return {
        "loss": average_loss,
        "accuracy": accuracy,
        **f1_scores,
    }
