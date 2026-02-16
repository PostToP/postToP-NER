from typing import Dict, List

import torch
from sklearn.metrics import f1_score
from torch import nn
from config.config import DEVICE, NUM_LABELS, TABLE_BACK, TRANSFORMER_MODEL_NAME

from transformers import AutoModel


class TransformerModel(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(TRANSFORMER_MODEL_NAME)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits


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
    val_loader,
) -> Dict[str, float]:
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    model.eval()

    loss_sum = 0.0
    total_tokens = 0
    correct_tokens = 0
    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            logits = model(input_ids, attention_mask)

            loss_sum += criterion(logits.view(-1, NUM_LABELS), labels.view(-1))

            predictions = torch.argmax(logits, dim=-1)

            for pred, label in zip(predictions.cpu().numpy(), labels.cpu().numpy()):
                mask = label != -100
                y_pred.extend(pred[mask])
                y_true.extend(label[mask])
                correct_tokens += (pred[mask] == label[mask]).sum()
                total_tokens += mask.sum()

    f1_scores = _compute_f1(y_true, y_pred)
    average_loss = loss_sum / len(val_loader)
    accuracy = (correct_tokens / total_tokens) if total_tokens > 0 else 0.0

    return {
        "loss": average_loss,
        "accuracy": accuracy,
        **f1_scores,
    }
