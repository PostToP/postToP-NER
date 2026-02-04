import copy
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from config.config import TABLE_BACK
from model.model import build_model, evaluate_model


class NERDataset(Dataset):
    def __init__(
        self,
        titles: np.ndarray,
        token_features: np.ndarray,
        global_features: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self.titles = titles.astype(np.int64)
        self.token_features = token_features.astype(np.float32)
        self.global_features = global_features.astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.titles)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.titles[idx]),
            torch.from_numpy(self.token_features[idx]),
            torch.from_numpy(self.global_features[idx]),
            torch.from_numpy(self.labels[idx]),
        )


def load_split(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return (
        data["titles"],
        data["token_features"],
        data["global_features"],
        data["ner_tags"],
    )


def set_seed(seed: int):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_with_seed(seed: int):
    set_seed(seed)
    device = torch.device("cuda")

    train_titles, train_token_feats, train_global_feats, train_labels = load_split(
        "dataset/p5_dataset_train.npz"
    )
    val_titles, val_token_feats, val_global_feats, val_labels = load_split(
        "dataset/p5_dataset_val.npz"
    )

    print("Dataset stats:")
    print(f"Train samples: {len(train_titles)} | Val samples: {len(val_titles)}")
    print(
        f"Sequence length: {train_titles.shape[1]} | Token feature dim: {train_token_feats.shape[2]} | Global feature dim: {train_global_feats.shape[1]}"
    )

    train_dataset = NERDataset(
        train_titles, train_token_feats, train_global_feats, train_labels
    )
    val_dataset = NERDataset(val_titles, val_token_feats, val_global_feats, val_labels)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=1024,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id),
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id),
        generator=g,
    )

    token_feature_dim = train_token_feats.shape[2]
    global_feature_dim = train_global_feats.shape[1]
    model = build_model(token_feature_dim, global_feature_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=10,
        min_lr=1e-6,
    )
    criterion = torch.nn.CrossEntropyLoss()

    best_f1 = -float("inf")
    best_state = None
    patience = 100
    delta = 0.000
    patience_counter = 0
    max_epochs = 5000

    for epoch in range(1, max_epochs + 1):
        model.train()
        token_loss_sum = 0.0
        token_count = 0

        for tokens, token_features, global_features, labels in train_loader:
            tokens = tokens.to(device)
            token_features = token_features.to(device)
            global_features = global_features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(tokens, token_features, global_features)
            mask = tokens != 0
            if not mask.any():
                continue

            valid_logits = logits[mask]
            valid_labels = labels[mask]
            loss = criterion(valid_logits, valid_labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            token_loss_sum += loss.item() * valid_labels.numel()
            token_count += valid_labels.numel()

        train_loss = token_loss_sum / token_count if token_count > 0 else 0.0
        val_metrics = evaluate_model(model, val_loader, device)
        scheduler.step(val_metrics["f1_micro"])

        print(
            f"Epoch {epoch:04d} | train_loss {train_loss:.4f} | val_loss {val_metrics['loss']:.4f} | val_f1 {val_metrics['f1_micro']:.4f} | val_acc {val_metrics['accuracy']:.4f}"
        )

        if val_metrics["f1_micro"] > best_f1:
            best_f1 = val_metrics["f1_micro"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        elif val_metrics["f1_micro"] > best_f1 - delta:
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "token_feature_dim": token_feature_dim,
            "global_feature_dim": global_feature_dim,
            "num_labels": len(TABLE_BACK),
        },
        "model/final_model.pth",
    )

    final_metrics = evaluate_model(model, val_loader, device)
    print("Final validation metrics:", final_metrics)

    return final_metrics["f1_macro"]


def main() -> None:
    SEEDS = [123]
    f1_macros = []
    for seed in SEEDS:
        print(f"Running training with seed {seed}...")
        f1_macro = run_with_seed(seed)
        f1_macros.append(f1_macro)
        print(f"Seed {seed} | F1 Macro: {f1_macro:.4f}")

    avg_f1_macro = sum(f1_macros) / len(f1_macros)
    print(f"Average F1 Macro over seeds: {avg_f1_macro:.4f}")
