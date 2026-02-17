import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from config.config import DEVICE, TABLE, TABLE_BACK
from model.EarlyStopping import EarlyStopping
from model.model import TransformerModel, evaluate_model
from torch.amp import autocast, GradScaler
from torch.nn.utils.rnn import pad_sequence


class NERDataset(Dataset):
    def __init__(self, df):
        self.input_ids = [
            torch.tensor(x, dtype=torch.long) for x in df["Word IDs"].values
        ]
        self.attention_masks = [
            torch.tensor(x, dtype=torch.long) for x in df["Attention Mask"].values
        ]
        self.features = [
            torch.tensor(x, dtype=torch.float) for x in df["Features"].values
        ]
        self.labels = [torch.tensor(x, dtype=torch.long) for x in df["NER"].values]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "features": self.features[idx],
            "labels": self.labels[idx],
        }


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


def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    features = [item["features"] for item in batch]
    labels = [item["labels"] for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    features = pad_sequence(features, batch_first=True, padding_value=0.0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "features": features,
        "labels": labels,
    }


def run_with_seed(seed: int = None, verbose: bool = True) -> float:
    if seed is None:
        seed = np.random.randint(0, 10000)
    set_seed(seed)

    train_df = pd.read_json("dataset/p5_dataset_train.json")
    train_df = train_df.sample(frac=1).sort_values(
        by="Word IDs", key=lambda x: x.str.len()
    )
    val_df = pd.read_json("dataset/p5_dataset_val.json")

    train_dataset = NERDataset(train_df)
    val_dataset = NERDataset(val_df)

    if verbose:
        print("Dataset stats:")
        print(f"Train samples: {len(train_df)} | Val samples: {len(val_df)}")
        print(
            f"Sequence length: {train_dataset.input_ids[0].shape[0]} | Token feature dim: {0} | Global feature dim: {0}"
        )

    g = torch.Generator()
    g.manual_seed(seed)

    BATCH_SIZE = 32

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id),
        collate_fn=collate_fn,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id),
        generator=g,
    )

    model = TransformerModel(num_labels=len(TABLE_BACK)).to(DEVICE)

    LR = 3e-5
    EPOCHS = 40
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * EPOCHS
    )
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    scaler = GradScaler()

    early_stopping = EarlyStopping(patience=3, min_delta=0.000)

    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            features = batch["features"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            optimizer.zero_grad()

            with autocast("cuda"):
                logits = model(input_ids, attention_mask, features)
                loss = criterion(logits.view(-1, len(TABLE)), labels.view(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": total_loss / (progress_bar.n + 1)})

        train_loss = total_loss / len(train_loader)
        val_metrics = evaluate_model(model, val_loader)
        scheduler.step()

        if verbose:
            print(
                f"Epoch {epoch:04d} | train_loss {train_loss:.4f} | val_loss {val_metrics['loss']:.4f} | val_f1_micro {val_metrics['f1_micro']:.4f} | val_f1_macro {val_metrics['f1_macro']:.4f} | val_acc {val_metrics['accuracy']:.4f}"
            )

        if val_metrics["f1_macro"] > early_stopping.best_score:
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        if early_stopping(val_metrics["f1_macro"]):
            if verbose:
                print("Early stopping triggered.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "num_labels": len(TABLE_BACK),
        },
        "model/final_model.pth",
    )

    final_metrics = evaluate_model(model, val_loader)
    if verbose:
        print("Final validation metrics:", final_metrics)

    return final_metrics["f1_macro"]


def main() -> None:
    SEEDS = [42, 123, 2024, 7, 999]
    f1_macros = []
    for seed in SEEDS:
        print(f"Running training with seed {seed}...")
        f1_macro = run_with_seed(seed=seed, verbose=True)
        f1_macros.append(f1_macro)
        print(f"Seed {seed} | F1 Macro: {f1_macro:.4f}")

    avg_f1_macro = sum(f1_macros) / len(f1_macros)
    print(f"Average F1 Macro over seeds: {avg_f1_macro:.4f}")
    print(f"F1 Macro Std Dev: {np.std(f1_macros):.4f}")
