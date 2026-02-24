import pandas as pd
import torch

from config.config import TABLE_BACK
from torch.utils.data import DataLoader, Dataset

from model.ModelWrapper import ModelWrapper
from model.model import TransformerModel, _compute_f1
from model.train import NERDataset, collate_fn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compile_model():
    model = TransformerModel(len(TABLE_BACK)).to(DEVICE)
    checkpoint = torch.load("model/final_model.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    val_df = pd.read_json("dataset/p5_dataset_val.json")

    val_dataset = NERDataset(val_df)
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            features = batch["features"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            logits = model(input_ids, attention_mask, features)
            predictions = torch.argmax(logits, dim=-1)
            mask = labels != -100

            y_pred.append(predictions[mask].detach())
            y_true.append(labels[mask].detach())

    y_pred = torch.cat(y_pred).cpu().tolist()
    y_true = torch.cat(y_true).cpu().tolist()

    res = _compute_f1(y_true, y_pred)

    print("Evaluation results on validation set:")
    print(f"F1 Micro: {res['f1_micro']:.4f}")
    print(f"F1 Macro: {res['f1_macro']:.4f}")
    print(f"F1 Weighted: {res['f1_weighted']:.4f}")
    print("F1 per class:")
    for label, f1 in res["f1_per_class"].items():
        print(f"  {label}: {f1:.4f}")

    model_wrapper = ModelWrapper(model)
    model_wrapper.serialize("model/compiled_model.tar.gz")

    model_wrapper = ModelWrapper.deserialize("model/compiled_model.tar.gz")

    session = model_wrapper.session

    all_predictions = []
    all_labels = []
    for row in val_dataset:
        input_ids = row["input_ids"].unsqueeze(0).cpu().numpy()
        attention_mask = row["attention_mask"].unsqueeze(0).cpu().numpy()
        # features = row["features"].unsqueeze(0).cpu().numpy()
        outputs = session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                # "features": torch.zeros(1, 512, dtype=torch.float).cpu().numpy(),
            },
        )
        logits = torch.from_numpy(outputs[0])
        predictions = torch.argmax(logits, dim=-1).squeeze().cpu().tolist()
        mask = row["labels"] != -100
        all_predictions.append([pred for pred, m in zip(predictions, mask) if m])
        all_labels.append([label.item() for label, m in zip(row["labels"], mask) if m])
    print(all_labels[0])
    print(all_predictions[0])
    all_labels_flat = [item for sublist in all_labels for item in sublist]
    all_predictions_flat = [item for sublist in all_predictions for item in sublist]
    f1_scores = _compute_f1(all_labels_flat, all_predictions_flat)
    print("Evaluation results on validation set (after deserialization):")
    print(f"F1 Micro: {f1_scores['f1_micro']:.4f}")
    print(f"F1 Macro: {f1_scores['f1_macro']:.4f}")
    print(f"F1 Weighted: {f1_scores['f1_weighted']:.4f}")
    print("F1 per class:")
    for label, f1 in f1_scores["f1_per_class"].items():
        print(f"  {label}: {f1:.4f}")

    print("Performance Degradation Check:")
    print(f"F1 Micro Degradation: {res['f1_micro'] - f1_scores['f1_micro']:.4f}")
    print(f"F1 Macro Degradation: {res['f1_macro'] - f1_scores['f1_macro']:.4f}")
    print(
        f"F1 Weighted Degradation: {res['f1_weighted'] - f1_scores['f1_weighted']:.4f}"
    )
    print("F1 per class Degradation:")
    for label in res["f1_per_class"]:
        degradation = res["f1_per_class"][label] - f1_scores["f1_per_class"].get(
            label, 0
        )
        print(f"  {label}: {degradation:.4f}")
