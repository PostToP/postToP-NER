import tarfile
import tempfile
from pathlib import Path
import json
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from config.config import TRANSFORMER_MODEL_NAME


class ModelWrapper:
    def __init__(self, model, session=None):
        self.model_name = TRANSFORMER_MODEL_NAME
        self.model = model
        self.session = session
        self.tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)

    def serialize(self, location):
        import torch
        from onnxruntime.quantization import quantize_dynamic, QuantType
        from onnxruntime.transformers import optimizer

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.onnx"

            device = next(self.model.parameters()).device
            self.model = self.model.cpu()

            torch.onnx.export(
                self.model,
                (
                    torch.zeros(1, 512, dtype=torch.long),
                    torch.zeros(1, 512, dtype=torch.long),
                    torch.zeros(1, 512, dtype=torch.float),
                ),
                model_path.as_posix(),
                input_names=["input_ids", "attention_mask", "features"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "batch_size", 1: "sequence_length"},
                    "features": {0: "batch_size", 1: "sequence_length"},
                    "logits": {0: "batch_size", 1: "sequence_length"},
                },
                opset_version=18,
            )

            opt_model = optimizer.optimize_model(
                model_path.as_posix(),
                model_type="bert",
                num_heads=12,
                hidden_size=768,
            )
            opt_model.save_model_to_file(model_path.as_posix())

            final_model_path = Path(tmpdir) / "model_final.onnx"
            quantize_dynamic(
                model_input=model_path.as_posix(),
                model_output=final_model_path.as_posix(),
                weight_type=QuantType.QInt8,
            )

            self.model = self.model.to(device)

            config = {
                "model_name": self.model_name,
            }
            config_path = Path(tmpdir) / "config.json"
            with open(config_path, "w") as f:
                json.dump(config, f)

            with tarfile.open(location, "w:gz") as tar:
                tar.add(final_model_path, arcname="model.onnx")
                tar.add(config_path, arcname="config.json")

    @staticmethod
    def deserialize(location):
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(location, "r:gz") as tar:
                tar.extractall(tmpdir)
            config_path = Path(tmpdir) / "config.json"
            with open(config_path, "r") as f:
                config = json.load(f)
            model_name = config["model_name"]
            model_path = Path(tmpdir) / "model.onnx"
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = ort.InferenceSession(
                model_path.as_posix(),
                so,
                providers=["CPUExecutionProvider"],
            )
            return ModelWrapper(model=None, session=session)

    def warmup(self):
        if self.session is not None:
            dummy_input_ids = np.zeros((1, 512), dtype=np.int64)
            dummy_attention_mask = np.zeros((1, 512), dtype=np.int64)
            dummy_features = np.zeros((1, 512), dtype=np.float32)
            self.session.run(
                None,
                {
                    "input_ids": dummy_input_ids,
                    "attention_mask": dummy_attention_mask,
                    # "features": dummy_features,
                },
            )

    def preprocess_input(self, title, description):
        text = title + " [SEP] " + description
        encoding = self.tokenizer(
            text,
            padding="longest",
            truncation=True,
            max_length=512,
            return_tensors="np",
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        features = np.zeros((input_ids.shape[0], input_ids.shape[1]), dtype=np.float32)
        return input_ids, attention_mask, features

    def predict(self, title, description):
        input_ids, attention_mask, features = self.preprocess_input(title, description)
        outputs = self.session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                # "features": features,
            },
        )
        logits = outputs[0]
        return logits
