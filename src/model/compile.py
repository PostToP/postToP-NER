import torch

from config.config import TABLE_BACK

from model.ModelWrapper import ModelWrapper
from model.model import TransformerModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compile_model():
    model = TransformerModel(len(TABLE_BACK)).to(DEVICE)
    checkpoint = torch.load("model/final_model.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    model_wrapper = ModelWrapper(model)
    model_wrapper.serialize("model/compiled_model.tar.gz")
