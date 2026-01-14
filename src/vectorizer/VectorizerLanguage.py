import numpy as np


def VectorizerLanguage(lang: str) -> np.ndarray:
    lang = lang.split("-")[0] if lang else None
    if lang == "ja":
        return np.array([4])
    elif lang == "en":
        return np.array([3])
    elif lang == "hu":
        return np.array([2])
    elif lang is None:
        return np.array([1])
    else:
        return np.array([0])
