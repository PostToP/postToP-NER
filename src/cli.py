import sys

import log


def run_fetch() -> None:
    from data.database import main as fetch_videos

    fetch_videos()


def run_preprocess() -> None:
    from data.preprocess import preprocess_dataset

    preprocess_dataset()


def run_split() -> None:
    from data.split_dataset import split_dataset

    split_dataset()


def run_feature_extraction() -> None:
    from model.feature_extraction import feature_extraction

    feature_extraction()


def run_train() -> None:
    from model.train import main as train_model

    train_model()


def run_tokenize() -> None:
    from model.tokenize import tokenize_dataset

    tokenize_dataset()


def main() -> None:
    COMMANDS = {
        "fetch": run_fetch,
        "split": run_split,
        "tokenize": run_tokenize,
        "preprocess": run_preprocess,
        "feature": run_feature_extraction,
        "train": run_train,
    }

    if len(sys.argv) < 2:
        print("Usage: python cli.py <operations>")
        print(f"Available operations: {', '.join(COMMANDS.keys())}")
        return

    commands = sys.argv[1:]

    for i in commands:
        if i not in COMMANDS:
            print(f"Unknown command: {i}")
            return

    while commands:
        command = commands.pop(0)
        COMMANDS[command]()


if __name__ == "__main__":
    main()
