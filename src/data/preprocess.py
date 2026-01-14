import logging

import pandas as pd


logger = logging.getLogger("experiment")


def utf16_to_unicode_index(text, utf16_index):
    unicode_index = 0
    current_utf16_index = 0

    while current_utf16_index < utf16_index and unicode_index < len(text):
        current_utf16_index += 2 if ord(text[unicode_index]) >= 0x10000 else 1
        unicode_index += 1

    return unicode_index


def fix_dataset_NER(dataset):
    for idx, row in dataset.iterrows():
        ner_elements = dataset.loc[
            idx, "NER"
        ]  # {end, type, start,source, selectedText}[]
        dataset.loc[idx, "NER"] = []
        for element in ner_elements:
            tag = element["type"]
            source = (
                row["Title"] if element["source"] == "title" else row["Description"]
            )
            start = utf16_to_unicode_index(source, element["start"])
            end = utf16_to_unicode_index(source, element["end"])
            added = {
                "start": start,
                "end": end,
                "source": element["source"],
                "entry": element["selectedText"],
                "type": tag,
            }
            dataset.loc[idx, "NER"].append(added)
    return dataset


def validate_ner_indices(dataset):
    for idx, row in dataset.iterrows():
        title = row["Title"]
        description = row["Description"]
        for entity in row["NER"]:
            source = title if entity["source"] == "title" else description
            start = entity["start"]
            end = entity["end"]
            selected_text = entity["entry"]
            extracted_text = source[start:end]
            if extracted_text != selected_text:
                logger.warning(
                    f"Mismatch in row {idx}: expected '{selected_text}', got '{extracted_text}'"
                )
    logger.info("NER indices validation completed.")


def preprocess_dataset():
    dataset = pd.read_json("dataset/videos.json")
    dataset = fix_dataset_NER(dataset)
    print(dataset.head())
    validate_ner_indices(dataset)

    dataset.to_json("dataset/p2_dataset.json", indent=False)
