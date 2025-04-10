import numpy as np

from tokenizer import TokenizerCustom


def utf16_to_unicode_index(text, utf16_index):
    unicode_index = 0
    current_utf16_index = 0

    while current_utf16_index < utf16_index and unicode_index < len(text):
        current_utf16_index += 2 if ord(text[unicode_index]) >= 0x10000 else 1
        unicode_index += 1

    return unicode_index


def fix_dataset_NER(dataset):
    for idx, row in dataset.iterrows():
        for tag, enity in dataset.loc[idx, "NER"].items():
            dataset.loc[idx, "NER"][tag] = []
            for start, end, entry in enity:
                start = utf16_to_unicode_index(row["Title"], start)
                end = utf16_to_unicode_index(row["Title"], end)
                dataset.loc[idx, "NER"][tag].append([start, end, entry])
    return dataset


def convert_ner_tags(titles, ner_dict: list[dict]):
    numeric_tags = []
    custom_tokenizer = TokenizerCustom()
    for i, title in enumerate(titles):
        token = custom_tokenizer.encode(title)
        tags = [0] * len(title)

        for name, values in ner_dict[i].items():
            for value in values:
                start, end, entry = value
                if name == "Artist":
                    tags[start:end] = [1] * (end - start)
                if name == "Title":
                    tags[start:end] = [2] * (end - start)

        new_tags = [0] * len(token)
        for j, token_e in enumerate(token):
            token_start = title.find(token_e)
            token_end = token_start + len(token_e)
            title = title[:token_start] + " " * \
                (token_end - token_start) + title[token_end:]
            idx = np.round(np.average(
                [i for i in tags[token_start:token_end] if i != 0]))
            if idx == 1:
                new_tags[j] = "I-Artist"
            elif idx == 2:
                new_tags[j] = "I-Title"
            else:
                new_tags[j] = "O"
        numeric_tags.append(new_tags)

    return numeric_tags


def split_dataset(dataset, fraction=0.8, random_state=42):
    train_df = dataset.sample(frac=fraction, random_state=random_state)
    validation_df = dataset.drop(train_df.index)
    train_df = train_df.reset_index(drop=True)
    validation_df = validation_df.reset_index(drop=True)
    return train_df, validation_df
