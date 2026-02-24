import traceback
from typing import List, Tuple, Dict

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer

from config.config import TABLE_BACK, TRANSFORMER_MODEL_NAME
from model.ModelWrapper import ModelWrapper


model_wrapper = ModelWrapper.deserialize("model/compiled_model.tar.gz")
model_wrapper.warmup()

tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)


def unicode_to_utf16_index(text, unicode_index):
    utf16_index = 0
    for i in range(unicode_index):
        utf16_index += 2 if ord(text[i]) >= 0x10000 else 1
    return utf16_index


def extract_entities(
    title: str, description: str
) -> Tuple[List[Tuple], Dict[str, List[str]]]:
    text = title + " [SEP] " + description

    logits = model_wrapper.predict(title, description)
    predictions = np.argmax(logits, axis=-1)[0]

    encoding = tokenizer(
        text,
        padding="longest",
        truncation=True,
        max_length=512,
        return_tensors="np",
        return_offsets_mapping=True,
    )

    word_ids = encoding.word_ids()
    offsets = encoding["offset_mapping"].squeeze(0).tolist()

    entities = []
    current_entity = None
    previous_word_idx = None

    id2label = TABLE_BACK
    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue

        if word_idx != previous_word_idx:
            pred_id = predictions[idx]
            tag = id2label[pred_id]

            token_start, token_end = offsets[idx]

            if tag != "O":
                if current_entity and tag == current_entity["tag"]:
                    pass

                else:
                    if current_entity:
                        entities.append(
                            (
                                current_entity["tag"],
                                text[current_entity["start"] : current_entity["end"]],
                                current_entity["start"],
                                current_entity["end"],
                            )
                        )

                    current_entity = {
                        "tag": tag,
                        "start": token_start,
                        "end": token_end,
                    }
            else:
                if current_entity:
                    entities.append(
                        (
                            current_entity["tag"],
                            text[current_entity["start"] : current_entity["end"]],
                            current_entity["start"],
                            current_entity["end"],
                        )
                    )
                    current_entity = None

            if current_entity:
                cursor = idx
                while cursor + 1 < len(word_ids) and word_ids[cursor + 1] == word_idx:
                    cursor += 1

                true_word_end = offsets[cursor][1]
                current_entity["end"] = true_word_end

        previous_word_idx = word_idx

    if current_entity:
        entities.append(
            (
                current_entity["tag"],
                text[current_entity["start"] : current_entity["end"]],
                current_entity["start"],
                current_entity["end"],
            )
        )

    result = {
        "original_authors": filter_unique_entities(entities, "ORIGINAL_AUTHOR"),
        "title": filter_unique_entities(entities, "TITLE"),
        "featuring": filter_unique_entities(entities, "FEATURING"),
        "modifier": filter_unique_entities(entities, "MODIFIER"),
        "vocaloid": filter_unique_entities(entities, "VOCALOID"),
        "misc_person": filter_unique_entities(entities, "MISC_PERSON"),
        "performer": filter_unique_entities(entities, "VOCALIST"),
        "alt_title": filter_unique_entities(entities, "ALT_TITLE"),
        "album": filter_unique_entities(entities, "ALBUM"),
    }

    entities = [
        (
            tag,
            entity_text,
            unicode_to_utf16_index(text, start_char),
            unicode_to_utf16_index(text, end_char),
        )
        for tag, entity_text, start_char, end_char in entities
    ]

    return entities, result


def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def filter_unique_entities(decoded_tags: List[Tuple], tag_to_filter: str) -> List[str]:
    entities = []
    for tag, entity, _, _ in decoded_tags:
        if tag == tag_to_filter:
            entities.append(entity)

    if not entities:
        return []

    # Filter duplicates using TF-IDF similarity
    unique_entities = entities
    vectorizer = TfidfVectorizer(analyzer="char", lowercase=True)
    vectorizer.fit(unique_entities)
    vectors = vectorizer.transform(unique_entities)
    similarity_matrix = cosine_similarity(vectors)
    threshold = 0.5
    to_remove = set()
    for i in range(len(unique_entities)):
        for j in range(i + 1, len(unique_entities)):
            if similarity_matrix[i, j] > threshold:
                to_remove.add(j)
    unique_entities = [
        entity for idx, entity in enumerate(unique_entities) if idx not in to_remove
    ]

    # Further filter using Levenshtein distance
    filtered_entities = []
    for entity in unique_entities:
        if all(levenshtein_distance(entity, other) > 3 for other in filtered_entities):
            filtered_entities.append(entity)

    return filtered_entities


app = Flask(__name__)
CORS(app)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        title = data.get("title", "")
        description = data.get("description", "")
        entities, structured_result = extract_entities(title, description)
        return jsonify(
            {
                "prediction": {"entities": entities, "result": structured_result},
                "version": model_wrapper.version,
            }
        )
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


def run_server() -> None:
    app.run(host="0.0.0.0", port=5000)


if __name__ == "__main__":
    run_server()
