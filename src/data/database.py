import json
import logging
import os
import pandas as pd

import psycopg2
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("experiment")


def get_connection() -> psycopg2.extensions.connection:
    return psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
    )


def fetch_videos() -> list[tuple]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""SELECT
    v.yt_id,
    nr.language,
    title,
    description,
    ner_result,
    c.name as channel_name
from
    video_metadata vm
    INNER join ner_result nr on nr.video_id = vm.video_id
    and nr.language = vm.language
    INNER join video v on vm.video_id = v.id
    INNER JOIN channel c on c.id = v.channel_id""")
    videos = cursor.fetchall()
    cursor.close()
    conn.close()
    return videos


def convert_postgres_videos_to_json(videos: list[tuple]) -> list[dict]:
    all_vids = []
    for video in videos:
        v = {}
        v["YT ID"] = video[0]
        v["Language"] = video[1]
        v["Title"] = video[2]
        v["Description"] = video[3]
        v["NER"] = video[4]
        v["Channel Name"] = video[5]
        all_vids.append(v)
    return all_vids


def save_videos_to_json(
    videos: list[dict], filename: str = "dataset/videos.json"
) -> None:
    with open(filename, "w") as f:
        json.dump(videos, f, indent=4)


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


def main() -> None:
    videos = fetch_videos()
    if not videos:
        logger.error("No videos fetched from database")
        return

    video_json = convert_postgres_videos_to_json(videos)
    df = pd.DataFrame(video_json)
    df = fix_dataset_NER(df)
    validate_ner_indices(df)

    video_json = df.to_dict(orient="records")

    save_videos_to_json(video_json)
    logger.debug(f"Saved {len(video_json)} videos to dataset/videos.json")


if __name__ == "__main__":
    main()
