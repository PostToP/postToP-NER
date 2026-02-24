TABLE = {
    "O": 0,
    "ORIGINAL_AUTHOR": 1,
    "TITLE": 2,
    "MODIFIER": 3,
    "VOCALOID": 4,
    "ALBUM": 5,
    "MISC_PERSON": 6,
    "VOCALIST": 7,
    "ALT_TITLE": 8,
    "FEATURING": 9,
}
TABLE_BACK = {v: k for k, v in TABLE.items()}

NUM_LABELS = len(TABLE)

MAX_SEQUENCE_LENGTH = 512

TRANSFORMER_MODEL_NAME = "distilbert/distilbert-base-multilingual-cased"
VERSION = "v1.0.0"
