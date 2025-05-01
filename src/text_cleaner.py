from unicodedata import combining, normalize
from ftfy import fix_text
from unidecode import unidecode


def normalize_text_to_ascii(text: str) -> str:
    text = fix_text(text)
    normalized_text = normalize("NFKD", text)
    ascii_text = unidecode("".join([c for c in normalized_text if not combining(c)]))
    return ascii_text


def preprocess_tokens(tokens):
    new_tokens = []
    for token in tokens:
        token = token.lower()
        token = normalize_text_to_ascii(token)
        new_tokens.append(token)
    return new_tokens
