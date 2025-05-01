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


def mask_artists_and_titles(
    tokens: list[list[str]], ner_tags: list[list[str]]
) -> list[list[str]]:
    """
    Mask tokens thats part of any 'Inside' tag.
    If a token was ever tagged with an 'Outside' tag, it will not be masked.

    Args:
        tokens (list[list[str]]): List of tokens
        ner_tags (list[list[str]]): List of NER tags

    Returns:
        list[list[str]]: List of tokens with masked tokens
    """
    not_to_mask = set()
    for t, tags in zip(tokens, ner_tags):
        for i, tag in enumerate(tags):
            if tag == "O":
                not_to_mask.add(t[i])

    new_tokens = []
    for token in tokens:
        new_tokens.append([i if i in not_to_mask else "<oov>" for i in token])
    return new_tokens
