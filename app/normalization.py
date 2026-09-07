import unicodedata


def normalize_for_search(text: str) -> str:
    return unicodedata.normalize("NFKC", text)
