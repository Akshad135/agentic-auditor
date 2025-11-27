import unicodedata
import re


def clean_text(text: str) -> str:
    """Normalize text by fixing unicode artifacts and whitespace."""
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)

    # Normalize dashes and quotes
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    text = re.sub(r"[\u2018\u2019]", "'", text)
    text = re.sub(r"[\u201c\u201d]", '"', text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text
