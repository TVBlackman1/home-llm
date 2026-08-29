from __future__ import annotations


def normalize_text(text: str) -> str:
    folded = text.replace("ё", "е").replace("Ё", "Е").casefold()
    return " ".join(folded.split())
