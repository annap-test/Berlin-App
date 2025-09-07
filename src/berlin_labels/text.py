from __future__ import annotations

import math
from typing import Iterable, List, Set


UMLAUT_MAP = {
    "ä": "ae",
    "ö": "oe",
    "ü": "ue",
    "Ä": "ae",
    "Ö": "oe",
    "Ü": "ue",
    "ß": "ss",
}


def _fold_umlauts(s: str) -> str:
    for k, v in UMLAUT_MAP.items():
        s = s.replace(k, v)
    return s


def canon_nh(name: str) -> str:
    """Canonicalize neighborhood/district names for robust joins.

    - Lowercase, trim
    - Replace German umlauts/ß to ASCII
    - Remove non-alphanumeric characters
    """
    if name is None:
        return ""
    s = _fold_umlauts(str(name).strip().lower())
    # keep alnum only
    s = "".join(ch for ch in s if ch.isalnum())
    return s


# National cuisine vocabulary (English tokens)
national_cuisine_vocab: Set[str] = {
    # Europe
    "italian",
    "french",
    "spanish",
    "portuguese",
    "greek",
    "turkish",
    "german",
    "polish",
    "russian",
    "ukrainian",
    "balkan",
    "hungarian",
    "romanian",
    "bulgarian",
    "georgian",
    # Americas
    "mexican",
    "argentinian",
    "peruvian",
    "brazilian",
    "colombian",
    "venezuelan",
    "caribbean",
    "american",
    "texmex",
    # Middle East & Africa
    "lebanese",
    "israeli",
    "palestinian",
    "syrian",
    "iraqi",
    "iranian",
    "afghan",
    "moroccan",
    "tunisian",
    "algerian",
    "ethiopian",
    "eritrean",
    "egyptian",
    "southafrican",
    "nigerian",
    # Asia
    "indian",
    "pakistani",
    "bangladeshi",
    "srilankan",
    "nepali",
    "chinese",
    "japanese",
    "korean",
    "thai",
    "vietnamese",
    "laotian",
    "cambodian",
    "indonesian",
    "malaysian",
    "singaporean",
    "filipino",
}


_EXCLUDE_TOKENS: Set[str] = {
    # dishes / forms rather than national cuisines
    "pizza",
    "pasta",
    "sushi",
    "ramen",
    "doner",
    "döner",
    "kebab",
    "burger",
    "bbq",
    "grill",
    "steak",
    "noodles",
    "dumpling",
    "dumplings",
    "sandwich",
    "bakery",
    "cafe",
    "coffee",
    "bubbletea",
    "boba",
    "falafel",
}


def _normalize_token(tok: str) -> str:
    t = _fold_umlauts(tok.strip().lower())
    t = "".join(ch for ch in t if ch.isalnum())
    return t


def tokenize_cuisines(cuisines: str | Iterable[str]) -> List[str]:
    """Tokenize a semicolon-separated cuisine string into lowercase tokens.

    Keeps only tokens present in the national_cuisine_vocab, excluding dish words.
    """
    if cuisines is None:
        return []
    # Handle NaN/NaT floats gracefully
    if isinstance(cuisines, float):
        if math.isnan(cuisines):
            return []
        # treat other floats as invalid/non-iterable
        return []
    if isinstance(cuisines, str):
        parts = [p for p in cuisines.split(";")]
    else:
        # already iterable? if not, return []
        try:
            parts = list(cuisines)
        except TypeError:
            return []
    toks: List[str] = []
    for p in parts:
        t = _normalize_token(str(p))
        if not t or t in _EXCLUDE_TOKENS:
            continue
        if t in national_cuisine_vocab:
            toks.append(t)
    return toks


def nationals_set(cuisines: str | Iterable[str]) -> Set[str]:
    """Return a set of national cuisine tokens from a cuisine string."""
    return set(tokenize_cuisines(cuisines))

