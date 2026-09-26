"""
normalize.py — Text normalization for business entity resolution.

Generates normalized fields and multiple blocking keys used by all
three blocking layers (deterministic, TF-IDF, dense).
"""

import re
import unicodedata
from typing import Optional


# ── Honorific / generic business prefix patterns ────────────────────────────
_HONORIFIC_PREFIXES = re.compile(
    r"^(shri|sri|shree|m/s|ms|dr|doctor|prof|the)\b\s*",
    re.IGNORECASE,
)

# ── Legal suffix patterns (English + common Indian/French/US variants) ──────
_LEGAL_SUFFIXES = re.compile(
    r"\b("
    r"pvt\.?\s*ltd\.?|private\s+limited|private\s+ltd\.?|"
    r"p\.?\s*ltd\.?|llp|llc|inc\.?|corp\.?|corporation|incorporated|"
    r"limited|ltd\.?|co\.?|company|enterprises?|"
    r"industries|industry|group|holding|holdings|"
    r"trading|traders?|distributors?|solutions?|"
    r"technologies|technology|tech|services?|"
    r"international|intl\.?|plc|gmbh|"
    r"s\.a\.s|s\.a\.|sarl|sas|eurl|srl|"          # French / EU legal
    r"proprietorship|proprietor|prop\.?|"
    r"& sons|and sons|brothers|bros\.?"
    r")\b",
    re.IGNORECASE,
)

# ── Punctuation / noise patterns ─────────────────────────────────────────────
_PUNCT = re.compile(r"[^\w\s]")
_MULTI_SPACE = re.compile(r"\s+")

# Common abbreviation expansions (applied before normalization)
_ABBREV_MAP = {
    r"\bct\.?\b": "court",
    r"\brd\.?\b": "road",
    r"\bst\.?\b": "street",
    r"\bdr\.?\b": "drive",
    r"\bpl\.?\b": "place",
    r"\bave\.?\b|\bav\.?\b": "avenue",
    r"\bblvd\.?\b": "boulevard",
    r"\bln\.?\b": "lane",
    r"\bcir\.?\b": "circle",
    r"\bpkwy\.?\b": "parkway",
    r"\bhwy\.?\b": "highway",
    r"\bflr\.?\b|\bfl\.?\b": "floor",
    r"\bapt\.?\b": "apartment",
    r"\bste\.?\b": "suite",
    r"\bbldg\.?\b": "building",
    r"\bh\.?\s*no\.?\b": "house number",
    r"\bplot\.?\s*no\.?\b": "plot number",
    r"\bflat\.?\s*no\.?\b": "flat",
    r"\bnagar\b|\bng\.?\b": "nagar",
    r"\bm\.?\s*g\.?\b": "mahatma gandhi",
    r"\&": "and",
    r"\bno\.?\b|\bnos\.?\b": "number",
}
_ABBREV_PATTERNS = [(re.compile(p, re.IGNORECASE), r) for p, r in _ABBREV_MAP.items()]


def unicode_normalize(text: str) -> str:
    """NFKD-normalize and strip diacritics / non-ascii characters cleanly."""
    return unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")


def normalize_text(text: str, expand_abbrevs: bool = False) -> str:
    """
    Core normalization pipeline:
      1. Unicode / encoding artifact cleanup → ASCII
      2. Lowercase
      3. Optional abbreviation expansion
      4. Remove punctuation
      5. Collapse whitespace
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = unicode_normalize(text)
    text = text.lower().strip()
    if expand_abbrevs:
        for pattern, replacement in _ABBREV_PATTERNS:
            text = pattern.sub(replacement, text)
    text = _PUNCT.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text


def normalize_name(name: str) -> str:
    """Normalize business name: strip honorifics, strip legal suffixes."""
    if not isinstance(name, str):
        return ""
    n = normalize_text(name)
    n = _HONORIFIC_PREFIXES.sub("", n)
    n = _LEGAL_SUFFIXES.sub(" ", n)
    n = _MULTI_SPACE.sub(" ", n).strip()
    return n


def normalize_address(addr: str) -> str:
    """Normalize address with abbreviation expansion."""
    if not isinstance(addr, str):
        return ""
    return normalize_text(addr, expand_abbrevs=True)


def normalize_country(country: str) -> str:
    if not isinstance(country, str):
        return "unknown"
    return country.strip().lower()


# ── Blocking Key Generators ───────────────────────────────────────────────────

def key_name_prefix(norm_name: str, n: int = 5) -> str:
    """First n chars of normalized name (no spaces)."""
    compact = norm_name.replace(" ", "")
    return compact[:n]


def key_name_tokens_sorted(norm_name: str) -> str:
    """Sorted tokens of the name — handles word-order transpositions."""
    tokens = sorted(norm_name.split())
    return " ".join(tokens[:4])      # cap at 4 tokens to keep key stable


def key_country_nameprefix(country: str, norm_name: str, n: int = 4) -> str:
    return normalize_country(country) + "|" + key_name_prefix(norm_name, n)


def key_country_addr_prefix(country: str, norm_addr: str, n: int = 5) -> str:
    compact = norm_addr.replace(" ", "")
    return normalize_country(country) + "|" + compact[:n]


def key_name_addr_prefix(norm_name: str, norm_addr: str, nn: int = 3, na: int = 3) -> str:
    """Combined name+address prefix — discriminative for city-level matching."""
    np = norm_name.replace(" ", "")[:nn]
    ap = norm_addr.replace(" ", "")[:na]
    return np + "|" + ap


def build_blocking_keys(row) -> list[str]:
    """
    Return all deterministic blocking keys for one record row.
    Row must have: business_name, business_address, country
    """
    name   = normalize_name(str(row.get("business_name", "")))
    addr   = normalize_address(str(row.get("business_address", "")))
    country = normalize_country(str(row.get("country", "")))

    keys = []

    # Key A: country + first 4 chars of name
    ka = key_country_nameprefix(country, name, 4)
    if len(ka) > 3:
        keys.append(ka)

    # Key B: country + first 5 chars of address
    kb = key_country_addr_prefix(country, addr, 5)
    if len(kb) > 3:
        keys.append(kb)

    # Key C: first 6 chars of name only (catches cross-country duplicates)
    kc = key_name_prefix(name, 6)
    if len(kc) >= 3:
        keys.append(kc)

    # Key D: sorted tokens (handles word-order swaps)
    kd = key_name_tokens_sorted(name)
    if kd:
        keys.append(kd)

    # Key E: name prefix + address prefix (3+3)
    ke = key_name_addr_prefix(name, addr, 3, 3)
    if len(ke) > 4:
        keys.append(ke)

    return keys


def build_combined_text(row, weight_name: int = 2) -> str:
    """
    Build a single text string for TF-IDF / embedding encoding.
    Repeats name `weight_name` times to give it more importance.
    """
    name = normalize_name(str(row.get("business_name", "")))
    addr = normalize_address(str(row.get("business_address", "")))
    parts = [name] * weight_name + [addr]
    return " ".join(p for p in parts if p)
