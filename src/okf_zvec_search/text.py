from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

import yaml

FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


TOKEN_RE = re.compile(r"[\wА-Яа-яЁё-]+", re.UNICODE)


_MORPH: Any | None = None


_MORPH_UNAVAILABLE = False


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}, text

    loaded = yaml.safe_load(match.group(1)) or {}
    meta = loaded if isinstance(loaded, dict) else {}
    return meta, text[match.end() :]


def metadata_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def metadata_tags(value: Any) -> list[str]:
    if isinstance(value, list):
        return [metadata_text(item) for item in value if metadata_text(item)]
    if isinstance(value, str):
        stripped = value.strip().strip("[]")
        return [item.strip().strip("'\"") for item in stripped.split(",") if item.strip()]
    return []


def title_from_markdown(body: str, fallback: str) -> str:
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or fallback
    return fallback


def split_block_by_bullets(block: str) -> list[str]:
    lines = block.splitlines()
    chunks: list[str] = []
    current: list[str] = []

    for line in lines:
        if line.startswith("- "):
            if current:
                chunks.append("\n".join(current).strip())
            current = [line]
        elif current:
            current.append(line)
        elif line.strip():
            chunks.append(line.strip())

    if current:
        chunks.append("\n".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def section_chunks(body: str) -> list[tuple[str, str]]:
    matches = list(HEADING_RE.finditer(body))
    if not matches:
        stripped = body.strip()
        return [("", stripped)] if stripped else []

    chunks: list[tuple[str, str]] = []
    preface = body[: matches[0].start()].strip()
    if preface:
        chunks.append(("", preface))

    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        heading = match.group(2).strip()
        heading_line = match.group(0).strip()
        block = body[start:end].strip()
        if not block:
            chunks.append((heading, heading_line))
            continue

        bullet_chunks = split_block_by_bullets(block)
        if len(bullet_chunks) <= 1:
            chunks.append((heading, f"{heading_line}\n\n{block}".strip()))
        else:
            for bullet in bullet_chunks:
                chunks.append((heading, f"{heading_line}\n\n{bullet}".strip()))
    return chunks


def get_morph() -> Any | None:
    global _MORPH, _MORPH_UNAVAILABLE
    if _MORPH is not None:
        return _MORPH
    if _MORPH_UNAVAILABLE:
        return None
    try:
        from pymorphy3 import MorphAnalyzer
    except ImportError:
        _MORPH_UNAVAILABLE = True
        return None
    _MORPH = MorphAnalyzer()
    return _MORPH


@lru_cache(maxsize=8192)
def token_lemma(token: str) -> str:
    normalized = token.casefold()
    morph = get_morph()
    if morph is None or not re.search(r"[а-яё]", normalized):
        return normalized
    return morph.parse(normalized)[0].normal_form


def lexical_tokens(text: str) -> set[str]:
    return {token.casefold() for token in TOKEN_RE.findall(text) if len(token) >= 3}


def normalize_fts_text(text: str) -> str:
    return " ".join(token_lemma(token) for token in TOKEN_RE.findall(text.casefold()))


def normalize_raw_fts_text(text: str) -> str:
    return " ".join(token.casefold() for token in TOKEN_RE.findall(text))


def matching_terms(query: str, fields: dict[str, Any]) -> list[str]:
    query_lemmas = {token_lemma(token) for token in lexical_tokens(query)}
    if not query_lemmas:
        return []
    text = "\n".join(str(fields.get(name, "")) for name in ("title", "heading", "path", "text"))
    matched: dict[str, str] = {}
    for token in TOKEN_RE.findall(text):
        normalized = token.casefold()
        if token_lemma(normalized) in query_lemmas:
            matched.setdefault(normalized, token)
    return sorted(matched.values(), key=str.casefold)
