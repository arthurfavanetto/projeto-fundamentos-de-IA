"""Resolver determinístico de referências estruturais normativas (ANAC / resoluções / IS)."""

from __future__ import annotations

import re

from rbac_assistant.domain.entities.rag import ResolvedStructuralRef

# Ex.: "RBAC 43.3", "RBAC nº 43.9(a)(1)(i)", "RBAC 43 seção 43.3(b)", "43.1-I"
_RBAC_SECTION_RE = re.compile(
    r"\bRBAC\s*(?:n[º°.]?\s*)?(?P<reg>\d{1,3})\s*"
    r"(?:(?:,\s*)?(?:se[çc][ãa]o|sec\.?)\s*)?"
    r"(?P<sec>(?P=reg)\.\d+[A-Za-z0-9\-]*)\s*"
    r"(?:\((?P<par>[a-z])\))?\s*"
    r"(?:\((?P<item>\d+|[ivx]+)\))?\s*"
    r"(?:\((?P<subitem>[ivx]+|[A-Z]|\d+)\))?",
    re.IGNORECASE,
)

# Ex.: "RBAC 43.3", "RBAC 43.9(a)(1)(i)" (forma curta)
_RBAC_DOT_RE = re.compile(
    r"\bRBAC\s*(?:n[º°.]?\s*)?(?P<sec>(?P<reg>\d{1,3})\.\d+[A-Za-z0-9\-]*)"
    r"(?:\((?P<par>[a-z])\))?"
    r"(?:\((?P<item>\d+|[ivx]+)\))?"
    r"(?:\((?P<subitem>[ivx]+|[A-Z]|\d+)\))?",
    re.IGNORECASE,
)

# Ex.: "seção 43.3", "seção 145.109(a)(1)(i)", ou token isolado "43.3(a)(1)"
_STANDALONE_SECTION_RE = re.compile(
    r"(?:\bse[çc][ãa]o\s+)?\b(?P<sec>(?P<reg>\d{1,3})\.\d+[A-Za-z0-9\-]*)"
    r"(?:\((?P<par>[a-z])\))?"
    r"(?:\((?P<item>\d+|[ivx]+)\))?"
    r"(?:\((?P<subitem>[ivx]+|[A-Z]|\d+)\))?",
    re.IGNORECASE,
)

# Ex.: "Apêndice A", "Apêndice B do RBAC 43"
_APPENDIX_RE = re.compile(
    r"\b(?P<sec>Ap[êe]ndice\s+[A-Z0-9]+(?:\s*-\s*[A-Z0-9]+)?)"
    r"(?:\s+(?:do|da|de)\s+RBAC\s*(?:n[º°.]?\s*)?(?P<reg>\d{1,3}))?",
    re.IGNORECASE,
)

# Ex.: "Art. 1º", "Artigo 2º da Resolução 700"
_ARTICLE_RE = re.compile(
    r"\b(?P<sec>Art\.?\s*\d+[º°]?)"
    r"(?:\s+(?:da|do|de)\s+(?:Resolu[çc][ãa]o|Portaria|IS)\s*(?:n[º°.]?\s*)?(?P<reg>[A-Za-z0-9.\-]+))?",
    re.IGNORECASE,
)

# Ex.: "Emenda 05", "Emenda nº 04", "EMD 5"
_AMENDMENT_RE = re.compile(
    r"\b(?:emenda|emd\.?)\s*(?:n[º°.]?\s*)?(?P<amd>\d{1,3}[A-Za-z]?)\b",
    re.IGNORECASE,
)

# Ex.: "RBAC 43", "RBAC nº 145", "Resolução 700", "IS 43-001" (mesmo sem seção)
_REGULATION_MENTION_RE = re.compile(
    r"\b(?P<kind>RBAC|Resolu[çc][ãa]o|IS|Portaria)\s*(?:n[º°.]?\s*)?"
    r"(?P<num>\d{1,4}(?:-[A-Za-z0-9]+)?)\b",
    re.IGNORECASE,
)


def normalize_amendment_number(value: str) -> str:
    """Normaliza zeros à esquerda de emendas puramente numéricas (ex.: '05' -> '5')."""
    cleaned = value.strip().lower()
    if cleaned.isdigit():
        return str(int(cleaned))
    return cleaned


def amendments_match(left: str, right: str) -> bool:
    """Compara identificadores de emenda tratando zeros à esquerda ('05' == '5')."""
    return normalize_amendment_number(left) == normalize_amendment_number(right)


def extract_amendment_hint(query: str) -> str | None:
    """Detecta menção explícita de emenda na pergunta (ex.: 'Emenda 04')."""
    match = _AMENDMENT_RE.search(query)
    return match.group("amd") if match else None


def extract_regulation_hint(query: str) -> tuple[str | None, str | None]:
    """Detecta menção explícita e inequívoca a um único regulamento na pergunta.

    Retorna `(document_type_value, regulation_number)` somente quando há um único
    regulamento citado na consulta; se houver comparação entre dois regulamentos
    (ex.: 'RBAC 43 e RBAC 145'), retorna `(None, None)` para não restringir indevidamente.
    """
    found: set[tuple[str, str]] = set()
    for match in _REGULATION_MENTION_RE.finditer(query):
        raw_kind = match.group("kind").lower()
        if raw_kind == "rbac":
            doc_type = "RBAC"
        elif raw_kind.startswith("resolu"):
            doc_type = "RESOLUCAO"
        elif raw_kind == "is":
            doc_type = "IS"
        else:
            doc_type = "PORTARIA"
        found.add((doc_type, match.group("num").strip()))

    if len(found) == 1:
        return next(iter(found))
    return (None, None)


def resolve_structural_references(query: str) -> tuple[ResolvedStructuralRef, ...]:
    """Extrai referências estruturais explícitas da consulta em ordem de prioridade."""
    amendment = extract_amendment_hint(query)
    refs: list[ResolvedStructuralRef] = []
    seen: set[tuple[str | None, str | None, str | None, str | None, str | None]] = set()

    def _add(
        raw: str,
        reg: str | None,
        sec: str | None,
        par: str | None = None,
        item: str | None = None,
        subitem: str | None = None,
    ) -> None:
        norm_sec = sec.strip() if sec else None
        norm_reg = reg.strip() if reg else None
        norm_par = f"({par.lower()})" if par else None
        norm_item = f"({item.lower()})" if item else None
        norm_subitem = f"({subitem})" if subitem else None
        key = (
            norm_reg,
            norm_sec.lower() if norm_sec else None,
            norm_par,
            norm_item,
            norm_subitem.lower() if norm_subitem else None,
        )
        if norm_sec and key not in seen:
            seen.add(key)
            refs.append(
                ResolvedStructuralRef(
                    raw_match=raw.strip(),
                    regulation_number=norm_reg,
                    section_number=norm_sec,
                    paragraph_number=norm_par,
                    item_number=norm_item,
                    subitem_number=norm_subitem,
                    amendment_number=amendment,
                )
            )

    for pattern in (_RBAC_DOT_RE, _RBAC_SECTION_RE, _STANDALONE_SECTION_RE):
        for match in pattern.finditer(query):
            _add(
                match.group(0),
                match.group("reg"),
                match.group("sec"),
                match.group("par"),
                match.group("item"),
                match.group("subitem"),
            )

    for match in _APPENDIX_RE.finditer(query):
        sec_clean = re.sub(r"\s+", " ", match.group("sec")).strip()
        _add(match.group(0), match.group("reg"), sec_clean)

    for match in _ARTICLE_RE.finditer(query):
        sec_raw = re.sub(r"^Artigo\b", "Art.", match.group("sec"), flags=re.IGNORECASE)
        sec_clean = re.sub(r"\s+", " ", sec_raw).strip()
        _add(match.group(0), match.group("reg"), sec_clean)

    return tuple(refs)
