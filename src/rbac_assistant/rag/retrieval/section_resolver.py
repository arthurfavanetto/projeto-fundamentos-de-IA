"""Resolver determinístico de referências estruturais normativas (ANAC / resoluções / IS)."""

from __future__ import annotations

import re

from rbac_assistant.domain.entities.rag import ResolvedStructuralRef

# Ex.: "RBAC 43.3", "RBAC nº 43.9(a)(1)", "RBAC 43 seção 43.3(b)", "43.1-I"
_RBAC_SECTION_RE = re.compile(
    r"\bRBAC\s*(?:n[º°.]?\s*)?(?P<reg>\d{1,3})\s*"
    r"(?:(?:,\s*)?(?:se[çc][ãa]o|sec\.?)\s*)?"
    r"(?P<sec>(?P=reg)\.\d+[A-Za-z0-9\-]*)\s*"
    r"(?:\((?P<par>[a-z])\))?\s*(?:\((?P<item>\d+|[ivx]+)\))?",
    re.IGNORECASE,
)

# Ex.: "RBAC 43.3" (forma curta onde o número do regulamento está no prefixo da seção)
_RBAC_DOT_RE = re.compile(
    r"\bRBAC\s*(?:n[º°.]?\s*)?(?P<sec>(?P<reg>\d{1,3})\.\d+[A-Za-z0-9\-]*)"
    r"(?:\((?P<par>[a-z])\))?(?:\((?P<item>\d+|[ivx]+)\))?",
    re.IGNORECASE,
)

# Ex.: "seção 43.3", "seção 145.109(a)", ou token isolado "43.3(a)"
_STANDALONE_SECTION_RE = re.compile(
    r"(?:\bse[çc][ãa]o\s+)?\b(?P<sec>(?P<reg>\d{1,3})\.\d+[A-Za-z0-9\-]*)"
    r"(?:\((?P<par>[a-z])\))?(?:\((?P<item>\d+|[ivx]+)\))?\b",
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

# Ex.: "Emenda 05", "Emenda nº 04", "EMD 05"
_AMENDMENT_RE = re.compile(
    r"\b(?:emenda|emd\.?)\s*(?:n[º°.]?\s*)?(?P<amd>\d{1,3}[A-Za-z]?)\b",
    re.IGNORECASE,
)


def extract_amendment_hint(query: str) -> str | None:
    """Detecta menção explícita de emenda na pergunta (ex.: 'Emenda 04')."""
    match = _AMENDMENT_RE.search(query)
    return match.group("amd") if match else None


def resolve_structural_references(query: str) -> tuple[ResolvedStructuralRef, ...]:
    """Extrai referências estruturais explícitas da consulta em ordem de prioridade."""
    amendment = extract_amendment_hint(query)
    refs: list[ResolvedStructuralRef] = []
    seen: set[tuple[str | None, str | None, str | None]] = set()

    def _add(
        raw: str,
        reg: str | None,
        sec: str | None,
        par: str | None = None,
        item: str | None = None,
    ) -> None:
        norm_sec = sec.strip() if sec else None
        norm_reg = reg.strip() if reg else None
        norm_par = f"({par.lower()})" if par else None
        norm_item = f"({item})" if item else None
        key = (norm_reg, norm_sec.lower() if norm_sec else None, norm_par)
        if norm_sec and key not in seen:
            seen.add(key)
            refs.append(
                ResolvedStructuralRef(
                    raw_match=raw.strip(),
                    regulation_number=norm_reg,
                    section_number=norm_sec,
                    paragraph_number=norm_par,
                    item_number=norm_item,
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
            )

    for match in _APPENDIX_RE.finditer(query):
        sec_clean = re.sub(r"\s+", " ", match.group("sec")).strip()
        _add(match.group(0), match.group("reg"), sec_clean)

    for match in _ARTICLE_RE.finditer(query):
        sec_raw = re.sub(r"^Artigo\b", "Art.", match.group("sec"), flags=re.IGNORECASE)
        sec_clean = re.sub(r"\s+", " ", sec_raw).strip()
        _add(match.group(0), match.group("reg"), sec_clean)

    return tuple(refs)
