"""Resolver determinístico de referências estruturais normativas (ANAC / resoluções / IS)."""

from __future__ import annotations

import re

from rbac_assistant.domain.entities.rag import ResolvedStructuralRef

# Ex.: "RBAC 43.3", "RBAC nº 43.9(a)(1)(i)", "RBAC 43 seção 43.3(b)", "RBAC 43 F43.1(a)(1)"
_RBAC_SECTION_RE = re.compile(
    r"\bRBAC\s*(?:n[º°.]?\s*)?(?P<reg>\d{1,3})\s*"
    r"(?:(?:,\s*)?(?:se[çc][ãa]o|sec\.?|ap[êe]ndice)\s*)?"
    r"(?P<sec>[A-Z]?(?P=reg)\.\d+[A-Za-z0-9\-]*)\s*"
    r"(?:\((?P<par>[a-z])\))?\s*"
    r"(?:\((?P<item>\d+|[ivx]+)\))?\s*"
    r"(?:\((?P<subitem>[ivx]+|[A-Z]|\d+)\))?"
    r"(?:\((?P<leaf>[A-Z])\))?",
    re.IGNORECASE,
)

# Ex.: "RBAC 43.3", "RBAC 43.9(a)(1)(i)", "RBAC F43.1" (forma curta)
_RBAC_DOT_RE = re.compile(
    r"\bRBAC\s*(?:n[º°.]?\s*)?(?P<sec>(?:[A-Za-z](?=\d{2,3}\.))?(?P<reg>\d{1,3})\.\d+[A-Za-z0-9\-]*)"
    r"(?:\((?P<par>[a-z])\))?"
    r"(?:\((?P<item>\d+|[ivx]+)\))?"
    r"(?:\((?P<subitem>[ivx]+|[A-Z]|\d+)\))?"
    r"(?:\((?P<leaf>[A-Z])\))?",
    re.IGNORECASE,
)

# Ex.: "alínea (i) do inciso (5) do parágrafo (c) da seção 43.10"
_PT_ALINEA_OF_INCISO_RE = re.compile(
    r"\bal[íi]nea\s*\((?P<subitem>[ivx]+)\)\s+(?:do|da)\s+(?:inciso|item)\s*\((?P<item>\d+)\)\s+"
    r"(?:do|da)\s+par[áa]grafo\s*\((?P<par>[a-z])\)\s+(?:da|do)\s+"
    r"(?:se[çc][ãa]o\s+|RBAC\s*(?:n[º°.]?\s*)?)?"
    r"(?P<sec>(?:[A-Za-z](?=\d{2,3}\.))?(?P<reg>\d{1,3})\.\d+[A-Za-z0-9\-]*)",
    re.IGNORECASE,
)

# Ex.: "inciso (1) do parágrafo (a) da seção 43.10" / "item (1) do parágrafo (a) da seção 43.10"
_PT_ITEM_OF_PAR_RE = re.compile(
    r"\b(?:inciso|item)\s*\((?P<item>\d+)\)\s+(?:do|da)\s+par[áa]grafo\s*\((?P<par>[a-z])\)\s+"
    r"(?:da|do)\s+(?:se[çc][ãa]o\s+|RBAC\s*(?:n[º°.]?\s*)?)?"
    r"(?P<sec>(?:[A-Za-z](?=\d{2,3}\.))?(?P<reg>\d{1,3})\.\d+[A-Za-z0-9\-]*)",
    re.IGNORECASE,
)

# Ex.: "parágrafo (a) da seção 43.10"
_PT_PAR_OF_SEC_RE = re.compile(
    r"\bpar[áa]grafo\s*\((?P<par>[a-z])\)\s+(?:da|do)\s+"
    r"(?:se[çc][ãa]o\s+|RBAC\s*(?:n[º°.]?\s*)?)?"
    r"(?P<sec>(?:[A-Za-z](?=\d{2,3}\.))?(?P<reg>\d{1,3})\.\d+[A-Za-z0-9\-]*)",
    re.IGNORECASE,
)

# Ex.: "seção 43.3", "seção 145.109(a)(1)(i)", "43.3(a)(1)", "F43.1(a)(1)", "E43.1"
_STANDALONE_SECTION_RE = re.compile(
    r"(?:\bse[çc][ãa]o\s+)?(?<![A-Za-z0-9])"
    r"(?P<sec>(?:[A-Za-z](?=\d{2,3}\.))?(?P<reg>\d{1,3})\.\d+[A-Za-z0-9\-]*)"
    r"(?:\((?P<par>[a-z])\))?"
    r"(?:\((?P<item>\d+|[ivx]+)\))?"
    r"(?:\((?P<subitem>[ivx]+|[A-Z]|\d+)\))?"
    r"(?:\((?P<leaf>[A-Z])\))?",
    re.IGNORECASE,
)

# Ex.: "Apêndice A", "Apêndice B do RBAC 43", "Apêndice F do RBAC nº 43"
_APPENDIX_RE = re.compile(
    r"\b(?P<sec>Ap[êe]ndice\s+(?P<code>[A-Z0-9]+(?:\s*-\s*[A-Z0-9]+)?))"
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
    _, hinted_reg = extract_regulation_hint(query)
    refs: list[ResolvedStructuralRef] = []
    seen: set[tuple[str | None, str | None, str | None, str | None, str | None, str | None]] = set()
    covered_spans: list[tuple[int, int]] = []

    def _is_covered(span: tuple[int, int]) -> bool:
        return any(s <= span[0] and span[1] <= e for s, e in covered_spans)

    def _add(
        raw: str,
        reg: str | None,
        sec: str | None,
        par: str | None = None,
        item: str | None = None,
        subitem: str | None = None,
        leaf: str | None = None,
        span: tuple[int, int] | None = None,
    ) -> None:
        norm_sec = sec.strip() if sec else None
        norm_reg = reg.strip() if reg else None
        norm_par = f"({par.lower()})" if par else None
        norm_item = f"({item.lower()})" if item else None
        norm_subitem = f"({subitem})" if subitem else None
        norm_leaf = f"({leaf.upper()})" if leaf else None
        key = (
            norm_reg,
            norm_sec.lower() if norm_sec else None,
            norm_par,
            norm_item,
            norm_subitem.lower() if norm_subitem else None,
            norm_leaf,
        )
        if norm_sec and key not in seen:
            seen.add(key)
            if span is not None:
                covered_spans.append(span)
            refs.append(
                ResolvedStructuralRef(
                    raw_match=raw.strip(),
                    regulation_number=norm_reg,
                    section_number=norm_sec,
                    paragraph_number=norm_par,
                    item_number=norm_item,
                    subitem_number=norm_subitem,
                    amendment_number=amendment,
                    leaf_number=norm_leaf,
                )
            )

    for pattern in (
        _PT_ALINEA_OF_INCISO_RE,
        _PT_ITEM_OF_PAR_RE,
        _PT_PAR_OF_SEC_RE,
        _RBAC_DOT_RE,
        _RBAC_SECTION_RE,
        _STANDALONE_SECTION_RE,
    ):
        for match in pattern.finditer(query):
            span = match.span()
            if _is_covered(span):
                continue
            groups = match.groupdict()
            _add(
                match.group(0),
                groups.get("reg"),
                groups.get("sec"),
                groups.get("par"),
                groups.get("item"),
                groups.get("subitem"),
                groups.get("leaf"),
                span=span,
            )

    for match in _APPENDIX_RE.finditer(query):
        sec_clean = re.sub(r"\s+", " ", match.group("sec")).strip()
        reg_num = match.group("reg") or hinted_reg
        _add(match.group(0), reg_num, sec_clean)
        code = match.group("code").strip().upper()
        if reg_num and len(code) == 1 and code.isalpha():
            # Em RBACs da ANAC, Apêndice X normalmente corresponde à seção X<reg>.1 (ex.: F43.1)
            _add(match.group(0), reg_num, f"{code}{reg_num}.1")

    for match in _ARTICLE_RE.finditer(query):
        sec_raw = re.sub(r"^Artigo\b", "Art.", match.group("sec"), flags=re.IGNORECASE)
        sec_clean = re.sub(r"\s+", " ", sec_raw).strip()
        _add(match.group(0), match.group("reg"), sec_clean)

    return tuple(refs)
