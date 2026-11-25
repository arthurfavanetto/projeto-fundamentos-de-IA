"""Seletor determinístico e verificador literal de subunidades normativas no SSOT SQLite.

Garante que consultas com granularidade estrutural explícita (seção, parágrafo, inciso,
alínea ou subitem) retornem exatamente a extensão normativa solicitada a partir dos
registros `ParsedUnitRecord` já persistidos em `normative_units`, sem depender de árvore
hierárquica aninhada no banco e sem jamais sintetizar separadores arbitrários.

Invariantes fail-closed:
1. Toda composição de subunidades (ex.: parágrafo + incisos subordinados, ou inciso +
   alíneas subordinadas) é verificada caractere-a-caractere contra `root_unit.original_text`
   em fronteiras exatas de linha. Se a verificação falhar, a resposta é bloqueada.
2. Subunidade ausente ou ambígua jamais faz fallback para a seção inteira.
3. A correspondência distingue explicitamente inciso `(5)` de alínea `(5)(i)` e de subitem `(A)`.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from rbac_assistant.domain.entities.normative_unit import NormativeUnitType, StructureStatus
from rbac_assistant.domain.entities.rag import ResolvedStructuralRef
from rbac_assistant.domain.ports.validation_port import ParsedUnitRecord

_NUMERIC_TOKEN_RE = re.compile(r"^\(\d+\)$")
_ROMAN_TOKEN_RE = re.compile(r"^\([ivxlcdm]+\)$")
_UPPER_ALPHA_TOKEN_RE = re.compile(r"^\([A-Z]\)$")


class StructuralGranularity(StrEnum):
    """Granularidade estrutural efetivamente solicitada pela consulta."""

    SECTION = "section"
    PARAGRAPH = "paragraph"
    INCISO = "inciso"
    ALINEA = "alinea"
    SUBITEM = "subitem"


class SubunitSelectionStatus(StrEnum):
    """Resultado determinístico da resolução estrutural sobre `normative_units`."""

    MATCHED = "matched"
    NOT_FOUND = "not_found"
    AMBIGUOUS = "ambiguous"
    VERIFICATION_FAILED = "verification_failed"


@dataclass(frozen=True, slots=True)
class SubunitSelectionResult:
    """Resultado verificado da seleção granular contra a unidade-raiz no SSOT."""

    status: SubunitSelectionStatus
    granularity: StructuralGranularity
    found: bool
    root_unit: ParsedUnitRecord | None = None
    selected_units: tuple[ParsedUnitRecord, ...] = ()
    unit_id: str = ""
    paragraph_number: str = ""
    item_number: str = ""
    original_text: str = ""
    page_numbers: tuple[int, ...] = ()
    structure_status: StructureStatus = StructureStatus.PENDING_VALIDATION
    reason: str = ""


def _norm_token(value: str | None) -> str:
    return (value or "").strip().lower()


def determine_requested_granularity(
    target_ref: ResolvedStructuralRef | None,
) -> StructuralGranularity:
    """Identifica o nível estrutural exato solicitado pela referência resolvida."""
    if target_ref is None:
        return StructuralGranularity.SECTION
    if target_ref.leaf_number:
        return StructuralGranularity.SUBITEM
    if target_ref.subitem_number:
        return StructuralGranularity.ALINEA
    if target_ref.item_number:
        return StructuralGranularity.INCISO
    if target_ref.paragraph_number:
        return StructuralGranularity.PARAGRAPH
    return StructuralGranularity.SECTION


def _find_line_bounded_occurrences(haystack: str, needle: str) -> list[int]:
    """Localiza todas as ocorrências de `needle` em `haystack` alinhadas a fronteiras de linha."""
    if not needle:
        return []
    positions: list[int] = []
    start = 0
    needle_len = len(needle)
    hay_len = len(haystack)
    while True:
        idx = haystack.find(needle, start)
        if idx < 0:
            break
        left_ok = idx == 0 or haystack[idx - 1] == "\n"
        end_idx = idx + needle_len
        right_ok = end_idx == hay_len or haystack[end_idx] == "\n"
        if left_ok and right_ok:
            positions.append(idx)
        start = idx + 1
    return positions


def _verify_contiguous_slice_in_root(
    *,
    root_text: str,
    child_units: Sequence[ParsedUnitRecord],
    selected_indices: Sequence[int],
) -> tuple[str | None, str | None]:
    """Comprova que os registros selecionados formam fatia literal exata da unidade-raiz.

    Retorna `(verified_text, None)` se comprovado caractere-a-caractere; caso contrário
    retorna `(None, motivo_objetivo_do_bloqueio)`.
    """
    if not selected_indices:
        return None, "Nenhuma subunidade selecionada para verificação literal."

    # 1. Garante contiguidade estrita e ordem crescente de `order_index`
    expected_indices = list(range(selected_indices[0], selected_indices[0] + len(selected_indices)))
    if list(selected_indices) != expected_indices:
        return (
            None,
            "Registros selecionados não são contíguos na ordem estrutural (`order_index`).",
        )

    selected = [child_units[i] for i in selected_indices]
    for prev_u, next_u in itertools.pairwise(selected):
        if next_u.order_index <= prev_u.order_index:
            return (
                None,
                "Ordenação estrutural (`order_index`) não é estritamente crescente "
                "entre subunidades.",
            )

    composed = (
        selected[0].original_text
        if len(selected) == 1
        else "\n".join(u.original_text for u in selected)
    )
    if not composed:
        return None, "Subunidade selecionada possui `original_text` vazio."

    # 2. Caso inequívoco: exatamente 1 ocorrência alinhada em fronteira de linha na unidade-raiz
    occurrences = _find_line_bounded_occurrences(root_text, composed)
    if len(occurrences) == 1:
        pos = occurrences[0]
        verified_slice = root_text[pos : pos + len(composed)]
        if verified_slice == composed:
            return verified_slice, None

    if not occurrences:
        return (
            None,
            "A composição das subunidades não corresponde a uma fatia literal contígua "
            "do `original_text` da unidade-raiz (possível lacuna, truncamento ou divergência).",
        )

    # 3. Múltiplas ocorrências idênticas no texto-raiz: alinha monotonicamente todos os filhos
    # desde o início da seção até o fim da seleção para comprovar a posição exata sem heurística.
    cursor = 0
    span_start: int | None = None
    span_end: int | None = None
    sel_start = selected_indices[0]
    sel_end = selected_indices[-1]
    for idx in range(0, sel_end + 1):
        unit_text = child_units[idx].original_text
        if not unit_text:
            return None, "Subunidade anterior vazia impede verificação posicional determinística."
        found_at = -1
        search_from = cursor
        while True:
            pos = root_text.find(unit_text, search_from)
            if pos < 0:
                break
            left_ok = pos == 0 or root_text[pos - 1] == "\n"
            end_pos = pos + len(unit_text)
            right_ok = end_pos == len(root_text) or root_text[end_pos] == "\n"
            if left_ok and right_ok:
                found_at = pos
                break
            search_from = pos + 1
        if found_at < 0:
            return (
                None,
                "Não foi possível alinhar monotonicamente as subunidades ao `original_text` "
                "da unidade-raiz.",
            )
        end_pos = found_at + len(unit_text)
        if idx == sel_start:
            span_start = found_at
        elif idx > sel_start and (
            span_end is None or found_at != span_end + 1 or root_text[span_end] != "\n"
        ):
            return (
                None,
                "Fronteira entre subunidades selecionadas não corresponde exatamente a '\\n' "
                "no `original_text` da unidade-raiz.",
            )
        if idx >= sel_start:
            span_end = end_pos
        cursor = end_pos + 1

    if span_start is None or span_end is None:
        return None, "Falha ao delimitar intervalo exato da subunidade na unidade-raiz."

    verified_slice = root_text[span_start:span_end]
    if verified_slice != composed:
        return None, "Fatia extraída da unidade-raiz diverge da composição literal das subunidades."
    return verified_slice, None


def _locate_paragraph_span(
    child_units: Sequence[ParsedUnitRecord],
    *,
    paragraph_token: str,
    section_number: str,
) -> tuple[tuple[int, int] | None, SubunitSelectionStatus | None, str]:
    """Localiza o intervalo `[start, end)` de um parágrafo e todas as suas subunidades."""
    par_norm = _norm_token(paragraph_token)
    anchors = [
        idx
        for idx, unit in enumerate(child_units)
        if unit.unit_type == NormativeUnitType.PARAGRAPH
        and _norm_token(unit.paragraph_number) == par_norm
    ]
    if not anchors:
        return (
            None,
            SubunitSelectionStatus.NOT_FOUND,
            f"Parágrafo {paragraph_token} não encontrado na seção {section_number}.",
        )
    if len(anchors) > 1:
        return (
            None,
            SubunitSelectionStatus.AMBIGUOUS,
            f"Parágrafo {paragraph_token} ambíguo ({len(anchors)} ocorrências) "
            f"na seção {section_number}.",
        )
    start_idx = anchors[0]
    end_idx = start_idx + 1
    while end_idx < len(child_units):
        curr = child_units[end_idx]
        if curr.unit_type == NormativeUnitType.PARAGRAPH:
            break
        if _norm_token(curr.paragraph_number) != par_norm:
            return (
                None,
                SubunitSelectionStatus.VERIFICATION_FAILED,
                f"Inconsistência estrutural na seção {section_number}: subunidade "
                f"'{curr.unit_id}' (order_index={curr.order_index}) possui "
                f"paragraph_number='{curr.paragraph_number}' divergente de '{paragraph_token}'.",
            )
        end_idx += 1
    stray_same_par = [
        curr
        for idx, curr in enumerate(child_units)
        if not (start_idx <= idx < end_idx) and _norm_token(curr.paragraph_number) == par_norm
    ]
    if stray_same_par:
        return (
            None,
            SubunitSelectionStatus.AMBIGUOUS,
            f"Parágrafo {paragraph_token} possui registros não contíguos/duplicados fora de seu "
            f"bloco na seção {section_number}.",
        )
    return (start_idx, end_idx), None, ""


def _locate_inciso_span(
    child_units: Sequence[ParsedUnitRecord],
    *,
    par_start: int,
    par_end: int,
    paragraph_token: str,
    item_token: str,
    section_number: str,
) -> tuple[tuple[int, int] | None, SubunitSelectionStatus | None, str]:
    """Localiza o intervalo `[start, end)` de um inciso e suas alíneas/subitens subordinados."""
    inc_norm = _norm_token(item_token)
    raw_item = item_token.strip()

    # 1. Busca primária estrita em registros `INCISO` com `item_number` exato
    anchors = [
        idx
        for idx in range(par_start + 1, par_end)
        if child_units[idx].unit_type == NormativeUnitType.INCISO
        and _norm_token(child_units[idx].item_number) == inc_norm
    ]

    # 2. Caso um documento use alínea romana ou subitem alfabético diretamente sob o parágrafo
    # (sem inciso numérico intermediário), só aceita correspondência exata sem prefixo composto.
    if not anchors and not _NUMERIC_TOKEN_RE.match(inc_norm):
        if _UPPER_ALPHA_TOKEN_RE.match(raw_item):
            anchors = [
                idx
                for idx in range(par_start + 1, par_end)
                if child_units[idx].unit_type == NormativeUnitType.SUBITEM
                and child_units[idx].item_number.strip() == raw_item
            ]
        elif _ROMAN_TOKEN_RE.match(inc_norm):
            anchors = [
                idx
                for idx in range(par_start + 1, par_end)
                if child_units[idx].unit_type == NormativeUnitType.ALINEA
                and _norm_token(child_units[idx].item_number) == inc_norm
            ]

    if not anchors:
        return (
            None,
            SubunitSelectionStatus.NOT_FOUND,
            f"Inciso/item {paragraph_token}{item_token} não encontrado na seção {section_number}.",
        )
    if len(anchors) > 1:
        return (
            None,
            SubunitSelectionStatus.AMBIGUOUS,
            f"Inciso/item {paragraph_token}{item_token} ambíguo ({len(anchors)} ocorrências) "
            f"na seção {section_number}.",
        )

    inc_start = anchors[0]
    anchor_type = child_units[inc_start].unit_type
    inc_end = inc_start + 1
    while inc_end < par_end:
        curr = child_units[inc_end]
        if curr.unit_type in (NormativeUnitType.PARAGRAPH, NormativeUnitType.INCISO):
            break
        if anchor_type == NormativeUnitType.ALINEA and curr.unit_type == NormativeUnitType.ALINEA:
            break
        if anchor_type == NormativeUnitType.SUBITEM:
            break
        if (
            anchor_type == NormativeUnitType.INCISO
            and curr.unit_type == NormativeUnitType.ALINEA
            and not _norm_token(curr.item_number).startswith(inc_norm)
        ):
            return (
                None,
                SubunitSelectionStatus.VERIFICATION_FAILED,
                f"Inconsistência estrutural na seção {section_number}: alínea "
                f"'{curr.item_number}' não pertence ao inciso '{item_token}'.",
            )
        inc_end += 1

    return (inc_start, inc_end), None, ""


def _locate_alinea_span(
    child_units: Sequence[ParsedUnitRecord],
    *,
    inc_start: int,
    inc_end: int,
    paragraph_token: str,
    item_token: str,
    subitem_token: str,
    section_number: str,
) -> tuple[tuple[int, int] | None, SubunitSelectionStatus | None, str]:
    """Localiza o intervalo `[start, end)` de uma alínea (ex.: `(5)(i)`) e eventuais subitens."""
    inc_norm = _norm_token(item_token)
    sub_norm = _norm_token(subitem_token)
    composite_norm = f"{inc_norm}{sub_norm}"
    raw_sub = subitem_token.strip()

    # Correspondência primária em `ALINEA`: aceita formato composto `(5)(i)` ou simples `(i)`
    # estritamente dentro do intervalo do inciso `(5)`, jamais confundindo com o próprio `INCISO`.
    anchors = [
        idx
        for idx in range(inc_start + 1, inc_end)
        if child_units[idx].unit_type == NormativeUnitType.ALINEA
        and _norm_token(child_units[idx].item_number) in (composite_norm, sub_norm)
    ]

    # Caso o 3º nível seja diretamente um `SUBITEM` alfabético maiúsculo `(A)` sob inciso
    if not anchors and _UPPER_ALPHA_TOKEN_RE.match(raw_sub):
        anchors = [
            idx
            for idx in range(inc_start + 1, inc_end)
            if child_units[idx].unit_type == NormativeUnitType.SUBITEM
            and child_units[idx].item_number.strip() in (f"{item_token.strip()}{raw_sub}", raw_sub)
        ]

    if not anchors:
        return (
            None,
            SubunitSelectionStatus.NOT_FOUND,
            f"Alínea/subitem {paragraph_token}{item_token}{subitem_token} não encontrada "
            f"na seção {section_number}.",
        )
    if len(anchors) > 1:
        return (
            None,
            SubunitSelectionStatus.AMBIGUOUS,
            f"Alínea/subitem {paragraph_token}{item_token}{subitem_token} ambígua "
            f"({len(anchors)} ocorrências) na seção {section_number}.",
        )

    sub_start = anchors[0]
    anchor_type = child_units[sub_start].unit_type
    sub_end = sub_start + 1
    if anchor_type == NormativeUnitType.ALINEA:
        while sub_end < inc_end:
            curr = child_units[sub_end]
            if curr.unit_type != NormativeUnitType.SUBITEM:
                break
            sub_end += 1

    return (sub_start, sub_end), None, ""


def select_structural_units(
    units: Sequence[ParsedUnitRecord],
    target_ref: ResolvedStructuralRef | None,
) -> SubunitSelectionResult:
    """Seleciona e verifica literalmente a extensão normativa exata solicitada em `target_ref`.

    - Se `target_ref` for `None` ou de nível seção: retorna a unidade-raiz completa.
    - Se `target_ref` especificar parágrafo, inciso, alínea ou subitem: seleciona apenas os
      registros pertencentes àquele dispositivo em `order_index` e comprova correspondência
      literal exata contra `root_unit.original_text`.
    """
    granularity = determine_requested_granularity(target_ref)
    if not units:
        return SubunitSelectionResult(
            status=SubunitSelectionStatus.NOT_FOUND,
            granularity=granularity,
            found=False,
            reason="Nenhuma unidade normativa encontrada no SQLite para a seção solicitada.",
        )

    ordered_units = sorted(units, key=lambda u: (u.order_index, u.unit_id))
    root_units = [u for u in ordered_units if u.parent_unit_id is None]

    if not root_units:
        return SubunitSelectionResult(
            status=SubunitSelectionStatus.VERIFICATION_FAILED,
            granularity=granularity,
            found=False,
            reason=(
                "Unidade-raiz (`parent_unit_id=None`) ausente no SQLite para esta seção; "
                "por política fail-closed, o sistema não utiliza uma subunidade isolada como "
                "seção completa nem libera transcrição sem comprovação literal na unidade-raiz."
            ),
        )

    if len(root_units) > 1 and granularity != StructuralGranularity.SECTION:
        return SubunitSelectionResult(
            status=SubunitSelectionStatus.AMBIGUOUS,
            granularity=granularity,
            found=False,
            reason=(
                f"Múltiplas unidades-raiz ({len(root_units)}) encontradas para a seção "
                f"'{root_units[0].section_number}'; seleção granular bloqueada por ambiguidade."
            ),
        )

    root_unit = root_units[0]

    # Nível 1: Seção completa
    if granularity == StructuralGranularity.SECTION or target_ref is None:
        all_pages = tuple(sorted({p for u in ordered_units for p in u.page_numbers}))
        sec_status = (
            StructureStatus.PARSED
            if all(u.structure_status == StructureStatus.PARSED for u in root_units)
            else StructureStatus.PENDING_VALIDATION
        )
        return SubunitSelectionResult(
            status=SubunitSelectionStatus.MATCHED,
            granularity=StructuralGranularity.SECTION,
            found=True,
            root_unit=root_unit,
            selected_units=(root_unit,),
            unit_id=root_unit.unit_id,
            paragraph_number="",
            item_number="",
            original_text=root_unit.original_text,
            page_numbers=all_pages,
            structure_status=sec_status,
            reason="matched:section_root",
        )

    child_units = [u for u in ordered_units if u.parent_unit_id == root_unit.unit_id]
    if not child_units:
        return SubunitSelectionResult(
            status=SubunitSelectionStatus.NOT_FOUND,
            granularity=granularity,
            found=False,
            root_unit=root_unit,
            reason=(
                f"A seção {root_unit.section_number} não possui subunidades estruturadas "
                f"no SQLite para resolver {target_ref.canonical_label}."
            ),
        )

    sec_num = root_unit.section_number
    selected_indices: list[int] = []
    anchor_unit: ParsedUnitRecord | None = None
    resolved_par = ""
    resolved_item = ""

    if target_ref.paragraph_number is None:
        # Defesa fail-closed para referência sem parágrafo explícito (ex.: item sem `(a)`)
        inc_norm = _norm_token(target_ref.item_number)
        anchors = [
            idx
            for idx, u in enumerate(child_units)
            if u.unit_type == NormativeUnitType.INCISO and _norm_token(u.item_number) == inc_norm
        ]
        if not anchors:
            return SubunitSelectionResult(
                status=SubunitSelectionStatus.NOT_FOUND,
                granularity=granularity,
                found=False,
                root_unit=root_unit,
                reason=(
                    f"Item {target_ref.item_number} não encontrado na seção {sec_num}."
                ),
            )
        if len(anchors) > 1:
            return SubunitSelectionResult(
                status=SubunitSelectionStatus.AMBIGUOUS,
                granularity=granularity,
                found=False,
                root_unit=root_unit,
                reason=(
                    f"Item {target_ref.item_number} ambíguo ({len(anchors)} ocorrências) "
                    f"na seção {sec_num}; informe o parágrafo correspondente."
                ),
            )
        inc_start = anchors[0]
        inc_end = inc_start + 1
        while inc_end < len(child_units) and child_units[inc_end].unit_type not in (
            NormativeUnitType.PARAGRAPH,
            NormativeUnitType.INCISO,
        ):
            inc_end += 1
        selected_indices = list(range(inc_start, inc_end))
        anchor_unit = child_units[inc_start]
        resolved_par = anchor_unit.paragraph_number
        resolved_item = anchor_unit.item_number
    else:
        par_span, par_err_status, par_err_msg = _locate_paragraph_span(
            child_units,
            paragraph_token=target_ref.paragraph_number,
            section_number=sec_num,
        )
        if par_span is None:
            return SubunitSelectionResult(
                status=par_err_status or SubunitSelectionStatus.NOT_FOUND,
                granularity=granularity,
                found=False,
                root_unit=root_unit,
                reason=par_err_msg,
            )
        par_start, par_end = par_span

        if granularity == StructuralGranularity.PARAGRAPH:
            selected_indices = list(range(par_start, par_end))
            anchor_unit = child_units[par_start]
            resolved_par = anchor_unit.paragraph_number
            resolved_item = ""
        else:
            assert target_ref.item_number is not None
            inc_span, inc_err_status, inc_err_msg = _locate_inciso_span(
                child_units,
                par_start=par_start,
                par_end=par_end,
                paragraph_token=target_ref.paragraph_number,
                item_token=target_ref.item_number,
                section_number=sec_num,
            )
            if inc_span is None:
                return SubunitSelectionResult(
                    status=inc_err_status or SubunitSelectionStatus.NOT_FOUND,
                    granularity=granularity,
                    found=False,
                    root_unit=root_unit,
                    reason=inc_err_msg,
                )
            inc_start, inc_end = inc_span

            if granularity == StructuralGranularity.INCISO:
                selected_indices = list(range(inc_start, inc_end))
                anchor_unit = child_units[inc_start]
                resolved_par = anchor_unit.paragraph_number
                resolved_item = anchor_unit.item_number
            else:
                assert target_ref.subitem_number is not None
                ali_span, ali_err_status, ali_err_msg = _locate_alinea_span(
                    child_units,
                    inc_start=inc_start,
                    inc_end=inc_end,
                    paragraph_token=target_ref.paragraph_number,
                    item_token=target_ref.item_number,
                    subitem_token=target_ref.subitem_number,
                    section_number=sec_num,
                )
                if ali_span is None:
                    return SubunitSelectionResult(
                        status=ali_err_status or SubunitSelectionStatus.NOT_FOUND,
                        granularity=granularity,
                        found=False,
                        root_unit=root_unit,
                        reason=ali_err_msg,
                    )
                ali_start, ali_end = ali_span

                if granularity == StructuralGranularity.ALINEA:
                    selected_indices = list(range(ali_start, ali_end))
                    anchor_unit = child_units[ali_start]
                    resolved_par = anchor_unit.paragraph_number
                    resolved_item = anchor_unit.item_number
                else:
                    # 4º nível: Subitem `(A)` dentro da alínea `(i)`
                    assert target_ref.leaf_number is not None
                    raw_leaf = target_ref.leaf_number.strip()
                    leaf_anchors = [
                        idx
                        for idx in range(ali_start + 1, ali_end)
                        if child_units[idx].unit_type == NormativeUnitType.SUBITEM
                        and child_units[idx].item_number.strip() == raw_leaf
                    ]
                    if not leaf_anchors:
                        return SubunitSelectionResult(
                            status=SubunitSelectionStatus.NOT_FOUND,
                            granularity=granularity,
                            found=False,
                            root_unit=root_unit,
                            reason=(
                                f"Subitem {target_ref.canonical_label} não encontrado "
                                f"na seção {sec_num}."
                            ),
                        )
                    if len(leaf_anchors) > 1:
                        return SubunitSelectionResult(
                            status=SubunitSelectionStatus.AMBIGUOUS,
                            granularity=granularity,
                            found=False,
                            root_unit=root_unit,
                            reason=(
                                f"Subitem {target_ref.canonical_label} ambíguo "
                                f"({len(leaf_anchors)} ocorrências) na seção {sec_num}."
                            ),
                        )
                    leaf_idx = leaf_anchors[0]
                    selected_indices = [leaf_idx]
                    anchor_unit = child_units[leaf_idx]
                    resolved_par = anchor_unit.paragraph_number
                    resolved_item = (
                        f"{target_ref.item_number.strip()}"
                        f"{target_ref.subitem_number.strip()}"
                        f"{anchor_unit.item_number.strip()}"
                    )

    verified_text, verify_error = _verify_contiguous_slice_in_root(
        root_text=root_unit.original_text,
        child_units=child_units,
        selected_indices=selected_indices,
    )
    if verified_text is None or anchor_unit is None:
        return SubunitSelectionResult(
            status=SubunitSelectionStatus.VERIFICATION_FAILED,
            granularity=granularity,
            found=False,
            root_unit=root_unit,
            reason=verify_error or "Falha na verificação literal da subunidade selecionada.",
        )

    selected_records = tuple(child_units[i] for i in selected_indices)
    pages = tuple(sorted({p for u in selected_records for p in u.page_numbers}))
    all_parsed = root_unit.structure_status == StructureStatus.PARSED and all(
        u.structure_status == StructureStatus.PARSED for u in selected_records
    )
    return SubunitSelectionResult(
        status=SubunitSelectionStatus.MATCHED,
        granularity=granularity,
        found=True,
        root_unit=root_unit,
        selected_units=selected_records,
        unit_id=anchor_unit.unit_id,
        paragraph_number=resolved_par,
        item_number=resolved_item,
        original_text=verified_text,
        page_numbers=pages,
        structure_status=(
            StructureStatus.PARSED if all_parsed else StructureStatus.PENDING_VALIDATION
        ),
        reason=f"matched:{granularity.value}",
    )
