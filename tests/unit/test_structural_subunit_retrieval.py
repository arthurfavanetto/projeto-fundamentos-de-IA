"""Testes unitários de resolução estrutural granular, verificação literal e fusão RRF."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from rbac_assistant.domain.entities.normative_unit import NormativeUnitType, StructureStatus
from rbac_assistant.domain.entities.rag import (
    ResolvedStructuralRef,
    RetrievalCandidate,
    RetrievalSource,
)
from rbac_assistant.domain.ports.validation_port import ParsedUnitRecord
from rbac_assistant.rag.retrieval.hybrid_retriever import fuse_candidates_rrf
from rbac_assistant.rag.retrieval.section_resolver import resolve_structural_references
from rbac_assistant.rag.retrieval.subunit_selector import (
    StructuralGranularity,
    SubunitSelectionStatus,
    select_structural_units,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_rbac43_10_flat_units() -> list[ParsedUnitRecord]:
    """Espelha fielmente o armazenamento plano `normative_units` gerado para a seção 43.10."""
    ver = "anac_rbac_43_emd_05_5026d2da"
    root_id = f"{ver}::43.10"

    entries: list[tuple[NormativeUnitType, str, str, tuple[int, ...], str]] = [
        (
            NormativeUnitType.PARAGRAPH,
            "(a)",
            "",
            (9,),
            "(a) Definições usadas nesta seção. Para os propósitos desta seção se aplicam "
            "as seguintes\ndefinições:",
        ),
        (
            NormativeUnitType.INCISO,
            "(a)",
            "(1)",
            (9,),
            "(1) Peça com limite de vida significa qualquer peça ou parte para a qual um "
            "limite obrigatório\nde substituição é especificado no projeto de tipo.",
        ),
        (
            NormativeUnitType.INCISO,
            "(a)",
            "(2)",
            (9,),
            "(2) Situação de vida significa número acumulado de ciclos, horas ou qualquer "
            "outro limite.",
        ),
        (
            NormativeUnitType.PARAGRAPH,
            "(b)",
            "",
            (9,),
            "(b) Remoção temporária de peças de produtos com certificado de tipo:",
        ),
        (
            NormativeUnitType.INCISO,
            "(b)",
            "(1)",
            (9,),
            "(1) a situação de vida da peça não foi alterada;",
        ),
        (
            NormativeUnitType.PARAGRAPH,
            "(c)",
            "",
            (9,),
            "(c) Controle de peças removidas de produtos com certificado de tipo:",
        ),
        (
            NormativeUnitType.INCISO,
            "(c)",
            "(5)",
            (10,),
            "(5) Segregação. Tais métodos devem incluir, pelo menos:",
        ),
        (
            NormativeUnitType.ALINEA,
            "(c)",
            "(5)(i)",
            (10,),
            "(i) a conservação dos registros do número da peça, seu número de série e sua "
            "situação de\nvida; e",
        ),
        (
            NormativeUnitType.ALINEA,
            "(c)",
            "(5)(ii)",
            (10,),
            "(ii) a garantia de que a peça está armazenada em local fisicamente separado.",
        ),
        (
            NormativeUnitType.INCISO,
            "(c)",
            "(6)",
            (10,),
            "(6) Mutilação. A peça pode ser mutilada para impedir sua instalação.",
        ),
        (
            NormativeUnitType.PARAGRAPH,
            "(d)",
            "",
            (10,),
            "(d) Transferência de peças com limite de vida.",
        ),
    ]

    root_text = "43.10 Controle de peças com limite de vida\n" + "\n".join(e[4] for e in entries)
    records: list[ParsedUnitRecord] = [
        ParsedUnitRecord(
            unit_id=root_id,
            version_id=ver,
            parent_unit_id=None,
            unit_type=NormativeUnitType.SECTION,
            section_number="43.10",
            subsection_number="",
            paragraph_number="",
            item_number="",
            title="Controle de peças com limite de vida",
            subpart="",
            original_text=root_text,
            normalized_text=root_text,
            start_page=9,
            end_page=10,
            page_numbers=(9, 10),
            order_index=109,
            structure_status=StructureStatus.PARSED,
            content_hash=_sha(root_text),
        )
    ]
    for offset, (utype, par, item, pages, text) in enumerate(entries, start=110):
        records.append(
            ParsedUnitRecord(
                unit_id=f"{root_id}::{offset}",
                version_id=ver,
                parent_unit_id=root_id,
                unit_type=utype,
                section_number="43.10",
                subsection_number="",
                paragraph_number=par,
                item_number=item,
                title="",
                subpart="",
                original_text=text,
                normalized_text=text,
                start_page=pages[0],
                end_page=pages[-1],
                page_numbers=pages,
                order_index=offset,
                structure_status=StructureStatus.PARSED,
                content_hash=_sha(text),
            )
        )
    return records


@pytest.mark.unit
def test_section_resolver_captures_granular_refs_and_natural_pt_phrasings() -> None:
    # Caso C real reportado pelo usuário
    refs_c = resolve_structural_references(
        "Transcreva literalmente o item 43.10(a)(1) do RBAC n 43, Emenda n 05."
    )
    assert len(refs_c) == 1
    assert refs_c[0].regulation_number == "43"
    assert refs_c[0].section_number == "43.10"
    assert refs_c[0].paragraph_number == "(a)"
    assert refs_c[0].item_number == "(1)"
    assert refs_c[0].subitem_number is None
    assert refs_c[0].amendment_number == "05"
    assert refs_c[0].has_subunit_target is True
    assert refs_c[0].canonical_label == "43.10(a)(1)"

    # Caso D (alínea)
    refs_d = resolve_structural_references("Transcreva literalmente o item 43.10(c)(5)(i).")
    assert len(refs_d) == 1
    assert refs_d[0].section_number == "43.10"
    assert refs_d[0].paragraph_number == "(c)"
    assert refs_d[0].item_number == "(5)"
    assert refs_d[0].subitem_number == "(i)"
    assert refs_d[0].canonical_label == "43.10(c)(5)(i)"

    # Caso 4 níveis (subitem alfabético sob alínea)
    refs_leaf = resolve_structural_references("Transcreva 43.3(g)(1)(i)(A) do RBAC 43.")
    assert len(refs_leaf) == 1
    assert refs_leaf[0].paragraph_number == "(g)"
    assert refs_leaf[0].item_number == "(1)"
    assert refs_leaf[0].subitem_number == "(i)"
    assert refs_leaf[0].leaf_number == "(A)"

    # Formas em português natural não duplicam referência genérica da seção inteira
    refs_pt_par = resolve_structural_references("Transcreva o parágrafo (a) da seção 43.10.")
    assert len(refs_pt_par) == 1
    assert refs_pt_par[0].section_number == "43.10"
    assert refs_pt_par[0].paragraph_number == "(a)"
    assert refs_pt_par[0].item_number is None

    refs_pt_item = resolve_structural_references(
        "Transcreva o inciso (1) do parágrafo (a) da seção 43.10."
    )
    assert len(refs_pt_item) == 1
    assert refs_pt_item[0].paragraph_number == "(a)"
    assert refs_pt_item[0].item_number == "(1)"

    refs_pt_ali = resolve_structural_references(
        "Transcreva a alínea (i) do inciso (5) do parágrafo (c) da seção 43.10."
    )
    assert len(refs_pt_ali) == 1
    assert refs_pt_ali[0].paragraph_number == "(c)"
    assert refs_pt_ali[0].item_number == "(5)"
    assert refs_pt_ali[0].subitem_number == "(i)"

    # Múltiplas referências distintas na mesma seção preservam ambas as granularidades
    refs_multi = resolve_structural_references(
        "Compare os itens 43.10(a)(1) e 43.10(a)(2) do RBAC 43."
    )
    assert [r.canonical_label for r in refs_multi] == ["43.10(a)(1)", "43.10(a)(2)"]


@pytest.mark.unit
def test_subunit_selector_enforces_exact_granularity_for_section_paragraph_inciso_and_alinea() -> (
    None
):
    units = _build_rbac43_10_flat_units()
    root = units[0]

    # Caso A — Seção inteira
    sel_sec = select_structural_units(
        units,
        ResolvedStructuralRef(raw_match="43.10", regulation_number="43", section_number="43.10"),
    )
    assert sel_sec.found is True
    assert sel_sec.granularity == StructuralGranularity.SECTION
    assert sel_sec.unit_id == root.unit_id
    assert sel_sec.original_text == root.original_text
    assert sel_sec.page_numbers == (9, 10)

    # Caso B — Parágrafo (a) inclui (a) + (1) + (2), excluindo (b), (c), (d)
    sel_par = select_structural_units(
        units,
        ResolvedStructuralRef(
            raw_match="43.10(a)",
            regulation_number="43",
            section_number="43.10",
            paragraph_number="(a)",
        ),
    )
    assert sel_par.found is True
    assert sel_par.granularity == StructuralGranularity.PARAGRAPH
    assert sel_par.unit_id == units[1].unit_id
    assert sel_par.paragraph_number == "(a)"
    assert sel_par.item_number == ""
    assert sel_par.page_numbers == (9,)
    expected_par_a = "\n".join(u.original_text for u in units[1:4])
    assert sel_par.original_text == expected_par_a
    assert sel_par.original_text in root.original_text
    assert "(b)" not in sel_par.original_text
    assert "(c)" not in sel_par.original_text
    assert "(d)" not in sel_par.original_text

    # Caso C — Inciso específico 43.10(a)(1): SOMENTE (1), sem (2) e sem (b)/(c)/(d)
    sel_inc = select_structural_units(
        units,
        ResolvedStructuralRef(
            raw_match="43.10(a)(1)",
            regulation_number="43",
            section_number="43.10",
            paragraph_number="(a)",
            item_number="(1)",
        ),
    )
    assert sel_inc.found is True
    assert sel_inc.granularity == StructuralGranularity.INCISO
    assert sel_inc.unit_id == units[2].unit_id
    assert sel_inc.paragraph_number == "(a)"
    assert sel_inc.item_number == "(1)"
    assert sel_inc.page_numbers == (9,)
    assert sel_inc.original_text == units[2].original_text
    assert "(2)" not in sel_inc.original_text
    assert "(a) Definições" not in sel_inc.original_text
    assert "(b)" not in sel_inc.original_text

    # Caso C2 — Inciso com alíneas subordinadas 43.10(c)(5): inclui (5) + (i) + (ii), sem (6)
    sel_inc_c5 = select_structural_units(
        units,
        ResolvedStructuralRef(
            raw_match="43.10(c)(5)",
            regulation_number="43",
            section_number="43.10",
            paragraph_number="(c)",
            item_number="(5)",
        ),
    )
    assert sel_inc_c5.found is True
    assert sel_inc_c5.original_text == "\n".join(u.original_text for u in units[7:10])
    assert "(6) Mutilação" not in sel_inc_c5.original_text
    assert "(d)" not in sel_inc_c5.original_text

    # Caso D — Alínea específica 43.10(c)(5)(i): SOMENTE (i), não confunde com inciso (5) nem (ii)
    sel_ali = select_structural_units(
        units,
        ResolvedStructuralRef(
            raw_match="43.10(c)(5)(i)",
            regulation_number="43",
            section_number="43.10",
            paragraph_number="(c)",
            item_number="(5)",
            subitem_number="(i)",
        ),
    )
    assert sel_ali.found is True
    assert sel_ali.granularity == StructuralGranularity.ALINEA
    assert sel_ali.unit_id == units[8].unit_id
    assert sel_ali.paragraph_number == "(c)"
    assert sel_ali.item_number == "(5)(i)"
    assert sel_ali.page_numbers == (10,)
    assert sel_ali.original_text == units[8].original_text
    assert "(5) Segregação" not in sel_ali.original_text
    assert "(ii)" not in sel_ali.original_text
    assert "(6)" not in sel_ali.original_text


@pytest.mark.unit
def test_subunit_selector_blocks_fail_closed_on_missing_ambiguous_or_corrupted_units() -> None:
    units = _build_rbac43_10_flat_units()

    # Caso E1: Inciso, parágrafo e alínea inexistentes -> NOT_FOUND (nunca seção-pai)
    for bad_ref in (
        ResolvedStructuralRef("43.10(a)(99)", "43", "43.10", "(a)", "(99)"),
        ResolvedStructuralRef("43.10(z)", "43", "43.10", "(z)"),
        ResolvedStructuralRef("43.10(c)(5)(ix)", "43", "43.10", "(c)", "(5)", "(ix)"),
        ResolvedStructuralRef("43.10(a)(1)(i)", "43", "43.10", "(a)", "(1)", "(i)"),
    ):
        res = select_structural_units(units, bad_ref)
        assert res.found is False
        assert res.status == SubunitSelectionStatus.NOT_FOUND
        assert res.original_text == ""

    # Caso E2: Subunidade duplicada dentro do bloco OU fora de ordem (não contígua) -> AMBIGUOUS
    dup_in_span = [
        *units[:3],
        replace(units[2], unit_id="anac_rbac_43_emd_05_5026d2da::43.10::111b", order_index=111),
        *units[3:],
    ]
    res_ambig_in = select_structural_units(
        dup_in_span,
        ResolvedStructuralRef("43.10(a)(1)", "43", "43.10", "(a)", "(1)"),
    )
    assert res_ambig_in.found is False
    assert res_ambig_in.status == SubunitSelectionStatus.AMBIGUOUS

    dup_stray = [
        *units,
        replace(units[2], unit_id="anac_rbac_43_emd_05_5026d2da::43.10::999", order_index=1110),
    ]
    res_ambig = select_structural_units(
        dup_stray,
        ResolvedStructuralRef("43.10(a)(1)", "43", "43.10", "(a)", "(1)"),
    )
    assert res_ambig.found is False
    assert res_ambig.status == SubunitSelectionStatus.AMBIGUOUS

    # Caso G1: Texto da subunidade corrompido (não é fatia literal do texto-raiz) -> BLOQUEIO
    corrupted = list(units)
    corrupted[2] = replace(
        units[2],
        original_text="(1) Texto alterado que não existe literalmente na unidade-raiz.",
    )
    res_corrupt = select_structural_units(
        corrupted,
        ResolvedStructuralRef("43.10(a)(1)", "43", "43.10", "(a)", "(1)"),
    )
    assert res_corrupt.found is False
    assert res_corrupt.status == SubunitSelectionStatus.VERIFICATION_FAILED

    # Caso G2: Lacuna interna (falta o inciso (1) entre (a) e (2)) -> falha na verificação literal
    gapped = [u for i, u in enumerate(units) if i != 2]
    res_gap = select_structural_units(
        gapped,
        ResolvedStructuralRef("43.10(a)", "43", "43.10", "(a)"),
    )
    assert res_gap.found is False
    assert res_gap.status == SubunitSelectionStatus.VERIFICATION_FAILED

    # Caso F (unitário): Seção completa sem unidade-raiz (apenas subunidades) -> fail-closed
    subunits_only = [u for u in units if u.parent_unit_id is not None]
    assert len(subunits_only) > 0
    res_no_root_sec = select_structural_units(
        subunits_only,
        ResolvedStructuralRef("43.10", "43", "43.10"),
    )
    assert res_no_root_sec.found is False
    assert res_no_root_sec.status == SubunitSelectionStatus.VERIFICATION_FAILED
    assert res_no_root_sec.original_text == ""
    assert res_no_root_sec.selected_units == ()

    # Caso G (unitário): Subunidade SEM unidade-raiz -> bloqueio por verificação literal impossível
    res_no_root_sub = select_structural_units(
        subunits_only,
        ResolvedStructuralRef("43.10(a)(1)", "43", "43.10", "(a)", "(1)"),
    )
    assert res_no_root_sub.found is False
    assert res_no_root_sub.status == SubunitSelectionStatus.VERIFICATION_FAILED
    assert res_no_root_sub.original_text == ""


@pytest.mark.unit
def test_rrf_fusion_preserves_multiple_granular_refs_on_same_chunk_and_blocks_expansion() -> None:
    """Caso F: RRF não colapsa 43.10(a)(1) e 43.10(a)(2) nem cria candidato de seção inteira."""
    ref_a1 = ResolvedStructuralRef("43.10(a)(1)", "43", "43.10", "(a)", "(1)")
    ref_a2 = ResolvedStructuralRef("43.10(a)(2)", "43", "43.10", "(a)", "(2)")

    candidates = [
        RetrievalCandidate(
            "chunk_43_10",
            RetrievalSource.EXACT_STRUCTURAL,
            rank=1,
            raw_score=1.0,
            target_ref=ref_a1,
        ),
        RetrievalCandidate(
            "chunk_43_10",
            RetrievalSource.EXACT_STRUCTURAL,
            rank=1,
            raw_score=1.0,
            target_ref=ref_a2,
        ),
        # Lexical e vetorial acertam o MESMO chunk-raiz da seção 43.10
        RetrievalCandidate("chunk_43_10", RetrievalSource.LEXICAL_FTS5, rank=1, raw_score=8.5),
        RetrievalCandidate("chunk_43_10", RetrievalSource.SEMANTIC_VECTOR, rank=1, raw_score=0.92),
        # Outro chunk diferente sem match estrutural
        RetrievalCandidate("chunk_43_9", RetrievalSource.LEXICAL_FTS5, rank=2, raw_score=3.0),
    ]
    fused = fuse_candidates_rrf(candidates, rrf_k=60)

    # Os 2 alvos granulares permanecem distintos e NENHUM candidato sem target_ref é criado!
    chunk_10_entries = [acc for acc in fused if acc.chunk_id == "chunk_43_10"]
    assert len(chunk_10_entries) == 2
    assert {acc.target_ref.canonical_label for acc in chunk_10_entries if acc.target_ref} == {
        "43.10(a)(1)",
        "43.10(a)(2)",
    }
    for acc in chunk_10_entries:
        assert acc.structural_rank == 1
        assert acc.lexical_rank == 1
        assert acc.vector_rank == 1
        assert acc.target_ref is not None


@pytest.mark.unit
def test_resolve_structural_references_avoids_phantom_repeated_digit_matches() -> None:
    """Impede match espúrio `7.3` dentro de `RBAC 77.3 Emenda 2` por backtracking de regex."""
    refs_77 = resolve_structural_references("RBAC 77.3 Emenda 2")
    assert len(refs_77) == 1
    assert refs_77[0].regulation_number == "77"
    assert refs_77[0].section_number == "77.3"
    assert refs_77[0].amendment_number == "2"

    refs_11 = resolve_structural_references("RBAC 11.1(a)(1) Emenda 01")
    assert len(refs_11) == 1
    assert refs_11[0].regulation_number == "11"
    assert refs_11[0].section_number == "11.1"
    assert refs_11[0].paragraph_number == "(a)"
    assert refs_11[0].item_number == "(1)"

    refs_spaced = resolve_structural_references("RBAC 43, seção 43.10 (a)(1) Emenda 05")
    assert len(refs_spaced) == 1
    assert refs_spaced[0].canonical_label == "43.10(a)(1)"
    assert refs_spaced[0].amendment_number == "05"
