"""Testes unitários das estratégias estruturais (ANAC RBAC e fallback genérico)."""

from __future__ import annotations

import hashlib

import pytest

from rbac_assistant.domain.entities.normative_unit import (
    ExtractionMethod,
    NormativeUnitType,
    StructureStatus,
)
from rbac_assistant.domain.entities.regulatory_document import DocumentType
from rbac_assistant.domain.entities.regulatory_page import (
    PageValidationStatus,
    RegulatoryPage,
)
from rbac_assistant.infrastructure.pdf.anac_structure_strategy import AnacRbacStructureStrategy
from rbac_assistant.infrastructure.pdf.generic_structure_strategy import GenericStructureStrategy


def _page(num: int, text: str) -> RegulatoryPage:
    return RegulatoryPage(
        page_id=f"v::p{num:04d}",
        version_id="v",
        pdf_page_number=num,
        printed_page_number=f"{num}/10",
        original_text=text,
        normalized_text=text,
        extraction_method=ExtractionMethod.NATIVE_PDF,
        validation_status=PageValidationStatus.NATIVE_QUALITY_CHECKED,
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


@pytest.mark.unit
def test_anac_strategy_skips_toc_parses_multipage_sections_and_hierarchy() -> None:
    p1 = _page(
        1,
        "\n".join(
            [
                "REGULAMENTO BRASILEIRO DA AVIAÇÃO CIVIL",
                "RBAC nº 43",
                "43.1 Aplicabilidade",
                "43.3 Pessoas autorizadas a executar manutenção,",
                "manutenção preventiva e alteração",
            ]
        ),
    )
    p2 = _page(
        2,
        "\n".join(
            [
                "Data da emissão: 15 de março de 2021 RBAC nº 43",
                "Data de vigência: 26 de maio de 2021 Emenda nº 05",
                "43.1 Aplicabilidade",
                "(a) Este regulamento estabelece regras para manutenção.",
                "(1) aeronave com certificado brasileiro;",
                "(i) célula e motor.",
                "43.3 Pessoas autorizadas a executar manutenção",
                "(a) Nenhuma pessoa pode executar manutenção sem autorização.",
                "43.17 podem executar serviços conforme previsto no RBAC 145.",
                "Origem: SAR 2/10",
            ]
        ),
    )
    p3 = _page(
        3,
        "\n".join(
            [
                "Data da emissão: 15 de março de 2021 RBAC nº 43",
                "Data de vigência: 26 de maio de 2021 Emenda nº 05",
                "(b) Continuação da seção 43.3 na página seguinte.",
                "(A) Subitem específico de verificação.",
                "APÊNDICE A DO RBAC 43",
                "A43.1 Grandes alterações",
                "(a) Grandes alterações de célula.",
                "Origem: SAR 3/10",
            ]
        ),
    )

    strategy = AnacRbacStructureStrategy()
    assert strategy.supports(authority="ANAC", document_type=DocumentType.RBAC) is True
    assert strategy.supports(authority="ANAC", document_type=DocumentType.RESOLUCAO) is False

    units = strategy.parse(version_id="v", regulation_number="43", pages=[p1, p2, p3])
    roots = {u.section_number: u for u in units if u.parent_unit_id is None}

    assert set(roots.keys()) == {"43.1", "43.3", "A43.1"}
    # A linha "43.17 podem..." em minúscula NÃO criou uma seção falsa!
    assert "43.17" not in roots
    assert roots["43.3"].page_numbers == (2, 3)
    assert roots["43.3"].start_page == 2
    assert roots["43.3"].end_page == 3
    assert any(r.target_regulation == "RBAC 145" for r in roots["43.3"].references)
    assert roots["A43.1"].unit_type == NormativeUnitType.APPENDIX

    unit_types = {u.unit_type for u in units}
    assert NormativeUnitType.PARAGRAPH in unit_types
    assert NormativeUnitType.INCISO in unit_types
    assert NormativeUnitType.ALINEA in unit_types
    assert NormativeUnitType.SUBITEM in unit_types


@pytest.mark.unit
def test_anac_strategy_marks_unstructured_text_as_pending_validation() -> None:
    p1 = _page(1, "Texto livre sem marcação clara de seção normativa.")
    units = AnacRbacStructureStrategy().parse(version_id="v", regulation_number="43", pages=[p1])
    assert len(units) == 1
    assert units[0].structure_status == StructureStatus.PENDING_VALIDATION


@pytest.mark.unit
def test_generic_strategy_parses_resolutions_and_falls_back_safely() -> None:
    p1 = _page(
        1,
        "Art. 1º Aprovar requisitos de teste.\nParágrafo único. Aplica-se.\n"
        "Art. 2º Entra em vigor hoje.",
    )
    units = GenericStructureStrategy().parse(
        version_id="res700", regulation_number="700", pages=[p1]
    )
    assert [u.section_number for u in units] == ["Art. 1º", "Art. 2º"]
    assert all(u.structure_status == StructureStatus.PARSED for u in units)

    p_unstructured = _page(1, "Documento técnico corrido sem numeração de artigos.")
    fallback = GenericStructureStrategy().parse(
        version_id="pub1", regulation_number="1", pages=[p_unstructured]
    )
    assert len(fallback) == 1
    assert fallback[0].structure_status == StructureStatus.PENDING_VALIDATION
