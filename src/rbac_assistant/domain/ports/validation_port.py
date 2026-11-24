"""Contratos (Ports) para verificação de qualidade textual e estratégias de parsing estrutural."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from rbac_assistant.domain.entities.normative_unit import (
    NormativeReference,
    NormativeTable,
    NormativeUnitType,
    StructureStatus,
)
from rbac_assistant.domain.entities.regulatory_document import DocumentType
from rbac_assistant.domain.entities.regulatory_page import QualityMetrics, RegulatoryPage


@dataclass(frozen=True, slots=True)
class ParsedUnitRecord:
    """Unidade normativa estruturada produzida por uma estratégia de parsing documental."""

    unit_id: str
    version_id: str
    parent_unit_id: str | None
    unit_type: NormativeUnitType
    section_number: str
    subsection_number: str
    paragraph_number: str
    item_number: str
    title: str
    subpart: str
    original_text: str
    normalized_text: str
    start_page: int
    end_page: int
    page_numbers: tuple[int, ...]
    order_index: int
    structure_status: StructureStatus
    content_hash: str
    references: tuple[NormativeReference, ...] = field(default_factory=tuple)


class TextQualityCheckerPort(Protocol):
    """Interface para verificações automatizadas de integridade textual por página."""

    def evaluate(
        self,
        *,
        original_text: str,
        normalized_text: str,
        min_chars: int = 50,
        unresolved_glyph_anomalies: Sequence[str] = (),
        correction_issues: Sequence[str] = (),
        tables: Sequence[NormativeTable] = (),
        table_ownership_issues: Sequence[str] = (),
        expected_incoming_open_parens: int = 0,
        allow_outgoing_open_parens: int = 0,
    ) -> QualityMetrics:
        """Avalia qualidade e integridade de página extraída (não equivale a revisão humana)."""


class StructureParserStrategy(Protocol):
    """Interface Strategy para extração hierárquica de unidades normativas por formato."""

    def supports(self, *, authority: str, document_type: DocumentType) -> bool:
        """Indica se a estratégia atende ao par (autoridade, tipo documental)."""

    def parse(
        self,
        *,
        version_id: str,
        regulation_number: str,
        pages: Sequence[RegulatoryPage],
        tables: Sequence[NormativeTable] = (),
    ) -> list[ParsedUnitRecord]:
        """Estrutura páginas ordenadas em unidades normativas com mapeamento multi-página."""
