"""Correções textuais verificadas, estritamente vinculadas a SHA-256, página e contexto exato.

Princípios obrigatórios (fail-closed):
- Nunca aplicar substituições globais ou heurísticas cegas (`+` -> `±`, `no` -> `nº`).
- Toda regra só é elegível quando o SHA-256 do PDF oficial corresponde exatamente.
- Cada regra exige correspondência exata de página física, contexto literal e contagem esperada.
- A execução é idempotente (`already_corrected` não gera erro nem altera bytes) e atômica por
  página: se qualquer regra da página estiver ausente, duplicada ou ambígua, nenhuma alteração
  parcial é mantida (rollback completo) e o diagnóstico bloqueia a aprovação automatizada.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from rbac_assistant.domain.entities.normative_unit import TableCellGeometry
from rbac_assistant.domain.ports.pdf_extractor_port import ExtractedTableData

RBAC43_EMD05_SHA256 = "5026d2da87fc3f55c61a6b5a20dd1f50d52a1367a8550897f3864074e1a46472"
VERIFIED_CORRECTION_METHOD = "verified_sha256_contextual_correction"


@dataclass(frozen=True, slots=True)
class VerifiedTextCorrectionRule:
    """Regra auditável de correção vinculada a um PDF oficial específico (SHA-256)."""

    correction_id: str
    source_sha256: str
    document_reference: str
    pdf_page_number: int
    original_context: str
    corrected_context: str
    expected_occurrences: int = 1
    table_cell_original: str | None = None
    table_cell_corrected: str | None = None
    expected_table_cell_occurrences: int = 0
    rationale: str = ""
    visual_verification_method: str = "pdf_render_2x_visual_and_vector_underline_audit"

    def __post_init__(self) -> None:
        if len(self.source_sha256.strip()) != 64:
            raise ValueError("source_sha256 deve conter 64 caracteres hexadecimais.")
        if self.pdf_page_number < 1:
            raise ValueError("pdf_page_number deve ser >= 1.")
        if not self.original_context or not self.corrected_context:
            raise ValueError("original_context e corrected_context são obrigatórios.")
        if self.original_context == self.corrected_context:
            raise ValueError("original_context e corrected_context devem diferir.")
        if self.expected_occurrences < 1:
            raise ValueError("expected_occurrences deve ser >= 1.")
        if self.expected_table_cell_occurrences < 0:
            raise ValueError("expected_table_cell_occurrences deve ser >= 0.")


@dataclass(frozen=True, slots=True)
class AppliedCorrectionRecord:
    """Registro de auditoria de uma correção verificada executada em uma página."""

    correction_id: str
    source_sha256: str
    pdf_page_number: int
    original_context: str
    corrected_context: str
    text_replacements: int
    table_cell_replacements: int
    method: str
    validation_result: str
    extractor_version: str

    def to_dict(self) -> dict[str, object]:
        """Serializa o registro de auditoria para JSON em metadados/snapshots."""
        return {
            "correction_id": self.correction_id,
            "source_sha256": self.source_sha256,
            "pdf_page_number": self.pdf_page_number,
            "original_context": self.original_context,
            "corrected_context": self.corrected_context,
            "text_replacements": self.text_replacements,
            "table_cell_replacements": self.table_cell_replacements,
            "method": self.method,
            "validation_result": self.validation_result,
            "extractor_version": self.extractor_version,
        }


@dataclass(frozen=True, slots=True)
class PageCorrectionOutcome:
    """Resultado atômico da aplicação de correções verificadas em uma página."""

    original_text: str
    tables: tuple[ExtractedTableData, ...]
    applied_corrections: tuple[AppliedCorrectionRecord, ...] = ()
    correction_issues: tuple[str, ...] = ()
    rolled_back: bool = False


# Regras verificadas visualmente (renderização 2x) e na camada vetorial (retângulos de
# sublinhado sob 'o' sobrescrito 8.04pt e sob '+' 12.0pt) para o PDF oficial RBAC 43 Emenda 05.
DEFAULT_VERIFIED_CORRECTION_RULES: tuple[VerifiedTextCorrectionRule, ...] = (
    VerifiedTextCorrectionRule(
        correction_id="rbac43_emd05_p12_lei_7565_ordinal",
        source_sha256=RBAC43_EMD05_SHA256,
        document_reference="ANAC RBAC 43 Emenda 05",
        pdf_page_number=12,
        original_context="no art. 120 da Lei no 7565, de 19 de dezembro",
        corrected_context="no art. 120 da Lei nº 7565, de 19 de dezembro",
        expected_occurrences=1,
        rationale=(
            "O PDF oficial compõe 'nº' usando 'o' (U+006F) sobrescrito (8.04pt) com "
            "traço vetorial sublinhado (w=4.08pt, h=0.38pt) na página 12."
        ),
    ),
    VerifiedTextCorrectionRule(
        correction_id="rbac43_emd05_p21_form_no_tel_ordinal",
        source_sha256=RBAC43_EMD05_SHA256,
        document_reference="ANAC RBAC 43 Emenda 05",
        pdf_page_number=21,
        original_context="Nome da Organização de Manutenção No Tel",
        corrected_context="Nome da Organização de Manutenção Nº Tel",
        expected_occurrences=1,
        table_cell_original="No Tel",
        table_cell_corrected="Nº Tel",
        expected_table_cell_occurrences=1,
        rationale=(
            "O PDF oficial compõe 'Nº' no formulário do Apêndice B usando 'o' (U+006F) "
            "sobrescrito (8.04pt) com traço vetorial sublinhado (w=4.08pt, h=0.36pt) na página 21."
        ),
    ),
    VerifiedTextCorrectionRule(
        correction_id="rbac43_emd05_p31_freq_1090_pm3_mhz",
        source_sha256=RBAC43_EMD05_SHA256,
        document_reference="ANAC RBAC 43 Emenda 05",
        pdf_page_number=31,
        original_context="frequência de resposta do sistema é de 1.090+3 MHz.",
        corrected_context="frequência de resposta do sistema é de 1.090±3 MHz.",
        expected_occurrences=1,
        rationale=(
            "O PDF oficial compõe '±' em F43.1(a)(1) usando '+' (U+002B) com retângulo "
            "vetorial sublinhado (w=6.72pt, h=0.60pt) na página 31."
        ),
    ),
    VerifiedTextCorrectionRule(
        correction_id="rbac43_emd05_p31_rate_235_pm5_interrogations",
        source_sha256=RBAC43_EMD05_SHA256,
        document_reference="ANAC RBAC 43 Emenda 05",
        pdf_page_number=31,
        original_context="com uma razão de 235+5 interrogações por",
        corrected_context="com uma razão de 235±5 interrogações por",
        expected_occurrences=1,
        rationale=(
            "O PDF oficial compõe '±' em F43.1(b)(2) usando '+' (U+002B) com retângulo "
            "vetorial sublinhado (w=6.72pt, h=0.60pt) na página 31."
        ),
    ),
    VerifiedTextCorrectionRule(
        correction_id="rbac43_emd05_p31_mtl_minus73_pm4_dbm",
        source_sha256=RBAC43_EMD05_SHA256,
        document_reference="ANAC RBAC 43 Emenda 05",
        pdf_page_number=31,
        original_context="do receptor do sistema é -73+4 dBm ou que para",
        corrected_context="do receptor do sistema é -73±4 dBm ou que para",
        expected_occurrences=1,
        rationale=(
            "O PDF oficial compõe '±' em F43.1(c)(1) (-73±4 dBm) usando '+' (U+002B) com "
            "retângulo vetorial sublinhado (w=6.72pt, h=0.60pt) na página 31."
        ),
    ),
    VerifiedTextCorrectionRule(
        correction_id="rbac43_emd05_p31_mtl_minus74_pm3_dbm",
        source_sha256=RBAC43_EMD05_SHA256,
        document_reference="ANAC RBAC 43 Emenda 05",
        pdf_page_number=31,
        original_context="(tipo P6), é -74+3 dBm usando um equipamento",
        corrected_context="(tipo P6), é -74±3 dBm usando um equipamento",
        expected_occurrences=1,
        rationale=(
            "O PDF oficial compõe '±' em F43.1(c)(1) (-74±3 dBm) usando '+' (U+002B) com "
            "retângulo vetorial sublinhado (w=6.72pt, h=0.60pt) na página 31."
        ),
    ),
)


def _replace_in_tables(
    tables: list[ExtractedTableData],
    *,
    needle: str,
    replacement: str,
) -> tuple[list[ExtractedTableData], int, int]:
    """Substitui `needle` por `replacement` em células de tabela e sua geometria.

    Retorna `(novas_tabelas, total_orig_encontradas, total_ja_corrigidas)`.
    """
    orig_count = sum(cell.count(needle) for tbl in tables for row in tbl.rows for cell in row)
    corr_count = sum(cell.count(replacement) for tbl in tables for row in tbl.rows for cell in row)

    if orig_count == 0:
        return tables, orig_count, corr_count

    updated_tables: list[ExtractedTableData] = []
    for tbl in tables:
        new_rows = tuple(
            tuple(cell.replace(needle, replacement) for cell in row) for row in tbl.rows
        )
        new_cells = tuple(
            TableCellGeometry(
                row_index=c.row_index,
                col_index=c.col_index,
                row_span=c.row_span,
                col_span=c.col_span,
                text=c.text.replace(needle, replacement),
                bbox=c.bbox,
                is_empty=c.is_empty,
            )
            for c in tbl.cells
        )
        updated_tables.append(replace(tbl, rows=new_rows, cells=new_cells))
    return updated_tables, orig_count, corr_count


def apply_verified_page_corrections(
    *,
    source_sha256: str,
    pdf_page_number: int,
    original_text: str,
    tables: Sequence[ExtractedTableData],
    rules: Sequence[VerifiedTextCorrectionRule] = DEFAULT_VERIFIED_CORRECTION_RULES,
    extractor_version: str = "pdfplumber_verified_v1",
) -> PageCorrectionOutcome:
    """Aplica atomicamente todas as correções verificadas elegíveis para a página.

    Garantias:
    - Se `source_sha256` não corresponder exatamente, nenhuma regra é aplicada.
    - Se qualquer regra da página estiver ausente, duplicada ou ambígua, realiza rollback
      integral da página (mantendo texto e tabelas intactos) e retorna `correction_issues`.
    - Se o texto e as tabelas já estiverem no estado corrigido com contagem exata, registra
      `validation_result="already_corrected"` sem erro (idempotência).
    """
    normalized_sha = source_sha256.strip().lower()
    matching_rules = [
        rule
        for rule in rules
        if rule.source_sha256.strip().lower() == normalized_sha
        and rule.pdf_page_number == pdf_page_number
    ]
    if not matching_rules:
        return PageCorrectionOutcome(
            original_text=original_text,
            tables=tuple(tables),
            applied_corrections=(),
            correction_issues=(),
            rolled_back=False,
        )

    working_text = original_text
    working_tables = list(tables)
    records: list[AppliedCorrectionRecord] = []
    errors: list[str] = []

    for rule in matching_rules:
        orig_count = working_text.count(rule.original_context)
        corr_count = working_text.count(rule.corrected_context)
        text_replacements = 0
        text_status = ""

        if orig_count > 0 and corr_count > 0:
            errors.append(
                f"Correção '{rule.correction_id}' (p.{pdf_page_number}) ambígua: texto contém "
                f"{orig_count} ocorrência(s) original(is) e {corr_count} já corrigida(s)."
            )
            continue
        if orig_count == 0 and corr_count == 0:
            errors.append(
                f"Correção '{rule.correction_id}' (p.{pdf_page_number}) não aplicável: "
                "contexto esperado ausente na página."
            )
            continue
        if orig_count > 0:
            if orig_count != rule.expected_occurrences:
                errors.append(
                    f"Correção '{rule.correction_id}' (p.{pdf_page_number}) divergente: "
                    f"encontradas {orig_count} ocorrência(s) originais, esperado "
                    f"{rule.expected_occurrences}."
                )
                continue
            working_text = working_text.replace(rule.original_context, rule.corrected_context)
            text_replacements = orig_count
            text_status = "applied"
        else:
            if corr_count != rule.expected_occurrences:
                errors.append(
                    f"Correção '{rule.correction_id}' (p.{pdf_page_number}) inconsistente: "
                    f"encontradas {corr_count} ocorrência(s) já corrigidas, esperado "
                    f"{rule.expected_occurrences}."
                )
                continue
            text_status = "already_corrected"

        table_replacements = 0
        table_status = text_status
        if (
            rule.table_cell_original is not None
            and rule.table_cell_corrected is not None
            and rule.expected_table_cell_occurrences > 0
        ):
            tbl_orig = sum(
                cell.count(rule.table_cell_original)
                for tbl in working_tables
                for row in tbl.rows
                for cell in row
            )
            tbl_corr = sum(
                cell.count(rule.table_cell_corrected)
                for tbl in working_tables
                for row in tbl.rows
                for cell in row
            )
            if tbl_orig > 0 and tbl_corr > 0:
                errors.append(
                    f"Correção de célula '{rule.correction_id}' (p.{pdf_page_number}) ambígua: "
                    f"{tbl_orig} original(is) e {tbl_corr} já corrigida(s) nas tabelas."
                )
                continue
            if tbl_orig == 0 and tbl_corr == 0:
                errors.append(
                    f"Correção de célula '{rule.correction_id}' (p.{pdf_page_number}) ausente: "
                    f"célula '{rule.table_cell_original}' não encontrada nas tabelas da página."
                )
                continue
            if tbl_orig > 0:
                if tbl_orig != rule.expected_table_cell_occurrences:
                    errors.append(
                        f"Correção de célula '{rule.correction_id}' (p.{pdf_page_number}) "
                        f"divergente: {tbl_orig} encontrada(s), esperado "
                        f"{rule.expected_table_cell_occurrences}."
                    )
                    continue
                working_tables, _, _ = _replace_in_tables(
                    working_tables,
                    needle=rule.table_cell_original,
                    replacement=rule.table_cell_corrected,
                )
                table_replacements = tbl_orig
                table_status = "applied"
            else:
                if tbl_corr != rule.expected_table_cell_occurrences:
                    errors.append(
                        f"Correção de célula '{rule.correction_id}' (p.{pdf_page_number}) "
                        f"inconsistente: {tbl_corr} já corrigida(s), esperado "
                        f"{rule.expected_table_cell_occurrences}."
                    )
                    continue
                table_status = "already_corrected"

        overall_result = (
            "already_corrected"
            if text_status == "already_corrected" and table_status == "already_corrected"
            else "applied"
        )
        records.append(
            AppliedCorrectionRecord(
                correction_id=rule.correction_id,
                source_sha256=normalized_sha,
                pdf_page_number=pdf_page_number,
                original_context=rule.original_context,
                corrected_context=rule.corrected_context,
                text_replacements=text_replacements,
                table_cell_replacements=table_replacements,
                method=VERIFIED_CORRECTION_METHOD,
                validation_result=overall_result,
                extractor_version=extractor_version,
            )
        )

    if errors:
        # Rollback atômico: nenhuma modificação parcial pode sobreviver em caso de falha!
        return PageCorrectionOutcome(
            original_text=original_text,
            tables=tuple(tables),
            applied_corrections=(),
            correction_issues=tuple(errors),
            rolled_back=True,
        )

    return PageCorrectionOutcome(
        original_text=working_text,
        tables=tuple(working_tables),
        applied_corrections=tuple(records),
        correction_issues=(),
        rolled_back=False,
    )
