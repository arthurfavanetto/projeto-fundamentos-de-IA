"""Testes unitários de correções verificadas por SHA-256, qualidade textual, tabelas e estrutura."""

from __future__ import annotations

import hashlib

import pytest

from rbac_assistant.domain.entities.normative_unit import (
    ExtractionMethod,
    NormativeTable,
    NormativeUnitType,
    TableCellGeometry,
    render_structured_table,
    validate_table_integrity,
)
from rbac_assistant.domain.entities.regulatory_page import (
    PageValidationStatus,
    RegulatoryPage,
    compute_page_content_hash,
)
from rbac_assistant.domain.ports.pdf_extractor_port import ExtractedTableData
from rbac_assistant.infrastructure.pdf.anac_structure_strategy import AnacRbacStructureStrategy
from rbac_assistant.infrastructure.pdf.text_quality_checker import TextQualityChecker
from rbac_assistant.infrastructure.pdf.verified_corrections import (
    RBAC43_EMD05_SHA256,
    VerifiedTextCorrectionRule,
    apply_verified_page_corrections,
)
from rbac_assistant.rag.ingestion.structural_chunker import StructuralNormativeChunker
from rbac_assistant.rag.retrieval.section_resolver import resolve_structural_references


@pytest.mark.unit
def test_verified_corrections_apply_all_six_rbac43_fixes_and_preserve_legitimate_plus_and_no() -> (
    None
):
    # Página 12: corrige 'Lei no 7565' preservando a preposição 'no art. 120'
    p12_text = (
        "aeronaves perecidas conforme definido no art. 120 da Lei no 7565, "
        "de 19 de dezembro de 1986, no território nacional."
    )
    out12 = apply_verified_page_corrections(
        source_sha256=RBAC43_EMD05_SHA256,
        pdf_page_number=12,
        original_text=p12_text,
        tables=(),
    )
    assert out12.rolled_back is False
    assert out12.correction_issues == ()
    assert "no art. 120 da Lei nº 7565, de 19 de dezembro de 1986," in out12.original_text
    assert "no território nacional." in out12.original_text
    assert "Lei no 7565" not in out12.original_text
    assert len(out12.applied_corrections) == 1
    assert out12.applied_corrections[0].validation_result == "applied"
    assert out12.applied_corrections[0].text_replacements == 1

    # Idempotência na segunda execução
    out12_again = apply_verified_page_corrections(
        source_sha256=RBAC43_EMD05_SHA256,
        pdf_page_number=12,
        original_text=out12.original_text,
        tables=(),
    )
    assert out12_again.rolled_back is False
    assert out12_again.correction_issues == ()
    assert out12_again.original_text == out12.original_text
    assert out12_again.applied_corrections[0].validation_result == "already_corrected"
    assert out12_again.applied_corrections[0].text_replacements == 0

    # Página 21: corrige 'No Tel' -> 'Nº Tel' no texto, célula e geometria do formulário
    p21_text = "Nome da Organização de Manutenção No Tel\nEndereço da Organização de Manutenção"
    p21_table = ExtractedTableData(
        pdf_page_number=21,
        rows=(("Nome da Organização de Manutenção", "No Tel"),),
        cells=(
            TableCellGeometry(row_index=0, col_index=0, text="Nome da Organização de Manutenção"),
            TableCellGeometry(row_index=0, col_index=1, text="No Tel"),
        ),
        table_kind="form",
        header_row_count=0,
        visual_layout_preserved=False,
    )
    out21 = apply_verified_page_corrections(
        source_sha256=RBAC43_EMD05_SHA256,
        pdf_page_number=21,
        original_text=p21_text,
        tables=(p21_table,),
    )
    assert out21.rolled_back is False
    assert out21.correction_issues == ()
    assert "Nome da Organização de Manutenção Nº Tel" in out21.original_text
    assert out21.tables[0].rows[0][1] == "Nº Tel"
    assert out21.tables[0].cells[1].text == "Nº Tel"
    assert out21.applied_corrections[0].table_cell_replacements == 1

    # Página 31: corrige as 4 ocorrências de '+' -> '±' preservando os '±' já existentes
    p31_text = (
        "(1) frequência de resposta do sistema é de 1.090+3 MHz.\n"
        "(2) frequência de resposta é 1.090±3 MHz.\n"
        "(3) resposta de 1.090±1 MHz.\n"
        "(b)(2) com uma razão de 235+5 interrogações por segundo.\n"
        "(c)(1) do receptor do sistema é -73+4 dBm ou que para\n"
        "(tipo P6), é -74+3 dBm usando um equipamento de teste:"
    )
    out31 = apply_verified_page_corrections(
        source_sha256=RBAC43_EMD05_SHA256,
        pdf_page_number=31,
        original_text=p31_text,
        tables=(),
    )
    assert out31.rolled_back is False
    assert out31.correction_issues == ()
    assert len(out31.applied_corrections) == 4
    for expected in (
        "1.090±3 MHz.",
        "1.090±1 MHz.",
        "235±5 interrogações por",
        "-73±4 dBm ou que para",
        "-74±3 dBm usando um equipamento",
    ):
        assert expected in out31.original_text
    assert "+" not in out31.original_text

    # Página 30: valores positivos legítimos (+70, +21,3, +531, +893, +974) permanecem inalterados
    p30_text = (
        "1.000 304,8 +70 +21,3\n"
        "30,50 1.032,9 +531 161,8\n"
        "30,90 1.046,4 +893 272,2\n"
        "30,99 1.049,5 +974 296,9"
    )
    out30 = apply_verified_page_corrections(
        source_sha256=RBAC43_EMD05_SHA256,
        pdf_page_number=30,
        original_text=p30_text,
        tables=(),
    )
    assert out30.original_text == p30_text
    assert out30.applied_corrections == ()


@pytest.mark.unit
def test_verified_corrections_never_apply_on_wrong_sha256_and_rollback_atomically_on_errors() -> (
    None
):
    # 1. SHA-256 diferente -> jamais aplica correção mesmo com texto e página idênticos
    wrong_sha = "f" * 64
    p12_text = "no art. 120 da Lei no 7565, de 19 de dezembro"
    out_wrong_sha = apply_verified_page_corrections(
        source_sha256=wrong_sha,
        pdf_page_number=12,
        original_text=p12_text,
        tables=(),
    )
    assert out_wrong_sha.original_text == p12_text
    assert out_wrong_sha.applied_corrections == ()
    assert out_wrong_sha.correction_issues == ()
    assert out_wrong_sha.rolled_back is False

    # 2. Contexto ausente -> bloqueia com diagnóstico e mantém texto intacto
    out_missing = apply_verified_page_corrections(
        source_sha256=RBAC43_EMD05_SHA256,
        pdf_page_number=12,
        original_text="Texto totalmente diferente na página 12.",
        tables=(),
    )
    assert out_missing.rolled_back is True
    assert out_missing.applied_corrections == ()
    assert any("ausente" in msg for msg in out_missing.correction_issues)

    # 3. Contexto duplicado -> bloqueia com rollback
    out_dup = apply_verified_page_corrections(
        source_sha256=RBAC43_EMD05_SHA256,
        pdf_page_number=12,
        original_text=f"{p12_text}\n{p12_text}",
        tables=(),
    )
    assert out_dup.rolled_back is True
    assert out_dup.original_text == f"{p12_text}\n{p12_text}"
    assert any("divergente" in msg for msg in out_dup.correction_issues)

    # 4. Contexto ambíguo (contém original E corrigido simultaneamente) -> bloqueia com rollback
    ambiguous = (
        "no art. 120 da Lei no 7565, de 19 de dezembro\n"
        "no art. 120 da Lei nº 7565, de 19 de dezembro"
    )
    out_ambig = apply_verified_page_corrections(
        source_sha256=RBAC43_EMD05_SHA256,
        pdf_page_number=12,
        original_text=ambiguous,
        tables=(),
    )
    assert out_ambig.rolled_back is True
    assert out_ambig.original_text == ambiguous
    assert any("ambígua" in msg for msg in out_ambig.correction_issues)

    # 5. Falha parcial na p. 31 (3 regras presentes, 4ª ausente) -> rollback atômico total!
    partial_p31 = (
        "frequência de resposta do sistema é de 1.090+3 MHz.\n"
        "com uma razão de 235+5 interrogações por segundo.\n"
        "do receptor do sistema é -73+4 dBm ou que para\n"
        "linha sem a quarta regra esperada."
    )
    out_partial = apply_verified_page_corrections(
        source_sha256=RBAC43_EMD05_SHA256,
        pdf_page_number=31,
        original_text=partial_p31,
        tables=(),
    )
    assert out_partial.rolled_back is True
    assert out_partial.original_text == partial_p31
    assert "1.090±3 MHz." not in out_partial.original_text
    assert out_partial.applied_corrections == ()
    assert len(out_partial.correction_issues) == 1

    # 6. Falha na célula da tabela da p. 21 -> rollback também do texto da p. 21
    out_p21_no_table = apply_verified_page_corrections(
        source_sha256=RBAC43_EMD05_SHA256,
        pdf_page_number=21,
        original_text="Nome da Organização de Manutenção No Tel",
        tables=(),
    )
    assert out_p21_no_table.rolled_back is True
    assert out_p21_no_table.original_text == "Nome da Organização de Manutenção No Tel"
    assert any("célula" in msg for msg in out_p21_no_table.correction_issues)

    # 7. Validação de criação de regra inválida
    with pytest.raises(ValueError, match="64 caracteres"):
        VerifiedTextCorrectionRule(
            correction_id="bad",
            source_sha256="short",
            document_reference="doc",
            pdf_page_number=1,
            original_context="a",
            corrected_context="b",
        )


@pytest.mark.unit
def test_text_quality_checker_flags_glyph_tolerance_ordinal_and_table_defects_fail_closed() -> (
    None
):
    checker = TextQualityChecker()

    # Texto legítimo com sinais positivos (+70, +21,3, +531, +893, +974), ±, preposição 'no' e 'nº'
    clean_text = (
        "TABELA III Atrito 1.000 304,8 +70 +21,3 TABELA IV 30,50 +531 30,90 +893 30,99 +974. "
        "Conferido no art. 120 da Lei nº 7565 e frequência 1.090±3 MHz, -73±4 dBm e Nº Tel."
    )
    res_clean = checker.evaluate(original_text=clean_text, normalized_text=clean_text)
    assert res_clean.passed_automated_checks is True
    assert res_clean.issues == ()

    # Falha com tolerância corrompida ('1.090+3 MHz'), ordinal corrompido ('Lei no 7565'), CID, PUA
    for bad_text, expected_cue in (
        ("Verifique se a frequência de resposta do sistema é de 1.090+3 MHz na bancada.", "±"),
        ("Interrogação com uma razão de 235+5 interrogações por segundo no modo C.", "±"),
        ("O nível mínimo de acionamento do receptor do sistema é -73+4 dBm no teste.", "±"),
        ("Aeronaves perecidas conforme definido no art. 120 da Lei no 7565 de 1986.", "ordinal"),
        ("Dados do formulário: Nome da Organização de Manutenção No Tel e assinatura.", "ordinal"),
        ("Texto com token (cid:42) não mapeado pela fonte do documento oficial ANAC.", "CID"),
        ("Texto com caractere PUA \ue010 extraído de fonte personalizada do PDF ANAC.", "PUA"),
    ):
        res = checker.evaluate(original_text=bad_text, normalized_text=bad_text, min_chars=20)
        assert res.passed_automated_checks is False, f"Deveria falhar para: {bad_text}"
        assert any(expected_cue.lower() in issue.lower() for issue in res.issues)

    # Falha quando há anomalia vetorial não resolvida, erro de correção ou tabela sem cabeçalho
    bad_table = NormativeTable(
        table_id="tbl_broken",
        version_id="v1",
        pdf_page_number=30,
        rows=(("1.000", "304,8", "+70", "+21,3"), ("2.000", "609,6", "70", "21,3")),
        header_row_count=1,
    )
    res_meta = checker.evaluate(
        original_text=clean_text,
        normalized_text=clean_text,
        unresolved_glyph_anomalies=("underlined_plus_operator ('+')",),
        correction_issues=("contexto ausente",),
        tables=(bad_table,),
    )
    assert res_meta.passed_automated_checks is False
    assert len(res_meta.issues) >= 3


@pytest.mark.unit
def test_table_validation_rendering_and_canonical_hash_sensitivity() -> None:
    # Tabela composta com 2 linhas de cabeçalho mesclado (espelhando TABELA III/IV e TABELA II)
    valid_tbl = NormativeTable(
        table_id="tbl_t2",
        version_id="v1",
        pdf_page_number=29,
        caption="TABELA II — Tolerância do Teste",
        rows=(
            ("TESTE", "TOLERÂNCIA", ""),
            ("", "(pés)", "(m)"),
            ("Vazamento da caixa", "± 100", "30,48"),
            ("Histerese", "", ""),
            ("1° ponto (50% de altitude máxima)", "75", "22,86"),
        ),
        cells=(
            TableCellGeometry(row_index=0, col_index=0, col_span=1, text="TESTE"),
            TableCellGeometry(row_index=0, col_index=1, col_span=2, text="TOLERÂNCIA"),
            TableCellGeometry(row_index=1, col_index=0, col_span=1, text="", is_empty=True),
            TableCellGeometry(row_index=1, col_index=1, col_span=1, text="(pés)"),
            TableCellGeometry(row_index=1, col_index=2, col_span=1, text="(m)"),
        ),
        header_row_count=2,
        table_kind="data_table",
        has_merged_cells=True,
    )
    assert validate_table_integrity(valid_tbl) == ()
    rendered = render_structured_table(valid_tbl)
    assert "TABELA II — Tolerância do Teste" in rendered
    assert "Colunas: TESTE | TOLERÂNCIA (pés) | TOLERÂNCIA (m)" in rendered
    assert "TESTE=Vazamento da caixa | TOLERÂNCIA (pés)=± 100 | TOLERÂNCIA (m)=30,48" in rendered
    assert "TESTE=Histerese | TOLERÂNCIA (pés)=(vazio) | TOLERÂNCIA (m)=(vazio)" in rendered

    # Formulário gráfico sinaliza explicitamente ausência de fac-símile visual
    form_tbl = NormativeTable(
        table_id="tbl_form",
        version_id="v1",
        pdf_page_number=21,
        rows=(("Item:", ""), ("Nome da Organização de Manutenção", "Nº Tel")),
        cells=(
            TableCellGeometry(row_index=0, col_index=0, col_span=2, text="Item:"),
            TableCellGeometry(
                row_index=1, col_index=0, col_span=1, text="Nome da Organização de Manutenção"
            ),
            TableCellGeometry(row_index=1, col_index=1, col_span=1, text="Nº Tel"),
        ),
        header_row_count=0,
        table_kind="form",
        has_merged_cells=True,
        visual_layout_preserved=False,
    )
    assert validate_table_integrity(form_tbl) == ()
    rendered_form = render_structured_table(form_tbl)
    assert "FORMULÁRIO ESTRUTURADO" in rendered_form
    assert "fac-símile" in rendered_form
    assert "[mesclado 2 cols]" in rendered_form
    assert "Nº Tel" in rendered_form

    # Tabelas malformadas são rejeitadas fail-closed
    ragged = NormativeTable(
        table_id="tbl_ragged",
        version_id="v1",
        pdf_page_number=29,
        rows=(("A", "B"), ("1",)),
    )
    assert any("inconsistente" in i for i in validate_table_integrity(ragged))

    # Hash canônico reage a 1 caractere no texto ou 1 célula/vazio na tabela
    h1 = compute_page_content_hash("Texto p.21", (form_tbl.rows,))
    h2 = compute_page_content_hash(
        "Texto p.21",
        ((("Item:", ""), ("Nome da Organização de Manutenção", "No Tel")),),
    )
    h3 = compute_page_content_hash(
        "Texto p.21",
        ((("Item:", " "), ("Nome da Organização de Manutenção", "Nº Tel")),),
    )
    assert len({h1, h2, h3}) == 3


@pytest.mark.unit
def test_anac_parser_multiline_titles_appendices_and_roman_alineas_disambiguation() -> None:
    def _pg(num: int, text: str) -> RegulatoryPage:
        return RegulatoryPage(
            page_id=f"v::p{num:04d}",
            version_id="v",
            pdf_page_number=num,
            printed_page_number=f"{num}/5",
            original_text=text,
            normalized_text=text,
            extraction_method=ExtractionMethod.NATIVE_PDF,
            validation_status=PageValidationStatus.NATIVE_QUALITY_CHECKED,
            content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )

    pages = [
        _pg(
            1,
            "\n".join(
                [
                    (
                        "43.5 Aprovação para retorno ao serviço após manutenção, "
                        "manutenção preventiva,"
                    ),
                    "reconstrução e alteração",
                    "Uma pessoa somente pode aprovar o retorno ao serviço se:",
                    "(a) o registro for feito conforme o",
                    "Apêndice D deste regulamento.",
                ]
            ),
        ),
        _pg(
            2,
            "\n".join(
                [
                    "APÊNDICE B DO RBAC 43",
                    "REGISTROS DE GRANDES REPAROS E GRANDES ALTERAÇÕES",
                    "B43.1 Registros de grandes reparos e grandes alterações",
                    "(a) Preencher o formulário padrão.",
                    "APÊNDICE C DO RBAC 43",
                    "[RESERVADO]",
                    "APÊNDICE D DO RBAC 43",
                    "OBJETIVOS E DETALHES DE ITENS A SEREM INCLUÍDOS NAS INSPEÇÕES",
                    "ANUAIS E INSPEÇÕES DE 100 HORAS",
                    (
                        "D43.1 Objetivos e detalhes de itens a serem incluídos nas "
                        "inspeções anuais e de 100"
                    ),
                    "horas (conforme aplicável para aeronaves específicas)",
                    "(a) Remover portas de acesso.",
                ]
            ),
        ),
        _pg(
            3,
            "\n".join(
                [
                    "APÊNDICE F DO RBAC 43",
                    "TESTES E INSPEÇÕES NO TRANSPONDER",
                    "F43.1 Testes e inspeções no transponder",
                    "(a) Radiofrequência",
                    "(1) 1.090±3 MHz.",
                    "(b) Supressão",
                    "(1) pulso P1;",
                    "(2) razão de 235±5 interrogações.",
                    "(c) Sensibilidade do receptor",
                    "(1) nível mínimo -73±4 dBm e -74±3 dBm:",
                    "(i) conectado ao terminal;",
                    "(ii) corrigido; ou",
                    "(iii) irradiado; e",
                    "(2) diferença não excede 1 dB.",
                    "(d) Potência de Pico",
                    "(1) verificar potência:",
                    "(i) 1A;",
                    "(ii) 1B;",
                    "(iii) Mode S 1A;",
                    "(iv) Mode S 1B; e",
                    "(v) máximo.",
                    "(e) Isolamento.",
                ]
            ),
        ),
    ]

    units = AnacRbacStructureStrategy().parse(version_id="v", regulation_number="43", pages=pages)
    roots = {u.section_number: u for u in units if u.parent_unit_id is None}
    assert set(roots.keys()) == {"43.5", "B43.1", "Apêndice C", "D43.1", "F43.1"}

    # Título multi-linha completo SEM absorver "Uma pessoa somente pode..."
    assert roots["43.5"].title == (
        "Aprovação para retorno ao serviço após manutenção, manutenção preventiva, "
        "reconstrução e alteração"
    )
    assert "Uma pessoa somente pode" not in roots["43.5"].title
    # Prosa iniciada por "Apêndice D deste regulamento." não quebrou 43.5!
    assert "Apêndice D deste regulamento." in roots["43.5"].original_text

    # Banners de apêndice não vazaram para seções anteriores
    assert "APÊNDICE C" not in roots["B43.1"].original_text
    assert roots["Apêndice C"].title == "[RESERVADO]"
    assert roots["D43.1"].subpart == "APÊNDICE D DO RBAC 43"
    assert roots["D43.1"].title == (
        "Objetivos e detalhes de itens a serem incluídos nas inspeções anuais e de 100 "
        "horas (conforme aplicável para aeronaves específicas)"
    )
    assert "TESTES E INSPEÇÕES NO TRANSPONDER" not in roots["D43.1"].original_text

    # Parágrafos (c) e (d) em F43.1 NÃO foram classificados como alíneas!
    f43_subs = [u for u in units if u.section_number == "F43.1" and u.parent_unit_id is not None]
    f43_pars = [u.paragraph_number for u in f43_subs if u.unit_type == NormativeUnitType.PARAGRAPH]
    assert f43_pars == ["(a)", "(b)", "(c)", "(d)", "(e)"]
    c_alineas = [
        u.item_number
        for u in f43_subs
        if u.paragraph_number == "(c)" and u.unit_type == NormativeUnitType.ALINEA
    ]
    assert c_alineas == ["(1)(i)", "(1)(ii)", "(1)(iii)"]
    d_alineas = [
        u.item_number
        for u in f43_subs
        if u.paragraph_number == "(d)" and u.unit_type == NormativeUnitType.ALINEA
    ]
    assert d_alineas == ["(1)(i)", "(1)(ii)", "(1)(iii)", "(1)(iv)", "(1)(v)"]

    # Resolver estrutural reconhece F43.1(c)(1)(i) e "Apêndice F do RBAC 43"
    refs_f = resolve_structural_references("Transcreva F43.1(c)(1)(i) do RBAC 43")
    assert any(
        r.section_number == "F43.1"
        and r.paragraph_number == "(c)"
        and r.item_number == "(1)"
        and r.subitem_number == "(i)"
        for r in refs_f
    )
    refs_app = resolve_structural_references("Consulte o Apêndice F do RBAC 43")
    assert any(r.section_number == "F43.1" and r.regulation_number == "43" for r in refs_app)

    # Chunker divide em fronteira de parágrafo preservando 100% dos caracteres
    chunker = StructuralNormativeChunker(max_chunk_tokens=50, chars_per_token=4)
    long_text = (
        "F43.1 Testes\n"
        "(a) " + ("texto do parágrafo a " * 6) + "\n"
        "(b) " + ("texto do parágrafo b " * 6) + "\n"
        "(c) " + ("texto do parágrafo c " * 6)
    )
    slices = chunker._split_preserving_boundaries(long_text)
    assert len(slices) >= 2
    assert "".join(slices) == long_text
    assert any(s.startswith("(b) ") or s.startswith("(c) ") for s in slices[1:])
