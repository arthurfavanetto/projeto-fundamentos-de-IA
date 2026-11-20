"""Fábrica de PDFs sintéticos determinísticos (via reportlab BSD) para testes automatizados.

NOTA: Os documentos gerados por este módulo são fixtures sintéticas de teste —
NUNCA representam normas oficiais da ANAC.
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle


def _draw_lines(
    pdf_canvas: canvas.Canvas,
    lines: list[str],
    *,
    start_y: float = 780.0,
    line_height: float = 15.0,
) -> float:
    y = start_y
    for line in lines:
        pdf_canvas.drawString(50, y, line)
        y -= line_height
    return y


def create_synthetic_rbac_pdf(
    target_path: Path,
    *,
    regulation_number: str = "43",
    amendment_number: str = "05",
    include_blank_ocr_page: bool = False,
    section_text_override: str | None = None,
) -> Path:
    """Gera PDF sintético multi-página no estilo ANAC com acentuação, tabela e página deslocada."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(target_path), pagesize=A4)

    # Página física 1 — Capa e Sumário (sem rodapé de página impressa padrão)
    c.setFont("Helvetica-Bold", 12)
    _draw_lines(
        c,
        [
            "REGULAMENTO BRASILEIRO DA AVIAÇÃO CIVIL",
            f"RBAC nº {regulation_number} — EMENDA nº {amendment_number}",
            "SUMÁRIO",
            f"{regulation_number}.1 Aplicabilidade",
            f"{regulation_number}.3 Pessoas autorizadas a executar manutenção",
            f"{regulation_number}.9 Conteúdo, forma e disposição de registros",
        ],
    )
    c.showPage()

    # Página física 2 — Página impressa deslocada ("10/25") com seção multi-página e acentuação
    c.setFont("Helvetica", 10)
    sec3_intro = section_text_override or (
        "(a) Exceto como previsto nesta seção, nenhuma pessoa pode executar manutenção, "
        "manutenção preventiva, reconstrução ou alteração em um artigo aplicável."
    )
    _draw_lines(
        c,
        [
            f"Data da emissão: 15 de março de 2021 RBAC nº {regulation_number}",
            f"Data de vigência: 26 de maio de 2021 Emenda nº {amendment_number}",
            f"{regulation_number}.1 Aplicabilidade",
            "(a) Este regulamento estabelece regras de manutenção aeronáutica e inspeção.",
            "(1) aeronave com certificado de aeronavegabilidade brasileiro;",
            "(i) célula, motor, hélice, acessório e componentes.",
            f"{regulation_number}.3 Pessoas autorizadas a executar manutenção",
            sec3_intro,
            "(b) Um mecânico de manutenção aeronáutica habilitado pela ANAC pode executar:",
            "(1) serviços previstos no RBAC 145 e na seção 43.9;",
            "Origem: SAR 10/25",
        ],
    )
    c.showPage()

    # Página física 3 — Continuação da seção X.3 + seção X.9 + tabela real + rodapé
    c.setFont("Helvetica", 10)
    y_after = _draw_lines(
        c,
        [
            f"Data da emissão: 15 de março de 2021 RBAC nº {regulation_number}",
            f"Data de vigência: 26 de maio de 2021 Emenda nº {amendment_number}",
            "(c) Uma organização de manutenção certificada pode aprovar o retorno ao serviço.",
            "(A) observando documentação técnica aprovada e registros obrigatórios.",
            f"{regulation_number}.9 Conteúdo, forma e disposição de registros",
            "(a) Cada pessoa que executar manutenção deve registrar data, descrição e assinatura.",
        ],
    )

    table = Table(
        [
            ["Item", "Descrição", "Referência"],
            ["01", "Inspeção periódica", f"Seção {regulation_number}.9"],
            ["02", "Retorno ao serviço", f"Seção {regulation_number}.3"],
        ],
        colWidths=[60, 200, 140],
    )
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.75, colors.black),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]
        )
    )
    table.wrapOn(c, 450, 200)
    table.drawOn(c, 50, max(120.0, y_after - 110.0))
    c.setFont("Helvetica", 10)
    c.drawString(50, 40, "Origem: SAR 11/25")
    c.showPage()

    if include_blank_ocr_page:
        # Página física 4 — Sem texto nativo (simula página digitalizada que exige OCR)
        c.rect(80, 300, 400, 250, stroke=1, fill=0)
        c.showPage()

    c.save()
    return target_path


def create_synthetic_resolution_pdf(target_path: Path) -> Path:
    """Gera PDF sintético no formato de Resolução (Art. 1º, Art. 2º) para testar fallback."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(target_path), pagesize=A4)
    c.setFont("Helvetica", 10)
    _draw_lines(
        c,
        [
            "RESOLUÇÃO FICTÍCIA Nº 700, DE 10 DE JANEIRO DE 2026",
            "Art. 1º Fica aprovada a norma de teste para validação da plataforma multidocumento.",
            "Parágrafo único. Este documento é meramente fictício para testes automatizados.",
            "Art. 2º Esta Resolução entra em vigor na data de sua publicação oficial.",
            "Página 1 de 1",
        ],
    )
    c.showPage()
    c.save()
    return target_path
