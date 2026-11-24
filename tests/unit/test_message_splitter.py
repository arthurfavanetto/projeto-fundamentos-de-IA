"""Testes unitários do `MessageSplitter` (limite 4096 UTF-16, reconstrução 100% fiel e Unicode)."""

from __future__ import annotations

import pytest

from rbac_assistant.interfaces.telegram.message_splitter import (
    MessageSplitter,
    strip_pagination_header,
    utf16_length,
)


@pytest.mark.unit
def test_message_splitter_keeps_short_message_single_part() -> None:
    splitter = MessageSplitter(max_message_length=4096)
    text = "RBAC 43.3 — Pessoas autorizadas a executar manutenção."
    parts = splitter.split(text)
    assert parts == (text,)


@pytest.mark.unit
def test_message_splitter_reconstructs_100_percent_exact_normative_text() -> None:
    splitter = MessageSplitter(max_message_length=256)
    paragraphs = [
        f"43.{i} Dispositivo normativo — manutenção preventiva, reconstrução e alteração "
        f"(a)(1)(i); pontuação: exata, com acentuação: aeronavegabilidade, inspeção, hélice.\n"
        for i in range(1, 35)
    ]
    original = "\n".join(paragraphs)
    parts = splitter.split(original)

    assert len(parts) > 5
    for idx, part in enumerate(parts, start=1):
        assert part.startswith(f"[Parte {idx}/{len(parts)}]\n")
        assert utf16_length(part) <= 256

    reconstructed = "".join(strip_pagination_header(p) for p in parts)
    assert reconstructed == original


@pytest.mark.unit
def test_message_splitter_handles_unicode_emojis_utf16_and_giant_tokens() -> None:
    splitter = MessageSplitter(max_message_length=128)
    # Emojis ocupam 2 code units em UTF-16 cada
    emoji_text = "✈️ Regulamento ANAC RBAC 43 " + ("🛩️" * 90) + "\n" + ("X" * 300)
    parts = splitter.split(emoji_text)

    assert len(parts) >= 4
    for part in parts:
        assert utf16_length(part) <= 128

    reconstructed = "".join(strip_pagination_header(p) for p in parts)
    assert reconstructed == emoji_text


@pytest.mark.unit
def test_message_splitter_sanitizes_null_bytes_and_redacts_secrets() -> None:
    splitter = MessageSplitter(max_message_length=4096)
    fake_token = "123456789:AAFakeTelegramTokenSecret0123456789012"
    fake_api_key = "AIzaSyFakeSecretApiKeyNeverReal1234567890"
    raw = f"Texto\x00 limpo com {fake_token} e {fake_api_key}\r\nLinha 2"

    parts = splitter.split(raw)
    assert len(parts) == 1
    assert "\x00" not in parts[0]
    assert "\r" not in parts[0]
    assert fake_token not in parts[0]
    assert fake_api_key not in parts[0]
    assert parts[0].count("[REDACTED]") == 2


@pytest.mark.unit
def test_message_splitter_verbatim_preserves_unicode_crlf_and_blocks_nul_or_secrets() -> None:
    """Seção 6: prova caractere-a-caractere SEM sanitização nos dois lados e bloqueio de NUL."""
    from rbac_assistant.domain.entities.rag import StructuredNormativeAnswer
    from rbac_assistant.rag.generation.response_formatter import (
        LITERAL_END_MARKER,
        LITERAL_START_MARKER,
        extract_literal_excerpts_from_text,
        render_structured_answer_text,
    )

    # 1. Todos os caracteres críticos preservados bit-a-bit (sem sanitize_telegram_text!)
    excerpt_1 = (
        "F43.1 Testes e inspeções — Lei nº 7565 (1° ponto): "
        "1.090±3 MHz, +70 pés, +21,3 m, -73±4 dBm, -74±3 dBm.\r\n"
        '  Aspas "duplas", \'simples\', “tipográficas”, travessão – e —, parênteses (a)(1)(i).\n'
        "  Espaços   múltiplos preservados e Unicode fora do plano básico: ✈️ 🛩️ 🛫 𝕬𝕭𝕮.\n"
        + ("PALAVRAGIGANTESEMESPACOS" * 18)
        + "🛫" * 35
        + "\r\nFim exato do excerto 1."
    )
    excerpt_2 = (
        "E43.1 Pressão – diferença de altitude: 30,99 -> +974 pés (polegadas deHg).\n"
        + ("ÁÉÍÓÚÂÊÔÃÕÇ±°º" * 20)
    )
    normative_block = (
        "[IDENTIFICAÇÃO EXTERNA — RBAC 43 | Emenda 05 | Seção F43.1]\n"
        f"{LITERAL_START_MARKER}\n{excerpt_1}\n{LITERAL_END_MARKER}\n\n"
        "[IDENTIFICAÇÃO EXTERNA — RBAC 43 | Emenda 05 | Seção E43.1]\n"
        f"{LITERAL_START_MARKER}\n{excerpt_2}\n{LITERAL_END_MARKER}"
    )
    table_block = (
        "=== TABELAS E FORMULÁRIOS ESTRUTURADOS (SQLite SSOT) ===\n"
        "TABELA III — Atrito\n"
        "  - ALTITUDE (pés)=1.000 | TOLERÂNCIA (pés)=+70 | TOLERÂNCIA (m)=+21,3"
    )
    answer = StructuredNormativeAnswer(
        query="RBAC 43 F43.1",
        found=True,
        is_literal_validated=True,
        normative_text=normative_block,
        preliminary_excerpt=None,
        explanation="",
        references=("ANAC RBAC 43 (Emenda 05) — Seção F43.1 (p. 31)",),
        observations=("Teste de separação externa.",),
        formatted_text=render_structured_answer_text(
            normative_text=normative_block,
            preliminary_excerpt=None,
            explanation="",
            references=("ANAC RBAC 43 (Emenda 05) — Seção F43.1 (p. 31)",),
            observations=("Teste de separação externa.",),
            structured_tables_text=table_block,
            include_explanation_block=False,
        ),
        used_fallback_without_llm=True,
        generation_model=None,
        structured_tables_text=table_block,
        literal_excerpts=(excerpt_1, excerpt_2),
    )

    # Testa com limite pequeno (160 UTF-16) forçando cortes no meio de palavra e perto de surrogates
    splitter_small = MessageSplitter(max_message_length=160)
    parts_small = splitter_small.split_normative_answer(answer)
    assert len(parts_small) >= 8
    for part in parts_small:
        assert utf16_length(part) <= 160

    # Reconstrução RAW sem aplicar sanitize_telegram_text em nenhum dos lados!
    reconstructed = "".join(strip_pagination_header(p) for p in parts_small)
    assert reconstructed == answer.formatted_text
    assert "=== EXPLICAÇÃO (GERADA POR IA) ===" not in reconstructed
    assert "=== TABELAS E FORMULÁRIOS ESTRUTURADOS (SQLite SSOT) ===" in reconstructed
    assert "=== REFERÊNCIA ===" in reconstructed

    extracted_literals = extract_literal_excerpts_from_text(reconstructed)
    assert extracted_literals == (excerpt_1, excerpt_2)
    assert "\r\n" in extracted_literals[0]  # CRLF preservado sem normalização silenciosa!

    # Testa também acima do limite real de 4096 UTF-16 do Telegram
    huge_literal = (excerpt_1 + "\n") * 12
    assert utf16_length(huge_literal) > 4096
    splitter_real = MessageSplitter(max_message_length=4096)
    huge_parts = splitter_real.split_verbatim(huge_literal)
    assert len(huge_parts) >= 2
    assert all(utf16_length(p) <= 4096 for p in huge_parts)
    assert "".join(strip_pagination_header(p) for p in huge_parts) == huge_literal

    # 2. Byte NUL (\x00) no texto literal autoritativo -> BLOQUEIO FAIL-CLOSED (jamais remove!)
    nul_excerpt = "43.3 Texto com byte nulo \x00 corrompido."
    with pytest.raises(ValueError, match="NUL"):
        splitter_real.split_verbatim(nul_excerpt)

    nul_answer = StructuredNormativeAnswer(
        query="RBAC 43.3",
        found=True,
        is_literal_validated=True,
        normative_text=(
            f"[IDENTIFICAÇÃO EXTERNA — RBAC 43]\n"
            f"{LITERAL_START_MARKER}\n{nul_excerpt}\n{LITERAL_END_MARKER}"
        ),
        preliminary_excerpt=None,
        explanation="",
        references=("ANAC RBAC 43",),
        observations=(),
        formatted_text="",
        used_fallback_without_llm=True,
        generation_model=None,
        literal_excerpts=(nul_excerpt,),
    )
    blocked_parts = splitter_real.split_normative_answer(nul_answer)
    blocked_joined = "".join(strip_pagination_header(p) for p in blocked_parts)
    assert "BLOQUEIO DE FIDELIDADE LITERAL NO ENVIO" in blocked_joined
    assert "NUL" in blocked_joined
    assert LITERAL_START_MARKER not in blocked_joined
    assert "Texto com byte nulo" not in blocked_joined

    # 3. Padrão que acionaria redação dentro de texto literal -> BLOQUEIO FAIL-CLOSED!
    secret_excerpt = "Texto com token 123456789:AAFakeTelegramTokenSecret0123456789012"
    with pytest.raises(ValueError, match="redação"):
        splitter_real.split_verbatim(secret_excerpt)
    secret_answer = StructuredNormativeAnswer(
        query="RBAC 43.3",
        found=True,
        is_literal_validated=True,
        normative_text=(
            f"[IDENTIFICAÇÃO EXTERNA — RBAC 43]\n"
            f"{LITERAL_START_MARKER}\n{secret_excerpt}\n{LITERAL_END_MARKER}"
        ),
        preliminary_excerpt=None,
        explanation="",
        references=("ANAC RBAC 43",),
        observations=(),
        formatted_text="",
        used_fallback_without_llm=True,
        generation_model=None,
        literal_excerpts=(secret_excerpt,),
    )
    blocked_secret = "".join(
        strip_pagination_header(p) for p in splitter_real.split_normative_answer(secret_answer)
    )
    assert "BLOQUEIO DE FIDELIDADE LITERAL NO ENVIO" in blocked_secret
    assert "AAFakeTelegramTokenSecret" not in blocked_secret
    assert LITERAL_START_MARKER not in blocked_secret
