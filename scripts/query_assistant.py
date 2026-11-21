"""CLI operacional para executar consultas ao motor RAG híbrido sem depender do Telegram.

Exemplos:
    uv run python scripts/query_assistant.py --question "Quem pode executar manutenção no 43.3?"
    uv run python scripts/query_assistant.py --question "RBAC 43.9" --no-llm --debug
    uv run python scripts/query_assistant.py --question "registros" --regulation 43 --only-approved
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rbac_assistant.application.services.chat_service import NormativeChatService
from rbac_assistant.application.services.validation_service import ValidationService
from rbac_assistant.config.settings import load_settings
from rbac_assistant.domain.entities.rag import (
    RetrievalDiagnostics,
    SearchFilter,
    StructuredNormativeAnswer,
)
from rbac_assistant.domain.entities.regulatory_document import ValidationPolicy
from rbac_assistant.domain.ports.rag_ports import EmbeddingPort, LLMPort
from rbac_assistant.infrastructure.llm.gemini_client import (
    GeminiClientAdapter,
    GeminiEmbeddingAdapter,
)
from rbac_assistant.infrastructure.storage.sqlite_document_repository import (
    SqliteDocumentRepository,
)
from rbac_assistant.infrastructure.vector_db.chroma_vector_store import ChromaVectorStore
from rbac_assistant.rag.generation.prompt_templates import PromptTemplateStore
from rbac_assistant.rag.retrieval.hybrid_retriever import HybridNormativeRetriever
from rbac_assistant.security.secret_redactor import redact_text


def _serialize_diagnostics(diag: RetrievalDiagnostics | None) -> dict[str, Any] | None:
    if diag is None:
        return None
    return {
        "query": redact_text(diag.query),
        "effective_filters": dict(diag.effective_filters),
        "resolved_references": [
            {
                "raw_match": r.raw_match,
                "regulation_number": r.regulation_number,
                "section_number": r.section_number,
                "paragraph_number": r.paragraph_number,
                "item_number": r.item_number,
                "subitem_number": r.subitem_number,
                "amendment_number": r.amendment_number,
            }
            for r in diag.resolved_references
        ],
        "structural_hit_ids": list(diag.structural_hit_ids),
        "lexical_hits": [
            {"chunk_id": h.chunk_id, "rank": h.rank, "bm25_score": round(h.bm25_score, 6)}
            for h in diag.lexical_hits
        ],
        "vector_hits": [
            {
                "chunk_id": h.chunk_id,
                "rank": h.rank,
                "distance": round(h.distance, 6),
                "cosine_similarity": round(h.cosine_similarity, 6),
                "accepted": h.accepted,
                "rejection_reason": h.rejection_reason,
            }
            for h in diag.vector_hits
        ],
        "rrf_candidates": [
            {
                "chunk_id": r.chunk_id,
                "rrf_score": round(r.rrf_score, 6),
                "structural_rank": r.structural_rank,
                "lexical_rank": r.lexical_rank,
                "vector_rank": r.vector_rank,
                "sources": [s.value for s in r.sources],
            }
            for r in diag.rrf_candidates
        ],
        "evidence_decisions": [
            {
                "chunk_id": d.chunk_id,
                "document_id": d.document_id,
                "version_id": d.version_id,
                "regulation_number": d.regulation_number,
                "amendment_number": d.amendment_number,
                "section_number": d.section_number,
                "pdf_pages": list(d.pdf_pages),
                "trust_level": d.trust_level,
                "included": d.included,
                "reason": d.reason,
            }
            for d in diag.evidence_decisions
        ],
        "max_vector_distance": diag.max_vector_distance,
        "version_ambiguity_detected": diag.version_ambiguity_detected,
    }


def format_debug_report(answer: StructuredNormativeAnswer) -> str:
    """Renderiza diagnóstico detalhado de recuperação e citações sem expor segredos."""
    diag = answer.diagnostics
    lines: list[str] = ["", "=== DIAGNÓSTICO DO PIPELINE RAG (--debug) ==="]
    lines.append(f"Consulta: {redact_text(answer.query)}")
    if diag is None:
        lines.append("Diagnóstico de recuperação indisponível.")
        return "\n".join(lines)

    filters_str = ", ".join(f"{k}={v}" for k, v in diag.effective_filters) or "(nenhum)"
    lines.append(f"Filtros efetivos: {filters_str}")
    lines.append(
        f"Ambiguidade de versão detectada: {'SIM' if diag.version_ambiguity_detected else 'NÃO'}"
    )

    if diag.resolved_references:
        refs_str = "; ".join(
            f"{r.raw_match} [reg={r.regulation_number}, sec={r.section_number}, "
            f"par={r.paragraph_number}, item={r.item_number}, sub={r.subitem_number}, "
            f"emd={r.amendment_number}]"
            for r in diag.resolved_references
        )
    else:
        refs_str = "(nenhuma referência estrutural explícita)"
    lines.append(f"Referências estruturais resolvidas: {refs_str}")

    lines.append(
        f"Hits estruturais exatos ({len(diag.structural_hit_ids)}): "
        + (", ".join(diag.structural_hit_ids) if diag.structural_hit_ids else "(nenhum)")
    )

    lines.append(f"Hits lexicais FTS5 ({len(diag.lexical_hits)}):")
    for lhit in diag.lexical_hits:
        lines.append(f"  #{lhit.rank} chunk={lhit.chunk_id} bm25={lhit.bm25_score:.4f}")

    lines.append(
        f"Hits vetoriais ChromaDB ({len(diag.vector_hits)}, "
        f"limiar max_vector_distance={diag.max_vector_distance:.2f}):"
    )
    for vhit in diag.vector_hits:
        status = "ACEITO" if vhit.accepted else f"DESCARTADO ({vhit.rejection_reason})"
        lines.append(
            f"  #{vhit.rank} chunk={vhit.chunk_id} dist={vhit.distance:.4f} "
            f"sim={vhit.cosine_similarity:.4f} [{status}]"
        )

    lines.append(f"Fusão RRF ({len(diag.rrf_candidates)}):")
    for rcand in diag.rrf_candidates:
        src_list = ",".join(s.value for s in rcand.sources)
        lines.append(
            f"  chunk={rcand.chunk_id} rrf={rcand.rrf_score:.6f} "
            f"ranks(est={rcand.structural_rank}, fts={rcand.lexical_rank}, "
            f"vec={rcand.vector_rank}) fontes=[{src_list}]"
        )

    lines.append(f"Decisões de hidratação/revisão ({len(diag.evidence_decisions)}):")
    for dec in diag.evidence_decisions:
        action = "INCLUÍDA" if dec.included else "EXCLUÍDA"
        lines.append(
            f"  [{action}] chunk={dec.chunk_id} doc={dec.document_id} ver={dec.version_id} "
            f"reg={dec.regulation_number} emd={dec.amendment_number} sec={dec.section_number} "
            f"pdf_pages={list(dec.pdf_pages)} trust={dec.trust_level} motivo={dec.reason}"
        )

    lines.append(
        "Evidências enviadas ao Gemini: "
        + (", ".join(answer.evidences_sent_to_llm) if answer.evidences_sent_to_llm else "(nenhuma)")
    )
    lines.append(
        "IDs de evidência citados na explicação IA: "
        + (", ".join(answer.used_evidence_ids) if answer.used_evidence_ids else "(nenhum)")
    )
    if answer.discarded_citations:
        lines.append(
            "Citações descartadas pelo CitationValidator: " + ", ".join(answer.discarded_citations)
        )
    return redact_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Executa consulta de teste ao motor RAG híbrido (sem Telegram)."
    )
    parser.add_argument("--question", required=True, help="Pergunta ou dispositivo normativo.")
    parser.add_argument("--document-id", help="Filtra por document_id específico.")
    parser.add_argument("--regulation", help="Filtra por número do regulamento (ex.: 43, 145).")
    parser.add_argument("--amendment", help="Filtra por emenda específica (ex.: 05).")
    parser.add_argument(
        "--only-approved",
        action="store_true",
        help="Considera apenas seções com todas as páginas aprovadas para transcrição literal.",
    )
    parser.add_argument(
        "--include-historical",
        action="store_true",
        help="Inclui versões não vigentes além da emenda atual.",
    )
    parser.add_argument("--top-k", type=int, help="Número máximo de evidências finais.")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Executa somente recuperação híbrida/lexical sem chamar o modelo generativo Gemini.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Exibe diagnóstico completo do RAG (FTS5, ChromaDB, RRF, revisão e citações).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Exibe resposta estruturada em JSON em vez de texto formatado.",
    )
    parser.add_argument("--collection", help="Sobrescreve o nome da coleção no ChromaDB.")
    parser.add_argument("--documents-dir", type=Path, help="Sobrescreve o diretório de dados.")
    args = parser.parse_args(argv)

    settings = load_settings()
    base_dir = args.documents_dir or settings.storage.documents_dir
    repository = SqliteDocumentRepository(
        base_dir / settings.storage.sqlite_filename,
        busy_timeout_ms=settings.storage.sqlite_busy_timeout_ms,
    )
    repository.initialize()

    vector_store = ChromaVectorStore(
        host=settings.chroma_host,
        port=settings.chroma_port,
        heartbeat_path=settings.vector_db.heartbeat_path,
        timeout_seconds=settings.vector_db.request_timeout_seconds,
    )
    validation_service = ValidationService(
        policy=ValidationPolicy(settings.extraction.validation_policy)
    )

    gemini_client: GeminiClientAdapter | None = None
    embedding_port: EmbeddingPort | None = None
    llm_port: LLMPort | None = None

    has_api_key = (
        settings.gemini_api_key is not None
        and bool(settings.gemini_api_key.get_secret_value().strip())
    )
    if has_api_key:
        gemini_client = GeminiClientAdapter(
            api_key=settings.require_gemini_api_key(),
            generation_model=settings.gemini_generation_model,
            embedding_model=settings.gemini_embedding_model,
            embedding_dimensions=settings.effective_embedding_dimensions,
            embedding_instruction_version=settings.gemini.embedding_instruction_version,
            embedding_max_input_tokens=settings.gemini.embedding_max_input_tokens,
            timeout_seconds=settings.gemini.timeout_seconds,
            max_retries=settings.gemini.max_retries,
            retry_min_wait_seconds=settings.gemini.retry_min_wait_seconds,
            retry_max_wait_seconds=settings.gemini.retry_max_wait_seconds,
            default_temperature=settings.gemini.temperature,
            default_max_output_tokens=settings.gemini.max_output_tokens,
            embedding_batch_size=settings.gemini.embedding_batch_size,
        )
        embedding_port = GeminiEmbeddingAdapter(gemini_client)
        if not args.no_llm:
            llm_port = gemini_client
    elif not args.no_llm:
        parser.error(
            "GEMINI_API_KEY não configurada no ambiente. Configure .env ou utilize --no-llm "
            "para testar a recuperação lexical/estrutural sem chamadas externas."
        )

    retriever = HybridNormativeRetriever(
        repository=repository,
        vector_store=vector_store,
        embedding_port=embedding_port,
        collection_name=args.collection or settings.vector_db.collection_name,
        validation_service=validation_service,
        lexical_top_k=settings.retrieval.lexical_top_k,
        semantic_top_k=settings.retrieval.semantic_top_k,
        final_top_k=args.top_k or settings.retrieval.final_top_k,
        rrf_k=settings.retrieval.rrf_k,
        max_vector_distance=settings.retrieval.max_vector_distance,
        reconstruct_full_section_from_ssot=settings.retrieval.reconstruct_full_section_from_ssot,
    )
    chat_service = NormativeChatService(
        retriever=retriever,
        llm=llm_port,
        prompt_store=PromptTemplateStore(settings.gemini.prompts_dir),
    )

    search_filter = SearchFilter(
        document_id=args.document_id,
        regulation_number=args.regulation,
        amendment_number=args.amendment,
        only_current_version=not args.include_historical and not args.amendment,
        only_approved_literal=args.only_approved,
        include_preliminary_native=(
            settings.retrieval.allow_preliminary_native_search and not args.only_approved
        ),
    )

    try:
        answer = chat_service.ask(
            question=args.question,
            filters=search_filter,
            top_k=args.top_k,
            generate_explanation=not args.no_llm,
        )
        if args.json:
            payload: dict[str, Any] = {
                "query": answer.query,
                "found": answer.found,
                "is_literal_validated": answer.is_literal_validated,
                "is_preliminary_only": answer.is_preliminary_only,
                "has_blocked_sources": answer.has_blocked_sources,
                "generation_model": answer.generation_model,
                "used_fallback_without_llm": answer.used_fallback_without_llm,
                "normative_text": answer.normative_text,
                "preliminary_excerpt": answer.preliminary_excerpt,
                "explanation": answer.explanation,
                "references": list(answer.references),
                "observations": list(answer.observations),
                "evidences_sent_to_llm": list(answer.evidences_sent_to_llm),
                "used_evidence_ids": list(answer.used_evidence_ids),
                "discarded_citations": list(answer.discarded_citations),
            }
            if args.debug:
                payload["diagnostics"] = _serialize_diagnostics(answer.diagnostics)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(answer.formatted_text)
            if args.debug:
                print(format_debug_report(answer))
        return 0
    finally:
        if gemini_client is not None:
            gemini_client.close()


if __name__ == "__main__":
    sys.exit(main())
