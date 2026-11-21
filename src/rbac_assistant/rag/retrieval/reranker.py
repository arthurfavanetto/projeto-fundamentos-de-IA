"""Reranker determinístico leve para evidências normativas (sem LLM externo ou serviço pago).

Opera exclusivamente após a recuperação híbrida e os filtros de metadados/versão,
combinando:
- score RRF normalizado;
- similaridade vetorial cosseno (`1 - distance`);
- força lexical BM25 normalizada;
- coincidência estrutural exata de seção;
- sobreposição lexical de termos informativos com o título da seção.

Invariante: o reranking NUNCA sobrepõe filtros normativos (regulamento, emenda, vigência)
nem estados de revisão por página (`review_on_demand`).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from rbac_assistant.domain.entities.rag import NormativeChunk, ResolvedStructuralRef

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_LIGHT_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "o",
        "e",
        "de",
        "do",
        "da",
        "dos",
        "das",
        "em",
        "no",
        "na",
        "nos",
        "nas",
        "para",
        "por",
        "com",
        "um",
        "uma",
        "que",
        "qual",
        "quais",
        "quem",
        "como",
        "onde",
        "quando",
        "sobre",
        "se",
        "ao",
        "os",
        "as",
        "ou",
        "rbac",
        "anac",
        "secao",
        "emenda",
    }
)


def _normalize_tokens(text: str) -> set[str]:
    nfkd = unicodedata.normalize("NFKD", text.lower())
    ascii_text = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
    return {
        tok
        for tok in _TOKEN_RE.findall(ascii_text)
        if len(tok) > 1 and tok not in _LIGHT_STOPWORDS
    }


@dataclass(frozen=True, slots=True)
class RerankInput:
    """Entrada determinística de um candidato pré-filtrado para cálculo de score de reranking."""

    chunk_id: str
    chunk: NormativeChunk
    rrf_score: float
    structural_rank: int | None
    lexical_rank: int | None
    vector_rank: int | None
    bm25_score: float | None = None
    vector_distance: float | None = None


class DeterministicNormativeReranker:
    """Combina sinais estruturais, BM25, similaridade cosseno, RRF e título sem modelo externo."""

    def score_candidates(
        self,
        *,
        query: str,
        resolved_refs: Sequence[ResolvedStructuralRef],
        candidates: Sequence[RerankInput],
    ) -> dict[str, float]:
        """Retorna mapa `chunk_id -> rerank_score` determinístico para ordenação final."""
        if not candidates:
            return {}

        max_rrf = max((c.rrf_score for c in candidates), default=0.0) or 1.0
        max_bm25 = max(((c.bm25_score or 0.0) for c in candidates), default=0.0)
        query_tokens = _normalize_tokens(query)
        exact_sections = {
            ref.section_number.strip().lower()
            for ref in resolved_refs
            if ref.section_number
        }

        scores: dict[str, float] = {}
        for cand in candidates:
            rrf_norm = cand.rrf_score / max_rrf

            if cand.vector_distance is not None:
                semantic_signal = max(0.0, min(1.0, 1.0 - cand.vector_distance))
            else:
                semantic_signal = 0.0

            if cand.bm25_score is not None and max_bm25 > 0.0:
                lexical_signal = max(0.0, cand.bm25_score / max_bm25)
            else:
                lexical_signal = 0.0

            title_tokens = _normalize_tokens(cand.chunk.title)
            if query_tokens and title_tokens:
                title_overlap = len(query_tokens & title_tokens) / float(len(title_tokens))
            else:
                title_overlap = 0.0

            is_exact_structural = (
                cand.structural_rank is not None
                or cand.chunk.section_number.strip().lower() in exact_sections
            )
            structural_bonus = 0.35 if is_exact_structural else 0.0

            combined = (
                0.40 * rrf_norm
                + 0.25 * semantic_signal
                + 0.20 * lexical_signal
                + 0.10 * title_overlap
                + structural_bonus
            )
            scores[cand.chunk_id] = combined

        return scores
