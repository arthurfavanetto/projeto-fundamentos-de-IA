"""Camada determinística de entendimento e orientação conversacional para o Telegram.

Responsabilidades:
- Classificar consultas livres antes da recuperação normativa (zero embeddings e zero Gemini
  em perguntas claramente vagas, saudações ou pedidos genéricos de orientação).
- Preservar integralmente a prioridade absoluta de referências estruturais explícitas
  (ex.: `43.10`, `43.10(a)(1)`, `43.10(a)(1) e 43.10(a)(99)`), que nunca são interceptadas
  por orientação e seguem diretamente para o pipeline fail-closed do PR #21.
- Manter estado conversacional mínimo, isolado estritamente por `user_id` e com expiração
  automática (TTL), para continuidade imediata após esclarecimento ou referência contextual
  (`essa seção`, `essa exigência`), sem retenção indefinida e sem vazamento entre usuários.
"""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from rbac_assistant.rag.retrieval.section_resolver import (
    extract_amendment_hint,
    extract_regulation_hint,
    resolve_structural_references,
)

DEFAULT_CONVERSATION_TTL_SECONDS = 600.0
DEFAULT_MAX_TRACKED_USERS = 2048


class ConversationIntent(StrEnum):
    """Intenção determinística detectada antes da recuperação normativa."""

    LITERAL_QUERY = "literal_query"
    EXPLANATORY_QUERY = "explanatory_query"
    NEEDS_CLARIFICATION = "needs_clarification"


@dataclass(frozen=True, slots=True)
class ConversationalDecision:
    """Decisão determinística de roteamento conversacional para texto livre."""

    intent: ConversationIntent
    effective_query: str
    generate_explanation: bool
    reason: str
    guidance_message: str | None = None
    expects_clarification_followup: bool = False
    used_clarification_context: bool = False
    used_deictic_context: bool = False


@dataclass(frozen=True, slots=True)
class PendingClarification:
    """Contexto efêmero de um pedido de esclarecimento aguardando resposta do mesmo usuário."""

    original_question: str
    regulation_hint: str | None
    amendment_hint: str | None
    created_at: float
    turns: int = 1


@dataclass(slots=True)
class _UserConversationRecord:
    pending_clarification: PendingClarification | None = None
    last_normative_refs: tuple[str, ...] = ()
    last_refs_updated_at: float = 0.0


# Mapeamento determinístico das opções numéricas apresentadas na mensagem de orientação.
_GUIDANCE_OPTION_QUERIES: dict[str, str] = {
    "1": (
        "Quais são os requisitos aplicáveis sobre execução de manutenção, "
        "manutenção preventiva, reconstrução ou alteração?"
    ),
    "2": "Quais são os requisitos aplicáveis sobre inspeções de manutenção?",
    "3": "Quais são os requisitos aplicáveis sobre registros e documentação de manutenção?",
    "4": "Quais são os requisitos aplicáveis sobre aprovação para retorno ao serviço?",
    "5": "Quais são os requisitos aplicáveis sobre controle de componentes com limite de vida?",
}

_NUMBERED_CHOICE_RE = re.compile(
    r"^\s*(?:op[çc][ãa]o\s+|item\s+|n[úu]mero\s+|#)?(?P<num>[1-6])(?:[\s).:\-]+(?P<rest>.*))?$",
    re.IGNORECASE,
)

# Política determinística de intenção para texto livre:
# 1. Referência estrutural explícita -> NUNCA cai em orientação (preserva PR #21).
# 2. Pedido explícito de explicação/interpretação -> explicação separada via Gemini.
# 3. Pedido literal/transcrição -> somente texto literal e referência (zero Gemini).
# 4. Pergunta vaga / saudação / pedido genérico de orientação -> pede esclarecimento (zero RAG/LLM).
# 5. Pergunta analítica concreta -> explicação fundamentada via Gemini.
# 6. Termos técnicos soltos ou consulta ambígua concreta -> evidência literal (zero Gemini).
_EXPLICIT_EXPLANATION_RE = re.compile(
    r"\b(?:expliqu\w*|explicar|explica[çc][ãa]o|resum\w*|interpret\w*|"
    r"esclare[çc]\w*|esclarec\w*|o\s+que\s+significa|significa\s+o\s+qu[êe]|"
    r"como\s+funciona|qual\s+(?:[ée]\s+)?a\s+diferen[çc]a|diferen[çc]a\s+entre|"
    r"quando\s+se\s+aplica)\b",
    re.IGNORECASE,
)
_LITERAL_ONLY_REQUEST_RE = re.compile(
    r"\b(?:transcrev\w*|transcri[çc][ãa]o|texto\s+literal|texto\s+exato|"
    r"texto\s+integral|conte[úu]do\s+(?:literal|exato|integral|oficial)|"
    r"ipsis\s+litteris|literalmente|reproduz\w*|copi[ae]\w*|"
    r"(?:mostr[ae]|exib[ae]|traga|envi[ae]|qual\s+(?:[ée]\s+)?o)\s+o?\s*"
    r"(?:texto|conte[úu]do)\b|"
    r"ler?\s+a\s+se[çc][ãa]o|cit[ae]\s+literal\w*)\b",
    re.IGNORECASE,
)
_ANALYTICAL_QUESTION_RE = re.compile(
    r"(?:^\s*(?:quem|como|quando|por\s+que|por\s+qu[êe]|quais|qual|onde|"
    r"pode|podem|deve|devem|precisa|precisam|[ée]\s+obrigat[óo]rio|"
    r"[ée]\s+permitido|[ée]\s+poss[íi]vel)\b.*\?\s*$|"
    r"\b(?:quem\s+pode|como\s+deve|quando\s+deve|por\s+que\s+|"
    r"quais\s+s[ãa]o\s+os\s+requisitos)\b)",
    re.IGNORECASE,
)

_GREETING_ONLY_RE = re.compile(
    r"^\s*(?:oi|ol[áa]|opa|e\s+a[íi]|fala|hello|hi|"
    r"bom\s+dia|boa\s+tarde|boa\s+noite|sauda[çc][õo]es)"
    r"(?:[\s,!?.]+(?:tudo\s+bem|como\s+vai|pode\s+me\s+ajudar|"
    r"preciso\s+de\s+ajuda|queria\s+uma\s+ajuda|gostaria\s+de\s+ajuda|"
    r"me\s+ajuda|me\s+ajude|como\s+funciona))*"
    r"[\s!?.]*$",
    re.IGNORECASE,
)
_HELP_OR_META_ORIENTATION_RE = re.compile(
    r"^\s*(?:ajuda|socorro|menu|op[çc][õo]es|como\s+usar|como\s+funciona(?:\s+o\s+bot)?|"
    r"pode\s+me\s+ajudar|preciso\s+de\s+ajuda|me\s+ajuda|me\s+ajude|me\s+oriente|"
    r"n[ãa]o\s+sei\s+(?:o\s+que\s+perguntar|qual\s+se[çc][ãa]o|por\s+onde\s+come[çc]ar)|"
    r"por\s+onde\s+(?:come[çc]o|come[çc]ar)|"
    r"o\s+que\s+voc[êe]\s+(?:faz|responde|sabe|cobre|pode\s+fazer)|"
    r"quais\s+assuntos\s+(?:voc[êe]\s+)?(?:cobre|responde|posso\s+perguntar)|"
    r"como\s+posso\s+perguntar)[\s!?.]*$",
    re.IGNORECASE,
)
_VAGUE_SELF_CASE_RE = re.compile(
    r"\b(?:"
    r"qual\s+se[çc][ãa]o\s+(?:se\s+usa|devo\s+(?:consultar|usar|ler)|"
    r"se\s+aplica|[ée]\s+(?:para|a)\s+(?:o\s+meu|meu|mim))|"
    r"quais\s+se[çc][õo]es\s+(?:devo|preciso|se\s+aplicam|usar|consultar|ler)|"
    r"qual\s+(?:norma|regra|regulamento|dispositivo|item|parte)\s+"
    r"(?:se\s+usa|se\s+aplica|devo\s+(?:consultar|usar))\s+"
    r"(?:para\s+)?(?:o\s+meu\s+caso|meu\s+caso|mim|meu\s+trabalho|minha\s+atividade)|"
    r"para\s+o\s+meu\s+caso|"
    r"preciso\s+saber\s+(?:das\s+coisas|das\s+regras|das\s+normas|"
    r"o\s+que\s+preciso|por\s+onde\s+come[çc]ar)|"
    r"o\s+que\s+(?:eu\s+)?preciso\s+saber\b|"
    r"(?:trabalho|atuo)\s+(?:com|na\s+[áa]rea\s+de)\s+"
    r"(?:aeronavegabilidade|avia[çc][ãa]o|manuten[çc][ãa]o|oficina)"
    r")\b",
    re.IGNORECASE,
)
_BROAD_OVERVIEW_RE = re.compile(
    r"^\s*(?:me\s+fal[ae]|fal[ae]|quero\s+(?:saber|entender|conhecer)|"
    r"preciso\s+(?:saber|entender)|me\s+ajud[ae]\s+com|"
    r"o\s+que\s+(?:diz|trata|tem|fala|[ée]))\s+"
    r"(?:sobre\s+|do\s+|da\s+|no\s+|na\s+|o\s+|a\s+)*"
    r"(?:rbac(?:\s*(?:n[º°.]?\s*)?\d{1,3})?|anac|aeronavegabilidade|"
    r"manuten[çc][ãa]o|avia[çc][ãa]o)"
    r"(?:\s+(?:no\s+geral|em\s+geral))?[\s!?.]*$",
    re.IGNORECASE,
)
_BARE_GENERIC_TOPIC_RE = re.compile(
    r"^\s*(?:aeronavegabilidade|avia[çc][ãa]o|rbac(?:\s*(?:n[º°.]?\s*)?\d{1,3})?|"
    r"anac|norma[s]?|regra[s]?|regulamento[s]?|se[çc][ãa]o|se[çc][õo]es|"
    r"d[úu]vida[s]?|informa[çc][ãa]o|informa[çc][õo]es|meu\s+caso|"
    r"outra\s+atividade|outra|outro|n[ãa]o\s+sei|qualquer\s+coisa|tudo|\d)[\s!?.]*$",
    re.IGNORECASE,
)
_CONCRETE_SUBTOPIC_RE = re.compile(
    r"\b(?:"
    r"manuten[çc][ãa]o\s+preventiva|altera[çc][ãa]o|altera[çc][õo]es|reconstru[çc][ãa]o|"
    r"inspe[çc][ãa]o|inspe[çc][õo]es|registro[s]?\s+de\s+manuten[çc][ãa]o|"
    r"documenta[çc][ãa]o\s+de\s+manuten[çc][ãa]o|caderneta|"
    r"retorno\s+ao\s+servi[çc]o|aprova[çc][ãa]o\s+para\s+retorno|"
    r"limite\s+de\s+vida|pe[çc]a[s]?\s+com\s+limite|componente[s]?\s+com\s+limite|"
    r"segrega[çc][ãa]o|mutila[çc][ãa]o|alt[íi]metro|transponder|pitot|est[áa]tico|"
    r"formul[áa]rio|ficha|etiqueta|grande\s+reparo|grande\s+altera[çc][ãa]o|"
    r"reparo\s+maior|altera[çc][ãa]o\s+maior|aeronavegabilidade\s+continuada|"
    r"diretriz\s+de\s+aeronavegabilidade|piloto|comiss[áa]rio|licen[çc]a|"
    r"habilita[çc][ãa]o|certificado\s+m[ée]dico|peso\s+e\s+balanceamento|"
    r"instrumento[s]?|motor|h[ée]lice|c[ée]lula|aeronave\s+experimental"
    r")\b",
    re.IGNORECASE,
)
_DEICTIC_FOLLOWUP_RE = re.compile(
    r"\b(?:ess[ae]s?\s+(?:se[çc][ãa]o|se[çc][õo]es|exig[êe]ncia[s]?|requisito[s]?|"
    r"regra[s]?|item|itens|par[áa]grafo[s]?|inciso[s]?|al[íi]nea[s]?|"
    r"dispositivo[s]?|trecho[s]?|norma[s]?|dois\s+requisitos|dois\s+itens)|"
    r"est[ae]s?\s+(?:se[çc][ãa]o|exig[êe]ncia|requisito|regra|item|par[áa]grafo|"
    r"inciso|al[íi]nea|dispositivo)|"
    r"o\s+que\s+significa\s+isso|expliqu[ae]\s+isso|explicar\s+isso|"
    r"entre\s+eles|entre\s+elas|entre\s+ambos)\b",
    re.IGNORECASE,
)
_PLURAL_COMPARISON_RE = re.compile(
    r"\b(?:dois\s+(?:requisitos|itens)|diferen[çc]a\s+entre\s+(?:esses|estes|eles|elas|ambos)|"
    r"entre\s+(?:eles|elas|ambos))\b",
    re.IGNORECASE,
)


def build_orientation_guidance_message() -> str:
    """Mensagem curta de orientação para perguntas vagas (sem assumir cobertura total)."""
    return (
        "Posso ajudar você a localizar e compreender dispositivos nos regulamentos oficiais "
        "da ANAC aprovados e disponíveis na base.\n\n"
        "Para direcionar a consulta antes de buscar, qual atividade ou assunto está mais próximo "
        "da sua necessidade?\n\n"
        "1. Execução de manutenção ou alteração\n"
        "2. Inspeções\n"
        "3. Registros e documentação de manutenção\n"
        "4. Aprovação para retorno ao serviço\n"
        "5. Controle de componentes com limite de vida\n"
        "6. Requisitos de aeronavegabilidade continuada ou outra atividade\n\n"
        "Obs.: estes tópicos são apenas exemplos de assunto para orientar sua pergunta — "
        "o RBAC 43 não regula integralmente todos os temas de aviação, e só respondo com base "
        "nos documentos oficiais aprovados na base.\n\n"
        "Responda com o número (1 a 6), descreva brevemente o que precisa ou informe diretamente "
        "a seção desejada (ex.: 43.10(a)(1))."
    )


def build_other_activity_followup_message() -> str:
    """Solicitação breve quando o usuário escolhe 'outra atividade' sem descrever o tema."""
    return (
        "Entendido. Descreva em uma frase qual atividade, dúvida ou requisito técnico você "
        "procura (por exemplo: manutenção preventiva, inspeção de altímetro/transponder, "
        "quem pode assinar um registro) ou informe diretamente a seção normativa desejada."
    )


def build_missing_deictic_target_message(*, plural: bool = False) -> str:
    """Orientação quando o usuário usa 'essa seção'/'esses dois requisitos' sem contexto ativo."""
    if plural:
        return (
            "Não identifiquei duas referências anteriores nesta conversa para comparar.\n\n"
            "Informe quais seções, itens ou requisitos deseja comparar "
            "(ex.: \"Qual a diferença entre 43.10(a)(1) e 43.10(a)(2)?\" ou "
            "\"Qual a diferença entre manutenção preventiva e alteração?\")."
        )
    return (
        "Não identifiquei qual seção ou exigência você está referenciando nesta conversa.\n\n"
        "Informe o número do dispositivo (ex.: \"Explique a seção 43.10\" ou "
        "\"O que significa 43.10(a)(1)?\") ou descreva o assunto que deseja consultar."
    )


class UserConversationStore:
    """Estado conversacional efêmero, thread-safe e estritamente isolado por `user_id`."""

    def __init__(
        self,
        *,
        ttl_seconds: float = DEFAULT_CONVERSATION_TTL_SECONDS,
        max_users: int = DEFAULT_MAX_TRACKED_USERS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl_seconds = max(1.0, float(ttl_seconds))
        self._max_users = max(1, int(max_users))
        self._clock = clock
        self._lock = threading.Lock()
        self._records: OrderedDict[int, _UserConversationRecord] = OrderedDict()

    def _evict_expired_locked(self, now: float) -> None:
        expired_users: list[int] = []
        for uid, rec in self._records.items():
            if (
                rec.pending_clarification is not None
                and now - rec.pending_clarification.created_at > self._ttl_seconds
            ):
                rec.pending_clarification = None
            if rec.last_normative_refs and now - rec.last_refs_updated_at > self._ttl_seconds:
                rec.last_normative_refs = ()
                rec.last_refs_updated_at = 0.0
            if rec.pending_clarification is None and not rec.last_normative_refs:
                expired_users.append(uid)
        for uid in expired_users:
            del self._records[uid]

    def _get_or_create_locked(self, user_id: int, now: float) -> _UserConversationRecord:
        self._evict_expired_locked(now)
        rec = self._records.get(user_id)
        if rec is not None:
            self._records.move_to_end(user_id)
            return rec
        while len(self._records) >= self._max_users:
            self._records.popitem(last=False)
        rec = _UserConversationRecord()
        self._records[user_id] = rec
        return rec

    def get_pending_clarification(self, user_id: int) -> PendingClarification | None:
        """Retorna o pedido de esclarecimento ativo do usuário, se ainda dentro do TTL."""
        with self._lock:
            now = self._clock()
            self._evict_expired_locked(now)
            rec = self._records.get(user_id)
            if rec is None:
                return None
            return rec.pending_clarification

    def set_pending_clarification(
        self,
        user_id: int,
        *,
        original_question: str,
        regulation_hint: str | None = None,
        amendment_hint: str | None = None,
    ) -> None:
        """Registra que este usuário aguarda esclarecimento para sua consulta vaga."""
        with self._lock:
            now = self._clock()
            rec = self._get_or_create_locked(user_id, now)
            prev = rec.pending_clarification
            turns = prev.turns + 1 if prev is not None else 1
            rec.pending_clarification = PendingClarification(
                original_question=original_question,
                regulation_hint=regulation_hint or (prev.regulation_hint if prev else None),
                amendment_hint=amendment_hint or (prev.amendment_hint if prev else None),
                created_at=now,
                turns=turns,
            )

    def clear_pending_clarification(self, user_id: int) -> None:
        """Limpa apenas o pedido de esclarecimento pendente do usuário."""
        with self._lock:
            now = self._clock()
            rec = self._records.get(user_id)
            if rec is None:
                return
            rec.pending_clarification = None
            self._evict_expired_locked(now)

    def get_last_normative_refs(self, user_id: int) -> tuple[str, ...]:
        """Retorna as referências validadas mais recentes deste usuário (dentro do TTL)."""
        with self._lock:
            now = self._clock()
            self._evict_expired_locked(now)
            rec = self._records.get(user_id)
            if rec is None:
                return ()
            return rec.last_normative_refs

    def set_last_normative_refs(self, user_id: int, refs: tuple[str, ...]) -> None:
        """Armazena somente os identificadores normativos canônicos da última resposta validada."""
        clean_refs = tuple(dict.fromkeys(r.strip() for r in refs if r.strip()))[:4]
        with self._lock:
            now = self._clock()
            rec = self._get_or_create_locked(user_id, now)
            rec.last_normative_refs = clean_refs
            rec.last_refs_updated_at = now if clean_refs else 0.0
            self._evict_expired_locked(now)

    def reset_user(self, user_id: int) -> None:
        """Remove completamente o contexto conversacional efêmero do usuário."""
        with self._lock:
            self._records.pop(user_id, None)


def _append_reg_context(
    query: str,
    *,
    regulation_hint: str | None,
    amendment_hint: str | None,
) -> str:
    """Anexa apenas dica explícita de regulamento/emenda da pergunta inicial se ausente."""
    _, existing_reg = extract_regulation_hint(query)
    existing_amd = extract_amendment_hint(query)
    suffix_parts: list[str] = []
    if regulation_hint and not existing_reg:
        suffix_parts.append(f"RBAC {regulation_hint}")
    if amendment_hint and not existing_amd:
        suffix_parts.append(f"Emenda {amendment_hint}")
    if not suffix_parts:
        return query
    return f"{query} ({', '.join(suffix_parts)})"


def _wants_explanation_for_concrete_query(cleaned: str) -> bool:
    """Regra determinística de literal vs. explicação para consultas não-vagas."""
    if _EXPLICIT_EXPLANATION_RE.search(cleaned):
        return True
    if _LITERAL_ONLY_REQUEST_RE.search(cleaned):
        return False
    return bool(_ANALYTICAL_QUESTION_RE.search(cleaned))


def _is_vague_without_structural_ref(cleaned: str) -> str | None:
    """Retorna motivo objetivo se a consulta (sem ref estrutural) for vaga/genérica demais."""
    if _GREETING_ONLY_RE.match(cleaned):
        return "greeting"
    if _HELP_OR_META_ORIENTATION_RE.match(cleaned):
        return "help_orientation"
    if _BARE_GENERIC_TOPIC_RE.match(cleaned):
        return "bare_generic_topic"
    if _CONCRETE_SUBTOPIC_RE.search(cleaned):
        return None
    if _VAGUE_SELF_CASE_RE.search(cleaned):
        return "vague_self_case"
    if _BROAD_OVERVIEW_RE.match(cleaned):
        return "broad_overview"
    return None


def classify_conversational_intent(
    text: str,
    *,
    pending_clarification: PendingClarification | None = None,
    last_normative_refs: tuple[str, ...] = (),
) -> ConversationalDecision:
    """Decide deterministicamente entre consulta literal, explicação ou orientação.

    Invariante de fidelidade (PR #21):
    - Qualquer texto contendo referência estrutural explícita (`resolve_structural_references`)
      jamais é desviado para orientação: segue intacto para o pipeline de seleção estrutural
      e verificação literal fail-closed no SSOT.
    """
    cleaned = text.strip()
    if not cleaned:
        return ConversationalDecision(
            intent=ConversationIntent.NEEDS_CLARIFICATION,
            effective_query="",
            generate_explanation=False,
            reason="empty_input",
            guidance_message=build_orientation_guidance_message(),
            expects_clarification_followup=True,
        )

    # 1. Santuário estrutural (PR #21): referência estrutural explícita sempre vai intacta ao RAG.
    explicit_refs = resolve_structural_references(cleaned)
    if explicit_refs:
        wants_exp = _wants_explanation_for_concrete_query(cleaned)
        return ConversationalDecision(
            intent=(
                ConversationIntent.EXPLANATORY_QUERY
                if wants_exp
                else ConversationIntent.LITERAL_QUERY
            ),
            effective_query=cleaned,
            generate_explanation=wants_exp,
            reason="explicit_structural_ref",
        )

    # 2. Continuidade imediata quando este usuário está respondendo a um pedido de esclarecimento.
    if pending_clarification is not None:
        choice_match = _NUMBERED_CHOICE_RE.match(cleaned)
        if choice_match is not None:
            num = choice_match.group("num")
            rest = (choice_match.group("rest") or "").strip()
            if num in _GUIDANCE_OPTION_QUERIES:
                base_query = (
                    rest
                    if len(rest) >= 12 and _CONCRETE_SUBTOPIC_RE.search(rest)
                    else _GUIDANCE_OPTION_QUERIES[num]
                )
                effective = _append_reg_context(
                    base_query,
                    regulation_hint=pending_clarification.regulation_hint,
                    amendment_hint=pending_clarification.amendment_hint,
                )
                return ConversationalDecision(
                    intent=ConversationIntent.EXPLANATORY_QUERY,
                    effective_query=effective,
                    generate_explanation=True,
                    reason=f"clarification_option_{num}",
                    used_clarification_context=True,
                )
            # Opção 6 ("outra atividade"): se o usuário já descreveu o tema após "6", usa-o!
            if rest and not _BARE_GENERIC_TOPIC_RE.match(rest) and len(rest.split()) >= 2:
                effective = _append_reg_context(
                    rest,
                    regulation_hint=pending_clarification.regulation_hint,
                    amendment_hint=pending_clarification.amendment_hint,
                )
                wants_exp = not bool(_LITERAL_ONLY_REQUEST_RE.search(rest))
                return ConversationalDecision(
                    intent=(
                        ConversationIntent.EXPLANATORY_QUERY
                        if wants_exp
                        else ConversationIntent.LITERAL_QUERY
                    ),
                    effective_query=effective,
                    generate_explanation=wants_exp,
                    reason="clarification_option_6_with_detail",
                    used_clarification_context=True,
                )
            return ConversationalDecision(
                intent=ConversationIntent.NEEDS_CLARIFICATION,
                effective_query="",
                generate_explanation=False,
                reason="clarification_option_6_needs_detail",
                guidance_message=build_other_activity_followup_message(),
                expects_clarification_followup=True,
                used_clarification_context=True,
            )

        # Se a resposta ao esclarecimento ainda for vaga/genérica, solicita detalhe curto.
        still_vague = _is_vague_without_structural_ref(cleaned)
        if still_vague is not None:
            return ConversationalDecision(
                intent=ConversationIntent.NEEDS_CLARIFICATION,
                effective_query="",
                generate_explanation=False,
                reason=f"clarification_still_vague_{still_vague}",
                guidance_message=build_other_activity_followup_message(),
                expects_clarification_followup=True,
                used_clarification_context=True,
            )

        # Referência dêitica sem seção prévia ("essa seção") durante esclarecimento pede alvo.
        if _DEICTIC_FOLLOWUP_RE.search(cleaned) and not _CONCRETE_SUBTOPIC_RE.search(cleaned):
            is_plural = bool(_PLURAL_COMPARISON_RE.search(cleaned))
            min_refs = 2 if is_plural else 1
            if len(last_normative_refs) >= min_refs:
                selected_refs = (
                    last_normative_refs[:2] if is_plural else last_normative_refs[:1]
                )
                refs_context = ", ".join(selected_refs)
                wants_exp = _wants_explanation_for_concrete_query(cleaned)
                return ConversationalDecision(
                    intent=(
                        ConversationIntent.EXPLANATORY_QUERY
                        if wants_exp
                        else ConversationIntent.LITERAL_QUERY
                    ),
                    effective_query=f"{cleaned} (referente a: {refs_context})",
                    generate_explanation=wants_exp,
                    reason="deictic_followup_resolved",
                    used_deictic_context=True,
                )
            return ConversationalDecision(
                intent=ConversationIntent.NEEDS_CLARIFICATION,
                effective_query="",
                generate_explanation=False,
                reason="unresolved_deictic_reference",
                guidance_message=build_missing_deictic_target_message(plural=is_plural),
                expects_clarification_followup=True,
                used_clarification_context=True,
            )

        # Resposta concreta ao esclarecimento (ex.: "inspeções", "componentes com limite de vida",
        # "preciso registrar manutenção" ou tema fora da base que seguirá para recusa fail-closed)
        is_literal_req = bool(_LITERAL_ONLY_REQUEST_RE.search(cleaned))
        if is_literal_req:
            effective = _append_reg_context(
                cleaned,
                regulation_hint=pending_clarification.regulation_hint,
                amendment_hint=pending_clarification.amendment_hint,
            )
            return ConversationalDecision(
                intent=ConversationIntent.LITERAL_QUERY,
                effective_query=effective,
                generate_explanation=False,
                reason="clarification_followup_literal",
                used_clarification_context=True,
            )

        clarified_query = (
            cleaned
            if "?" in cleaned or len(cleaned.split()) >= 5
            else f"Quais são os requisitos aplicáveis sobre {cleaned}?"
        )
        effective = _append_reg_context(
            clarified_query,
            regulation_hint=pending_clarification.regulation_hint,
            amendment_hint=pending_clarification.amendment_hint,
        )
        return ConversationalDecision(
            intent=ConversationIntent.EXPLANATORY_QUERY,
            effective_query=effective,
            generate_explanation=True,
            reason="clarification_followup_resolved",
            used_clarification_context=True,
        )

    # 3. Referências dêiticas sem seção explícita ("essa seção", "essa exigência", "esses dois").
    if _DEICTIC_FOLLOWUP_RE.search(cleaned) and not _CONCRETE_SUBTOPIC_RE.search(cleaned):
        is_plural = bool(_PLURAL_COMPARISON_RE.search(cleaned))
        min_refs = 2 if is_plural else 1
        if len(last_normative_refs) >= min_refs:
            selected_refs = last_normative_refs[:2] if is_plural else last_normative_refs[:1]
            refs_context = ", ".join(selected_refs)
            wants_exp = _wants_explanation_for_concrete_query(cleaned)
            return ConversationalDecision(
                intent=(
                    ConversationIntent.EXPLANATORY_QUERY
                    if wants_exp
                    else ConversationIntent.LITERAL_QUERY
                ),
                effective_query=f"{cleaned} (referente a: {refs_context})",
                generate_explanation=wants_exp,
                reason="deictic_followup_resolved",
                used_deictic_context=True,
            )
        return ConversationalDecision(
            intent=ConversationIntent.NEEDS_CLARIFICATION,
            effective_query="",
            generate_explanation=False,
            reason="unresolved_deictic_reference",
            guidance_message=build_missing_deictic_target_message(plural=is_plural),
            expects_clarification_followup=True,
        )

    # 4. Pergunta vaga / saudação / pedido genérico de orientação (zero embeddings, zero Gemini).
    vague_reason = _is_vague_without_structural_ref(cleaned)
    if vague_reason is not None:
        return ConversationalDecision(
            intent=ConversationIntent.NEEDS_CLARIFICATION,
            effective_query="",
            generate_explanation=False,
            reason=vague_reason,
            guidance_message=build_orientation_guidance_message(),
            expects_clarification_followup=True,
        )

    # 5. Consulta concreta sem referência estrutural: decide entre literal ou explicação.
    wants_exp = _wants_explanation_for_concrete_query(cleaned)
    return ConversationalDecision(
        intent=(
            ConversationIntent.EXPLANATORY_QUERY
            if wants_exp
            else ConversationIntent.LITERAL_QUERY
        ),
        effective_query=cleaned,
        generate_explanation=wants_exp,
        reason="concrete_explanatory" if wants_exp else "concrete_literal",
    )


def should_generate_explanation_for_free_text(text: str) -> bool:
    """Decide deterministicamente se uma consulta livre exige explicação IA separada.

    Retorna `False` (zero chamada ao Gemini) para pedidos literais, citações diretas,
    perguntas vagas que requerem orientação prévia e termos soltos sem intenção explicativa.
    """
    decision = classify_conversational_intent(text)
    return decision.intent == ConversationIntent.EXPLANATORY_QUERY
