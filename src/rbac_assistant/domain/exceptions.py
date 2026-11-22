"""Exceções puras da camada de domínio regulatório ANAC."""

from __future__ import annotations


class DomainError(Exception):
    """Exceção base para violações de regra de negócio no domínio."""


class DocumentNotFoundError(DomainError):
    """Sinaliza que o documento ou versão solicitado não está cadastrado no catálogo."""


class DuplicateDocumentError(DomainError):
    """Sinaliza tentativa de cadastrar documento ou versão duplicada."""


class InvalidPdfError(DomainError):
    """Sinaliza arquivo PDF ausente, corrompido, malformado ou fora dos limites de segurança."""


class SourceFetchError(DomainError):
    """Sinaliza falha na obtenção do documento a partir da fonte oficial ou local."""


class OcrUnavailableError(DomainError):
    """Sinaliza que OCR foi necessário para uma página, mas o adaptador OCR não está disponível."""


class SectionNotFoundError(DomainError):
    """Sinaliza que o dispositivo normativo solicitado não existe na base indexada."""


class AmendmentConflictError(DomainError):
    """Sinaliza conflito ou ambiguidade entre emendas de um mesmo regulamento."""


class ExtractionIntegrityError(DomainError):
    """Sinaliza falha na verificação de integridade textual de uma página extraída."""


class ValidationPolicyError(DomainError):
    """Sinaliza tentativa de transição de estado ou aprovação violando a política vigente."""


class UnvalidatedPageBlockedError(DomainError):
    """Sinaliza tentativa de transcrição literal envolvendo página não validada ou bloqueada."""

    def __init__(self, section_number: str, blocked_pdf_pages: tuple[int, ...]) -> None:
        self.section_number = section_number
        self.blocked_pdf_pages = blocked_pdf_pages
        pages_label = ", ".join(str(p) for p in blocked_pdf_pages)
        super().__init__(
            f"Transcrição literal bloqueada para a seção {section_number}: "
            f"páginas PDF não validadas/bloqueadas [{pages_label}]."
        )


class StorageConnectionError(DomainError):
    """Sinaliza indisponibilidade de comunicação com um repositório externo (ex.: ChromaDB)."""


class EmbeddingCompatibilityError(DomainError):
    """Sinaliza incompatibilidade de modelo ou dimensão de embeddings em coleção vetorial."""


class IndexingError(DomainError):
    """Sinaliza erro operacional durante chunking ou indexação vetorial/lexical."""


class LLMServiceError(DomainError):
    """Sinaliza falha na comunicação ou resposta inválida do serviço de IA generativa."""


class LLMRateLimitError(LLMServiceError):
    """Sinaliza atingimento de limite de taxa ou cota (HTTP 429) no serviço de IA."""


class LLMTimeoutError(LLMServiceError):
    """Sinaliza expiração de tempo limite (timeout) ao chamar o serviço de IA."""

