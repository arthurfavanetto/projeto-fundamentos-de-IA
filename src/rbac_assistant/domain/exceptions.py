"""Exceções puras da camada de domínio regulatório ANAC."""

from __future__ import annotations


class DomainError(Exception):
    """Exceção base para violações de regra de negócio no domínio."""


class SectionNotFoundError(DomainError):
    """Sinaliza que o dispositivo normativo solicitado não existe na base indexada."""


class AmendmentConflictError(DomainError):
    """Sinaliza conflito ou ambiguidade entre emendas de um mesmo regulamento."""


class ExtractionIntegrityError(DomainError):
    """Sinaliza falha na verificação de integridade textual de uma página extraída."""


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
