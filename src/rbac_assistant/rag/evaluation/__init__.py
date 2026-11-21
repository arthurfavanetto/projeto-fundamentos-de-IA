"""Framework determinístico de avaliação quantitativa do motor RAG híbrido."""

from rbac_assistant.rag.evaluation.rag_evaluator import (
    AblationExperimentResult,
    CategoryMetrics,
    CitationValidatorAuditReport,
    ErrorBreakdown,
    EvaluationCase,
    EvaluationModeMetrics,
    EvaluationSummaryReport,
    ExpectedTarget,
    QueryEvalDetail,
    RagEvaluator,
    StructuralSubsetMetrics,
    ThresholdSweepPoint,
    evaluate_citation_validator_suite,
)

__all__ = [
    "AblationExperimentResult",
    "CategoryMetrics",
    "CitationValidatorAuditReport",
    "ErrorBreakdown",
    "EvaluationCase",
    "EvaluationModeMetrics",
    "EvaluationSummaryReport",
    "ExpectedTarget",
    "QueryEvalDetail",
    "RagEvaluator",
    "StructuralSubsetMetrics",
    "ThresholdSweepPoint",
    "evaluate_citation_validator_suite",
]
