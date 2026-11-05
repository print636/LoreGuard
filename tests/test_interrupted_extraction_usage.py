"""Returned model usage survives interruption before a pipeline result exists."""
import pytest

from app.execution_timing import AnalysisDeadlineExceeded
from app.model_extractor import ModelEnhancedExtractor
from app.pipeline import DocumentInput
from tests.test_review_agent import ScriptedProvider, settings
from tests.test_semantic_repair import fact


@pytest.mark.parametrize("path", ["batch", "chunk", "repair"])
def test_reported_usage_is_retained_before_post_response_deadline(path):
    if path == "batch":
        responses = [{"documents": [{"doc_ref": "d1", "records": []}, {"doc_ref": "d2", "records": []}]}]
    elif path == "repair":
        responses = [{"records": [fact(modality="invalid")]}, {"patches": []}]
    else:
        responses = [{"records": []}]
    mock = ScriptedProvider(responses, settings(enable_review_agent=False))
    extractor = ModelEnhancedExtractor(mock)
    expected = 2 if path == "repair" else 1
    def checkpoint():
        if len(mock.calls) == expected:
            raise AnalysisDeadlineExceeded("run elapsed after reported response")
    extractor.begin_run(checkpoint=checkpoint)
    document = DocumentInput("d1", "first.md", "林澈的身份是领航员。")
    with pytest.raises(AnalysisDeadlineExceeded):
        if path == "batch":
            extractor.extract_batch([document, DocumentInput("d2", "second.md", "林澈走进大厅。")])
        else:
            extractor.extract(document)
    ledger = extractor.interrupted_model_safe_accounting()
    assert len(mock.calls) == ledger["logical_calls"] == expected
    assert (ledger["prompt_tokens"], ledger["completion_tokens"]) == (11 * expected, 7 * expected)
    assert ledger["charged_tokens"] == extractor.conservative_run_token_debit()
    assert ledger["schema_version"] == "model-extraction-accounting-v1"
    assert ledger["provider_calls"] is None
    assert extractor.review_agent_safe_accounting()["logical_calls"] == 0
    ledger["prompt_tokens"] = 999
    assert extractor.interrupted_model_safe_accounting()["prompt_tokens"] == 11 * expected
    extractor.begin_run()
    assert extractor.interrupted_model_safe_accounting()["logical_calls"] == 0


def test_agent_and_main_receipts_are_combined_once_without_changing_agent_ledger():
    mock = ScriptedProvider([
        {"records": [fact(value="舰长")]},
        {"actions": [{"action": "ABSTAIN", "candidate_indexes": [1], "reason_code": "insufficient_evidence"}]},
    ], settings(enable_review_agent=True))
    extractor = ModelEnhancedExtractor(mock)
    def checkpoint():
        if len(mock.calls) >= 2:
            raise AnalysisDeadlineExceeded("run elapsed after Agent response")
    extractor.begin_run(checkpoint=checkpoint)
    with pytest.raises(AnalysisDeadlineExceeded):
        extractor.extract(DocumentInput("d1", "first.md", "林澈的身份是领航员。"))
    agent = extractor.review_agent_safe_accounting()
    ledger = extractor.interrupted_model_safe_accounting()
    assert agent["logical_calls"] == 1 and ledger["logical_calls"] == 2
    assert (agent["prompt_tokens"], agent["completion_tokens"]) == (11, 7)
    assert (ledger["prompt_tokens"], ledger["completion_tokens"]) == (22, 14)
    assert ledger["charged_tokens"] == extractor.conservative_run_token_debit()
    assert set(ledger) == {"schema_version", "logical_calls", "prompt_tokens", "completion_tokens", "charged_tokens", "provider_calls"}
