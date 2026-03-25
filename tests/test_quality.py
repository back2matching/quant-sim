"""Tests for quality benchmark grading."""
import pytest
from quant_sim.quality import grade_response, QualityQuestion, QUALITY_TESTS


class TestGrading:
    def test_contains_match(self):
        q = QualityQuestion("facts", "Capital of France?", "contains:Paris")
        assert grade_response(q, "The capital is Paris.") == True
        assert grade_response(q, "paris") == True
        assert grade_response(q, "London") == False

    def test_code_match(self):
        q = QualityQuestion("coding", "Reverse a string", "code:[::-1]")
        assert grade_response(q, "s[::-1]") == True
        assert grade_response(q, "reversed(s)") == False

    def test_thinking_tags_stripped(self):
        q = QualityQuestion("facts", "Capital?", "contains:Paris")
        response = "<think>Let me think... France... capital...</think>Paris"
        assert grade_response(q, response) == True

    def test_case_insensitive(self):
        q = QualityQuestion("facts", "Planet?", "contains:Mercury")
        assert grade_response(q, "MERCURY") == True
        assert grade_response(q, "mercury") == True

    def test_all_questions_have_valid_checks(self):
        for q in QUALITY_TESTS:
            assert ":" in q.check, f"Question '{q.prompt[:30]}' has invalid check format"
            check_type = q.check.split(":")[0]
            assert check_type in ("contains", "exact", "code"), f"Unknown check type: {check_type}"

    def test_question_count(self):
        assert len(QUALITY_TESTS) == 20


class TestGpuDetection:
    def test_import(self):
        from quant_sim.gpu import detect_gpu
        # Should not crash even without GPU
        gpu = detect_gpu()
        # gpu can be None on machines without nvidia-smi


class TestOllamaClient:
    def test_import(self):
        from quant_sim.ollama import check_ollama, list_local_models
        # Should not crash even without Ollama running
