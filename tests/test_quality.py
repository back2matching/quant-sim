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


class TestBenchModule:
    def test_extract_quant(self):
        from quant_sim.ollama import _extract_quant_from_name
        assert _extract_quant_from_name("qwen2.5:7b-instruct-q4_k_m") == "Q4_K_M"
        assert _extract_quant_from_name("qwen2.5:7b-instruct-q3_k_s") == "Q3_K_S"
        assert _extract_quant_from_name("qwen2.5:7b-instruct-q8_0") == "Q8_0"
        assert _extract_quant_from_name("qwen2.5:7b") == "default"
        assert _extract_quant_from_name("model:fp16") == "FP16"

    def test_recommend_picks_fastest_above_threshold(self):
        from quant_sim.bench import recommend, QuantResult
        results = [
            QuantResult("a", "Q4", 4.0, 8000, 100, 50.0, 85, {}, True),
            QuantResult("b", "Q5", 5.0, 9000, 90, 40.0, 90, {}, True),
            QuantResult("c", "Q8", 7.0, 11000, 80, 30.0, 95, {}, True),
        ]
        rec = recommend(results)
        assert rec.model_tag == "a"  # fastest with quality >= 80%

    def test_recommend_picks_highest_quality_when_all_below(self):
        from quant_sim.bench import recommend, QuantResult
        results = [
            QuantResult("a", "Q3", 3.0, 7000, 100, 60.0, 50, {}, True),
            QuantResult("b", "Q4", 4.0, 8000, 90, 50.0, 70, {}, True),
        ]
        rec = recommend(results)
        assert rec.model_tag == "b"  # highest quality

    def test_recommend_skips_errors(self):
        from quant_sim.bench import recommend, QuantResult
        results = [
            QuantResult("a", "Q4", 4.0, 0, 0, 0, 0, {}, True, error="Inference failed"),
            QuantResult("b", "Q5", 5.0, 9000, 90, 40.0, 60, {}, True),
        ]
        rec = recommend(results)
        assert rec.model_tag == "b"

    def test_recommend_none_when_empty(self):
        from quant_sim.bench import recommend
        assert recommend([]) is None


class TestOllamaModule:
    def test_extract_quant_from_name(self):
        from quant_sim.ollama import _extract_quant_from_name
        assert _extract_quant_from_name("model:q4_k_m") == "Q4_K_M"
        assert _extract_quant_from_name("model:q8_0") == "Q8_0"
        assert _extract_quant_from_name("model:latest") == "default"

    def test_discover_includes_base_model(self):
        from quant_sim.ollama import discover_quant_tags
        tags = discover_quant_tags("qwen2.5:7b")
        assert "qwen2.5:7b" in tags
        assert len(tags) > 1
