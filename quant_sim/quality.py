"""Quality benchmark: test factual accuracy, math, coding, reasoning."""

from dataclasses import dataclass


@dataclass
class QualityQuestion:
    category: str
    prompt: str
    check: str  # "contains:Paris" or "exact:4" or "code:def"
    weight: float = 1.0


# Built-in quality test set — 20 questions across 4 categories
QUALITY_TESTS = [
    # Facts (5)
    QualityQuestion("facts", "What is the capital of France? Answer with just the city name.", "contains:Paris"),
    QualityQuestion("facts", "What planet is closest to the Sun? Answer with just the planet name.", "contains:Mercury"),
    QualityQuestion("facts", "Who wrote Romeo and Juliet? Answer with just the author name.", "contains:Shakespeare"),
    QualityQuestion("facts", "What is the chemical formula for water? Just letters and numbers.", "contains:H2O"),
    QualityQuestion("facts", "How many continents are there? Answer with just the number.", "contains:7"),

    # Math (5)
    QualityQuestion("math", "What is 15 * 23? Answer with just the number.", "contains:345"),
    QualityQuestion("math", "What is the square root of 144? Answer with just the number.", "contains:12"),
    QualityQuestion("math", "What is 2^10? Answer with just the number.", "contains:1024"),
    QualityQuestion("math", "A farmer has 17 sheep. All but 9 die. How many are left? Just the number.", "contains:9"),
    QualityQuestion("math", "What is 100 - 37? Answer with just the number.", "contains:63"),

    # Coding (5)
    QualityQuestion("coding", "Write a Python function that returns True if a number is even. Just the function.", "code:def"),
    QualityQuestion("coding", "Write a Python one-liner to reverse a string s.", "code:[::-1]"),
    QualityQuestion("coding", "Write a Python function to compute factorial of n. Just the function.", "code:def"),
    QualityQuestion("coding", "What does len([1,2,3]) return in Python? Just the number.", "contains:3"),
    QualityQuestion("coding", "Write a Python list comprehension that squares numbers 1-5.", "code:["),

    # Reasoning (5)
    QualityQuestion("reasoning", "If all roses are flowers and all flowers need water, do roses need water? Yes or no.", "contains:yes", 1.0),
    QualityQuestion("reasoning", "I have a brother. My brother has a brother. How many brothers minimum? Just the number.", "contains:1"),
    QualityQuestion("reasoning", "Which is heavier: a pound of feathers or a pound of steel? Answer: they weigh the same, or one is heavier.", "contains:same"),
    QualityQuestion("reasoning", "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets? Think carefully. Just the number of minutes.", "contains:5"),
    QualityQuestion("reasoning", "A bat and ball cost $1.10 together. The bat costs $1 more than the ball. How much is the ball in cents? Just the number.", "contains:5"),
]


def grade_response(question: QualityQuestion, response: str) -> bool:
    """Grade a response against expected answer."""
    response_lower = response.lower().strip()

    # Strip thinking tags (Qwen uses <think>...</think>)
    import re
    clean = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    if clean:
        response_lower = clean.lower().strip()

    check_type, check_value = question.check.split(":", 1)

    if check_type == "contains":
        return check_value.lower() in response_lower
    elif check_type == "exact":
        return check_value.lower() == response_lower
    elif check_type == "code":
        return check_value in response
    return False


def run_quality_benchmark(generate_fn, n_questions: int = 20) -> dict:
    """
    Run quality benchmark using the provided generate function.

    generate_fn(prompt: str) -> str (response text)

    Returns: {"score": 0-100, "by_category": {...}, "details": [...]}
    """
    questions = QUALITY_TESTS[:n_questions]
    results = []
    by_category = {}

    for q in questions:
        response = generate_fn(q.prompt)
        correct = grade_response(q, response)
        results.append({
            "category": q.category,
            "prompt": q.prompt[:50],
            "correct": correct,
            "response": response[:100],
        })

        if q.category not in by_category:
            by_category[q.category] = {"correct": 0, "total": 0}
        by_category[q.category]["total"] += 1
        if correct:
            by_category[q.category]["correct"] += 1

    total_correct = sum(1 for r in results if r["correct"])
    total = len(results)

    return {
        "score": round(total_correct / total * 100, 1) if total > 0 else 0,
        "correct": total_correct,
        "total": total,
        "by_category": {k: round(v["correct"] / v["total"] * 100) for k, v in by_category.items()},
        "details": results,
    }
