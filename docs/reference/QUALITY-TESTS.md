# Quality Tests

All 20 questions used in the quality benchmark, organized by category. Each question has a prompt, a grading rule, and equal weight (1.0).

## Grading System

Three check types:

| Type | Format | Behavior |
|------|--------|----------|
| `contains` | `contains:X` | Response must contain X (case-insensitive) |
| `exact` | `exact:X` | Response must exactly equal X (case-insensitive) |
| `code` | `code:X` | Response must contain X (case-sensitive, for syntax) |

**Pre-processing:** Before grading, `<think>...</think>` tags are stripped from the response. This handles Qwen-style thinking models that wrap their reasoning in XML tags. If stripping leaves non-empty text, the cleaned version is used for grading.

**Scoring:** Each correct answer = 1 point. Final score = (correct / total) * 100. Per-category scores are also computed (each category has 5 questions, so each correct answer in a category = 20%).

## Category 1: Facts (5 questions)

Tests basic factual recall. All use `contains` checks.

| # | Prompt | Expected | Check |
|---|--------|----------|-------|
| 1 | What is the capital of France? Answer with just the city name. | Paris | `contains:Paris` |
| 2 | What planet is closest to the Sun? Answer with just the planet name. | Mercury | `contains:Mercury` |
| 3 | Who wrote Romeo and Juliet? Answer with just the author name. | Shakespeare | `contains:Shakespeare` |
| 4 | What is the chemical formula for water? Just letters and numbers. | H2O | `contains:H2O` |
| 5 | How many continents are there? Answer with just the number. | 7 | `contains:7` |

**Design notes:** All prompts ask for "just the X" to encourage short answers. Longer answers still pass if they contain the keyword.

## Category 2: Math (5 questions)

Tests arithmetic and word problem comprehension. All use `contains` checks.

| # | Prompt | Expected | Check |
|---|--------|----------|-------|
| 6 | What is 15 * 23? Answer with just the number. | 345 | `contains:345` |
| 7 | What is the square root of 144? Answer with just the number. | 12 | `contains:12` |
| 8 | What is 2^10? Answer with just the number. | 1024 | `contains:1024` |
| 9 | A farmer has 17 sheep. All but 9 die. How many are left? Just the number. | 9 | `contains:9` |
| 10 | What is 100 - 37? Answer with just the number. | 63 | `contains:63` |

**Design notes:** Question 9 is a classic word problem trap. Many models answer 8 (17 - 9) instead of 9 ("all but 9" means 9 remain). This tests reading comprehension under quantization.

## Category 3: Coding (5 questions)

Tests Python code generation and comprehension. Mix of `code` and `contains` checks.

| # | Prompt | Expected | Check |
|---|--------|----------|-------|
| 11 | Write a Python function that returns True if a number is even. Just the function. | A function definition | `code:def` |
| 12 | Write a Python one-liner to reverse a string s. | Slice notation | `code:[::-1]` |
| 13 | Write a Python function to compute factorial of n. Just the function. | A function definition | `code:def` |
| 14 | What does len([1,2,3]) return in Python? Just the number. | 3 | `contains:3` |
| 15 | Write a Python list comprehension that squares numbers 1-5. | A list comprehension | `code:[` |

**Design notes:** Code checks are case-sensitive (Python syntax matters). Questions 11, 13, 15 test whether the model can produce syntactically valid Python. Question 12 specifically checks for the idiomatic `[::-1]` pattern. Question 14 is a knowledge check, not a generation task.

## Category 4: Reasoning (5 questions)

Tests logical reasoning, including classic trick questions. All use `contains` checks.

| # | Prompt | Expected | Check |
|---|--------|----------|-------|
| 16 | If all roses are flowers and all flowers need water, do roses need water? Yes or no. | yes | `contains:yes` |
| 17 | I have a brother. My brother has a brother. How many brothers minimum? Just the number. | 1 | `contains:1` |
| 18 | Which is heavier: a pound of feathers or a pound of steel? Answer: they weigh the same, or one is heavier. | same | `contains:same` |
| 19 | If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets? Think carefully. Just the number of minutes. | 5 | `contains:5` |
| 20 | A bat and ball cost $1.10 together. The bat costs $1 more than the ball. How much is the ball in cents? Just the number. | 5 | `contains:5` |

**Design notes:** These are classic cognitive reflection / trick questions:
- **Q16:** Basic syllogism. Nearly all models get this right.
- **Q17:** The speaker IS the brother's brother. Minimum is 1 (the speaker and his brother). Many models say 2.
- **Q18:** A pound is a pound. Models often say steel is heavier.
- **Q19:** Each machine makes 1 widget in 5 minutes. 100 machines make 100 widgets in 5 minutes, not 100 minutes. The "5" check also matches if the model says "5 minutes".
- **Q20:** The classic CRT bat-and-ball problem. The intuitive (wrong) answer is 10 cents. Correct: ball = 5 cents, bat = $1.05. Heavily quantized models often fail this.

## Quick Mode

In quick mode (`--quick`), only the first 5 questions are used (the entire Facts category). This gives a fast quality signal but only covers factual recall.

## Category Scoring

Each category has 5 questions worth 1 point each. Category score = (correct in category / 5) * 100%.

Example output:
```
facts: 100%, math: 80%, coding: 60%, reasoning: 40%
```

This tells you which capabilities degrade first under quantization. Typically: reasoning and coding degrade before facts and math.
