# T-SKIRT edge case test plan:
#   1. T=1 (single response) -- does the Wiener-process prior handle an empty diff array?
#   2. All-correct / all-incorrect streaks -- does fit_MAP still converge cleanly?
#   3. Extreme sigma (near-0, very large) -- graceful degradation, no nan/inf?
#   4. Extreme lambda_forget (very large decay) -- same question
#   5. A concept with zero responses in fit_all_concepts -- does anything assume nonzero data?
#
# Each case just checks: does it run without raising, does it converge (where applicable),
# and is the result free of nan/inf. This is a stress-test sanity pass, not a correctness
# re-verification (that was already done via scipy.optimize.check_grad during development).

import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tskirt import (
    QuestionType, Question, Answer, QuestionAnswer, Student,
    fit_MAP, fit_all_concepts, LAMBDA_FORGET
)


def make_qas(n, concept_id=0, correct_pattern=None, rng=None):
    rng = rng or np.random.default_rng(0)
    qas = []
    for i in range(n):
        q = Question(concept_id, QuestionType.MULTIPLE_CHOICE, difficulty=rng.uniform(-1, 1))
        correct = correct_pattern[i] if correct_pattern is not None else bool(rng.integers(0, 2))
        qas.append(QuestionAnswer(q, Answer(correct), time_since_last=1.0))
    return qas


def assert_converged(result):
    assert result.success, f"optimizer did not converge: {result.message}"
    assert np.all(np.isfinite(result.x)), f"non-finite theta values: {result.x}"


def test_single_response():
    qas = make_qas(1)
    result = fit_MAP(qas, lambda_forget=LAMBDA_FORGET)
    assert_converged(result)


def test_all_correct():
    qas = make_qas(20, correct_pattern=[True] * 20)
    result = fit_MAP(qas, lambda_forget=LAMBDA_FORGET)
    assert_converged(result)


def test_all_incorrect():
    qas = make_qas(20, correct_pattern=[False] * 20)
    result = fit_MAP(qas, lambda_forget=LAMBDA_FORGET)
    assert_converged(result)


def test_extreme_sigma_small():
    qas = make_qas(20)
    result = fit_MAP(qas, sigma=1e-6, lambda_forget=LAMBDA_FORGET)
    assert_converged(result)


def test_extreme_sigma_large():
    qas = make_qas(20)
    result = fit_MAP(qas, sigma=1e6, lambda_forget=LAMBDA_FORGET)
    assert_converged(result)


def test_extreme_lambda_forget():
    qas = make_qas(20)
    result = fit_MAP(qas, lambda_forget=100.0)
    assert_converged(result)


def test_zero_response_concept():
    student = Student(id=1)
    student.history = make_qas(20, concept_id=10)
    # No responses at all for concept_id=20 -- fit_all_concepts should simply not
    # produce an entry for it, not crash trying to fit an empty history.
    theta_by_concept = fit_all_concepts(student)
    assert 10 in theta_by_concept
    assert 20 not in theta_by_concept
