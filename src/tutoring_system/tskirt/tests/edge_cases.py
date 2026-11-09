# T-SKIRT edge case test plan:
#   1. T=1 (single response) -- does the Wiener-process prior handle an empty diff array?
#   2. All-correct / all-incorrect streaks -- does fit_MAP still converge cleanly?
#   3. Extreme sigma (near-0, very large) -- graceful degradation, no nan/inf?
#   4. Extreme lambda_forget (very large decay) -- same question
#   5. A kc with zero responses in fit_all_kcs -- does anything assume nonzero data?
#
# Each case just checks: does it run without raising, does it converge (where applicable),
# and is the result free of nan/inf. This is a stress-test sanity pass, not a correctness
# re-verification (that was already done via scipy.optimize.check_grad during development).

import numpy as np

from tutoring_system.structures.entities import KC, QuestionAnswer, Area
from tutoring_system.structures.questions import MultipleChoice
from tutoring_system.tskirt.tskirt import fit_MAP, fit_all_kcs
from utils.config import config


def make_qas(n, kc=KC("", 0), correct_pattern=None, rng=None):
    rng = rng or np.random.default_rng(0)
    qas = []
    for i in range(n):
        q = MultipleChoice(kc.id, rng.uniform(-1, 1), ["a", "b", "c", "d"], 0)
        correct = correct_pattern[i] if correct_pattern is not None else bool(rng.integers(0, 2))
        qas.append(QuestionAnswer(q, correct, time_since_last=1.0))
    return qas


def assert_converged(result):
    assert result.success, f"optimizer did not converge: {result.message}"
    assert np.all(np.isfinite(result.x)), f"non-finite theta values: {result.x}"


def test_single_response():
    qas = make_qas(1)
    result = fit_MAP(qas, lambda_forget=config["LAMBDA_FORGET"])
    assert_converged(result)


def test_all_correct():
    qas = make_qas(20, correct_pattern=[True] * 20)
    result = fit_MAP(qas, lambda_forget=config["LAMBDA_FORGET"])
    assert_converged(result)


def test_all_incorrect():
    qas = make_qas(20, correct_pattern=[False] * 20)
    result = fit_MAP(qas, lambda_forget=config["LAMBDA_FORGET"])
    assert_converged(result)


def test_extreme_sigma_small():
    qas = make_qas(20)
    result = fit_MAP(qas, sigma=1e-6, lambda_forget=config["LAMBDA_FORGET"])
    assert_converged(result)


def test_extreme_sigma_large():
    qas = make_qas(20)
    result = fit_MAP(qas, sigma=1e6, lambda_forget=config["LAMBDA_FORGET"])
    assert_converged(result)


def test_extreme_lambda_forget():
    qas = make_qas(20)
    result = fit_MAP(qas, lambda_forget=100.0)
    assert_converged(result)


def test_zero_response_kc():
    area = Area(id=1)
    area.history = make_qas(20, KC("", 10))
    # No responses at all for kc_id=20 -- fit_all_kcs should simply not
    # produce an entry for it, not crash trying to fit an empty history.
    theta_by_kc = fit_all_kcs(area)
    # Keyed directly by kc_id.
    assert 10 in theta_by_kc
    assert 20 not in theta_by_kc
