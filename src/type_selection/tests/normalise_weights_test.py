# normalise_weights() test plan:
#   1. Result always sums to 1 (within tolerance) and respects both TYPE_MIN_WEIGHT and
#      TYPE_MAX_WEIGHT, across a range of input shapes (dominant/tiny mix, multiple
#      violators, all-equal, etc).
#   2. Relative ordering between weights is preserved -- a bigger raw weight must not end
#      up smaller than a weight that started out smaller (the index-order bug fixed by the
#      scale-then-clip rewrite).
#   3. An empty list raises ValueError.
#   4. A call whose n makes TYPE_MIN_WEIGHT infeasible (n*TYPE_MIN_WEIGHT > 1) raises
#      ValueError, and likewise for TYPE_MAX_WEIGHT (n*TYPE_MAX_WEIGHT < 1).
#   5. A single weight is a deliberate exception to the bounds: it always returns [1],
#      even though 1 can fall outside [TYPE_MIN_WEIGHT, TYPE_MAX_WEIGHT] -- there's nothing
#      else to cap its dominance over.
#   6. The exact-boundary cases (n*TYPE_MIN_WEIGHT == 1, n*TYPE_MAX_WEIGHT == 1) return every
#      weight at that bound, rather than erroring or hanging.
#   7. NORM_SOFTMAX_TEMPERATURE actually affects the output: a lower temperature sharpens
#      the gap between a clear winner and the rest.

import pytest

from type_selection.type_selection import normalise_weights
from config import TYPE_MIN_WEIGHT, TYPE_MAX_WEIGHT, NORM_SOFTMAX_TEMPERATURE


def assert_valid_distribution(result: list[float]) -> None:
    assert sum(result) == pytest.approx(1.0, abs=1e-6)
    assert all(TYPE_MIN_WEIGHT - 1e-9 <= w <= TYPE_MAX_WEIGHT + 1e-9 for w in result)


@pytest.mark.parametrize("raw", [
    [10.0, -10.0],
    [5.0, 3.0, -10.0],
    [1.0, 1.0],
    [1.0, 1.0, 1.0],
    [10.0, -10.0, -10.0],
    [2.0, 1.9, 1.8, -5.0],
    [10.0, 10.0, 1.0],
])
def test_result_sums_to_one_and_respects_bounds(raw):
    assert_valid_distribution(normalise_weights(list(raw)))


def test_single_weight_always_returns_one():
    # Deliberate exception to TYPE_MIN_WEIGHT/TYPE_MAX_WEIGHT -- see test plan item 5.
    assert normalise_weights([0.3]) == [1]
    assert normalise_weights([-5.0]) == [1]


def test_ordering_is_preserved():
    # Reproduces the index-order bug: with the old lock-first-violator-found algorithm,
    # weight 0 (clamped first, purely because of its position) could end up smaller than
    # weight 2, even though weight 0 started out larger.
    raw = [0.08, 0.9, 0.02]
    result = normalise_weights(raw)
    assert result[0] > result[2]  # raw[0] (0.08) > raw[2] (0.02) must still hold after normalising


def test_empty_list_raises():
    with pytest.raises(ValueError):
        normalise_weights([])


def test_infeasible_min_weight_raises():
    # TYPE_MIN_WEIGHT=0.12: 10 weights would need 10*0.12=1.2 > 1, infeasible.
    with pytest.raises(ValueError):
        normalise_weights([1.0] * 10)


def test_infeasible_max_weight_raises(monkeypatch):
    # With the real TYPE_MAX_WEIGHT=0.7, n*TYPE_MAX_WEIGHT < 1 only happens at n=1, which
    # the deliberate single-weight exception intercepts before this check ever runs -- so
    # this branch needs a smaller TYPE_MAX_WEIGHT to actually exercise with n>=2.
    import type_selection.type_selection as ts
    monkeypatch.setattr(ts, "TYPE_MAX_WEIGHT", 0.3)
    with pytest.raises(ValueError):
        ts.normalise_weights([1.0, 1.0])  # 2*0.3=0.6 < 1, infeasible


def test_exact_min_boundary_returns_uniform_at_min(monkeypatch):
    # TYPE_MIN_WEIGHT doesn't divide evenly into 1 for any integer n with the real
    # constants, so this boundary branch is exercised with a value that does.
    import type_selection.type_selection as ts
    monkeypatch.setattr(ts, "TYPE_MIN_WEIGHT", 0.25)
    result = ts.normalise_weights([1.0, 1.0, 1.0, 1.0])  # 4*0.25=1.0 exactly
    assert result == pytest.approx([0.25] * 4)


def test_exact_max_boundary_returns_uniform_at_max(monkeypatch):
    import type_selection.type_selection as ts
    monkeypatch.setattr(ts, "TYPE_MAX_WEIGHT", 0.25)
    result = ts.normalise_weights([1.0, 1.0, 1.0, 1.0])  # 4*0.25=1.0 exactly
    assert result == pytest.approx([0.25] * 4)


def test_temperature_sharpens_the_winner():
    raw = [0.13, 0.08, -0.05]
    result = normalise_weights(list(raw))
    # With NORM_SOFTMAX_TEMPERATURE < 1, the top residual's share should be noticeably
    # larger than a plain (temperature=1) softmax would give it.
    import scipy as sp
    plain_softmax = sp.special.softmax(raw)
    if NORM_SOFTMAX_TEMPERATURE < 1.0:
        assert result[0] > plain_softmax[0]
