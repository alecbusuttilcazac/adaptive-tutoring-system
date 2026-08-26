# ZPDES test plan:
#   1. reward_function is a weighted average of the recent theta diffs, weighted so the
#      most recent diff counts most -- a single big diff right at the end should move the
#      reward more than the same-sized diff further back.
#   2. reward_function raises when given fewer than 2 thetas (no diff can be computed at all).
#   3. reward_function only looks at the last REWARD_FUNCTION_NUM_THETAS + 1 thetas -- ancient
#      history beyond that window must not affect the result.
#   4. confidence_penalty is 0 once a node has been exercised >= CONFIDENCE_PENALTY_THRESHOLD
#      times, and strictly positive (and decreasing) below that.
#   5. update_concept increments times_exercised and sets latest_reward to
#      reward_function(thetas) minus the confidence penalty.
#   6. update_concept does NOT mark a concept as mastered/completed if theta crosses
#      MASTERY_THETA_THRESHOLD but times_exercised is still below CONFIDENCE_PENALTY_THRESHOLD --
#      a single lucky early answer should not be enough on its own.
#   7. update_concept DOES mark a concept as mastered/completed once theta is at/above
#      MASTERY_THETA_THRESHOLD and times_exercised has caught up to the confidence threshold --
#      the node moves out of graph.eligible and into graph.completed.
#   8. select_concept only ever returns a node that is currently in graph.eligible.
#   9. select_concept overwhelmingly favours a node with a much higher reward than its
#      competitors (softmax should be close to greedy for a large reward gap).
#  10. select_concept is roughly uniform across nodes whose rewards are all equal (no
#      reward signal to prefer one node over another).

import numpy as np
import pytest

from zpdes.zpdes import (
    ZPDES,
    CONFIDENCE_PENALTY_THRESHOLD,
    MASTERY_THETA_THRESHOLD,
    REWARD_FUNCTION_NUM_THETAS,
)
from structures.structures import Concept


def make_zpdes(*concepts_and_levels):
    # Builds a flat, single-level ZPDES graph (no prerequisites) out of (concept, level)
    # pairs, all at level 1 so every node starts eligible -- convenient for selection tests
    # that don't care about the graph's level-gating behaviour.
    concepts = [(concept, level, set()) for concept, level in concepts_and_levels]
    return ZPDES(concepts)


def test_reward_function_weighs_recent_diffs_more():
    # Same-sized jump (+1.0), but happening at the very end of the window vs. near the
    # start -- the recent one should produce a larger reward since WEIGHTS decays with age.
    recent_jump = np.concatenate([np.zeros(10), [0.0, 1.0]])
    old_jump = np.concatenate([[0.0, 1.0], np.zeros(10)])

    reward_recent = ZPDES.reward_function(recent_jump)
    reward_old = ZPDES.reward_function(old_jump)

    assert reward_recent > reward_old


def test_reward_function_raises_with_fewer_than_two_thetas():
    with pytest.raises(RuntimeError):
        ZPDES.reward_function(np.array([0.0]))


def test_reward_function_ignores_history_beyond_window():
    # Ancient history (a huge early spike) far outside the REWARD_FUNCTION_NUM_THETAS + 1
    # window should not affect the reward at all.
    tail = np.linspace(0.0, 0.5, REWARD_FUNCTION_NUM_THETAS + 1)
    with_ancient_spike = np.concatenate([[100.0], tail])
    without_spike = tail

    assert ZPDES.reward_function(with_ancient_spike) == pytest.approx(
        ZPDES.reward_function(without_spike)
    )


def test_confidence_penalty_vanishes_after_threshold_exercises():
    zpdes = make_zpdes((Concept("A", 1), 1))
    node = next(iter(zpdes.graph.eligible))

    node.times_exercised = CONFIDENCE_PENALTY_THRESHOLD
    assert zpdes.confidence_penalty(node) == 0.0

    node.times_exercised = CONFIDENCE_PENALTY_THRESHOLD + 5
    assert zpdes.confidence_penalty(node) == 0.0


def test_confidence_penalty_decreases_towards_threshold():
    zpdes = make_zpdes((Concept("A", 1), 1))
    node = next(iter(zpdes.graph.eligible))

    node.times_exercised = 0
    penalty_at_zero = zpdes.confidence_penalty(node)
    node.times_exercised = CONFIDENCE_PENALTY_THRESHOLD - 1
    penalty_near_threshold = zpdes.confidence_penalty(node)

    assert penalty_at_zero > penalty_near_threshold > 0.0


def test_update_concept_sets_reward_and_increments_exercise_count():
    concept = Concept("A", 1)
    zpdes = make_zpdes((concept, 1))
    node = zpdes.graph.concept_id_to_node[concept.id]

    thetas = np.linspace(0.0, 0.3, REWARD_FUNCTION_NUM_THETAS)
    zpdes.update_concept(concept, thetas)

    assert node.times_exercised == 1
    expected_reward = ZPDES.reward_function(thetas) - zpdes.confidence_penalty(node)
    assert node.latest_reward == pytest.approx(expected_reward)


def test_update_concept_does_not_master_with_too_little_history():
    # theta is well above the mastery threshold, but this is the node's first exercise --
    # a single lucky reading should not be enough to complete the concept.
    concept = Concept("A", 1)
    zpdes = make_zpdes((concept, 1))

    thetas = np.full(2, MASTERY_THETA_THRESHOLD + 1.0)
    zpdes.update_concept(concept, thetas)

    node = zpdes.graph.concept_id_to_node[concept.id]
    assert node in zpdes.graph.eligible
    assert node not in zpdes.graph.completed


def test_update_concept_masters_once_theta_and_history_both_qualify():
    concept = Concept("A", 1)
    zpdes = make_zpdes((concept, 1))
    node = zpdes.graph.concept_id_to_node[concept.id]

    thetas = np.full(2, MASTERY_THETA_THRESHOLD + 1.0)
    for _ in range(CONFIDENCE_PENALTY_THRESHOLD):
        zpdes.update_concept(concept, thetas)

    assert node not in zpdes.graph.eligible
    assert node in zpdes.graph.completed


def test_select_concept_only_returns_eligible_nodes():
    a, b, c = Concept("A", 1), Concept("B", 2), Concept("C", 3)
    zpdes = make_zpdes((a, 1), (b, 1), (c, 1))

    node_a = zpdes.graph.concept_id_to_node[a.id]
    node_b = zpdes.graph.concept_id_to_node[b.id]
    node_c = zpdes.graph.concept_id_to_node[c.id]
    node_a.latest_reward = 0.05
    node_b.latest_reward = -0.05
    node_c.latest_reward = 0.0

    for _ in range(50):
        selected = zpdes.select_concept()
        assert selected in zpdes.graph.eligible


def test_select_concept_favours_dominant_reward():
    a, b = Concept("A", 1), Concept("B", 2)
    zpdes = make_zpdes((a, 1), (b, 1))

    node_a = zpdes.graph.concept_id_to_node[a.id]
    node_b = zpdes.graph.concept_id_to_node[b.id]
    node_a.latest_reward = 10.0  # far larger than realistic reward_function output
    node_b.latest_reward = -10.0

    selections = [zpdes.select_concept() for _ in range(200)]
    assert selections.count(node_a) > 190


def test_select_concept_roughly_uniform_when_rewards_equal():
    a, b = Concept("A", 1), Concept("B", 2)
    zpdes = make_zpdes((a, 1), (b, 1))

    node_a = zpdes.graph.concept_id_to_node[a.id]
    node_b = zpdes.graph.concept_id_to_node[b.id]
    node_a.latest_reward = 0.05
    node_b.latest_reward = 0.05

    selections = [zpdes.select_concept() for _ in range(200)]
    count_a = selections.count(node_a)
    # With 200 draws at p=0.5 each, an even split should land well within a generous margin.
    assert 60 < count_a < 140
