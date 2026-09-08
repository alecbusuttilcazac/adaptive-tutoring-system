# ZPDBandit test plan:
#   1. reward_function is a weighted average of the recent theta diffs, weighted so the
#      most recent diff counts most -- a single big diff right at the end should move the
#      reward more than the same-sized diff further back.
#   2. reward_function raises when given fewer than 2 thetas (no diff can be computed at all).
#   3. reward_function only looks at the last REWARD_FUNCTION_NUM_THETAS + 1 thetas -- ancient
#      history beyond that window must not affect the result.
#   4. ucb raises if the node has never been exercised, or if t == 0 -- callers must only call
#      it once a node has actually been played at least once (matches UCB1's own structure:
#      initialization plays every arm once, only then does the confidence-width formula apply).
#   5. ucb decreases as a node's own times_exercised grows (more personal evidence -> smaller
#      confidence width), for a fixed t.
#   6. ucb increases as t (total exercises across current eligible nodes) grows, for a fixed
#      times_exercised -- more overall experience raises the bar for what counts as "enough."
#   7. update_concept increments times_exercised and sets latest_reward to
#      reward_function(thetas) plus the UCB-scaled confidence bonus (added, not subtracted --
#      UCB is optimistic).
#   8. update_concept does NOT mark a concept as mastered/completed if theta crosses
#      MASTERY_THETA_THRESHOLD but times_exercised is still below MINIMUM_EXERCISES_THRESHOLD --
#      a single lucky early answer should not be enough on its own.
#   9. update_concept DOES mark a concept as mastered/completed once theta is at/above
#      MASTERY_THETA_THRESHOLD and times_exercised has caught up to the minimum threshold --
#      the node moves out of graph.eligible and into graph.completed.
#  10. select_concept forces play of any never-exercised eligible node before ever considering
#      softmax -- with a mix of tried and untried nodes, only untried ones are ever returned.
#  11. select_concept only ever returns a node that is currently in graph.eligible.
#  12. select_concept overwhelmingly favours a node with a much higher reward than its
#      competitors (softmax should be close to greedy for a large reward gap), once every
#      node has already been exercised at least once (so forced-play doesn't interfere).
#  13. select_concept is roughly uniform across nodes whose rewards are all equal (no
#      reward signal to prefer one node over another), same once-exercised precondition.

import numpy as np
import pytest

from contentsequencing.bandit import (
    ZPDBandit,
    MASTERY_THETA_THRESHOLD,
    MINIMUM_EXERCISES_THRESHOLD,
    REWARD_FUNCTION_NUM_THETAS,
)
from structures.structures import Concept


def make_bandit(*concepts_and_levels):
    # Builds a flat, single-level ZPDBandit graph (no prerequisites) out of (concept, level)
    # pairs, all at level 1 so every node starts eligible -- convenient for selection tests
    # that don't care about the graph's level-gating behaviour.
    concepts = [(concept, level, set()) for concept, level in concepts_and_levels]
    return ZPDBandit(concepts)


def test_reward_function_weighs_recent_diffs_more():
    # Same-sized jump (+1.0), but happening at the very end of the window vs. near the
    # start -- the recent one should produce a larger reward since WEIGHTS decays with age.
    recent_jump = np.concatenate([np.zeros(10), [0.0, 1.0]])
    old_jump = np.concatenate([[0.0, 1.0], np.zeros(10)])

    reward_recent = ZPDBandit.reward_function(recent_jump)
    reward_old = ZPDBandit.reward_function(old_jump)

    assert reward_recent > reward_old


def test_reward_function_raises_with_fewer_than_two_thetas():
    with pytest.raises(RuntimeError):
        ZPDBandit.reward_function(np.array([0.0]))


def test_reward_function_ignores_history_beyond_window():
    # Ancient history (a huge early spike) far outside the REWARD_FUNCTION_NUM_THETAS + 1
    # window should not affect the reward at all.
    tail = np.linspace(0.0, 0.5, REWARD_FUNCTION_NUM_THETAS + 1)
    with_ancient_spike = np.concatenate([[100.0], tail])
    without_spike = tail

    assert ZPDBandit.reward_function(with_ancient_spike) == pytest.approx(
        ZPDBandit.reward_function(without_spike)
    )


def test_ucb_raises_for_untried_node_or_zero_t():
    bandit = make_bandit((Concept("A", 1), 1))
    node = next(iter(bandit.graph.eligible))

    with pytest.raises(RuntimeError):
        bandit.ucb(node, t=5)

    node.times_exercised = 1
    with pytest.raises(RuntimeError):
        bandit.ucb(node, t=0)


def test_ucb_decreases_with_own_exercise_count():
    bandit = make_bandit((Concept("A", 1), 1))
    node = next(iter(bandit.graph.eligible))

    node.times_exercised = 1
    width_few = bandit.ucb(node, t=100)
    node.times_exercised = 20
    width_many = bandit.ucb(node, t=100)

    assert width_few > width_many > 0.0


def test_ucb_increases_with_total_exercises():
    bandit = make_bandit((Concept("A", 1), 1))
    node = next(iter(bandit.graph.eligible))
    node.times_exercised = 5

    width_early = bandit.ucb(node, t=10)
    width_later = bandit.ucb(node, t=1000)

    assert width_later > width_early


def test_update_concept_sets_reward_and_increments_exercise_count():
    concept = Concept("A", 1)
    bandit = make_bandit((concept, 1))
    node = bandit.graph.concept_id_to_node[concept.id]

    thetas = np.linspace(0.0, 0.3, REWARD_FUNCTION_NUM_THETAS)
    bandit.update_concept(concept, thetas)

    assert node.times_exercised == 1
    # After one exercise, t == this node's own times_exercised (only eligible node).
    expected_bonus_input = bandit.ucb(node, t=1)
    from contentsequencing.bandit import UCB_SCALE
    expected_reward = ZPDBandit.reward_function(thetas) + UCB_SCALE * expected_bonus_input
    assert node.latest_reward == pytest.approx(expected_reward)


def test_update_concept_does_not_master_with_too_little_history():
    # theta is well above the mastery threshold, but this is the node's first exercise --
    # a single lucky reading should not be enough to complete the concept.
    concept = Concept("A", 1)
    bandit = make_bandit((concept, 1))

    thetas = np.full(2, MASTERY_THETA_THRESHOLD + 1.0)
    bandit.update_concept(concept, thetas)

    node = bandit.graph.concept_id_to_node[concept.id]
    assert node in bandit.graph.eligible
    assert node not in bandit.graph.completed


def test_update_concept_masters_once_theta_and_history_both_qualify():
    concept = Concept("A", 1)
    bandit = make_bandit((concept, 1))
    node = bandit.graph.concept_id_to_node[concept.id]

    thetas = np.full(2, MASTERY_THETA_THRESHOLD + 1.0)
    for _ in range(MINIMUM_EXERCISES_THRESHOLD):
        bandit.update_concept(concept, thetas)

    assert node not in bandit.graph.eligible
    assert node in bandit.graph.completed


def test_select_concept_forces_play_of_untried_nodes():
    a, b = Concept("A", 1), Concept("B", 2)
    bandit = make_bandit((a, 1), (b, 1))

    node_a = bandit.graph.concept_id_to_node[a.id]
    node_b = bandit.graph.concept_id_to_node[b.id]
    # b has already been exercised; a has not -- a must always be forced regardless of reward.
    node_b.times_exercised = 5
    node_b.latest_reward = 100.0  # deliberately dominant reward, should still lose to forcing
    node_a.times_exercised = 0

    for _ in range(20):
        assert bandit.select_concept() is node_a


def test_select_concept_only_returns_eligible_nodes():
    a, b, c = Concept("A", 1), Concept("B", 2), Concept("C", 3)
    bandit = make_bandit((a, 1), (b, 1), (c, 1))

    node_a = bandit.graph.concept_id_to_node[a.id]
    node_b = bandit.graph.concept_id_to_node[b.id]
    node_c = bandit.graph.concept_id_to_node[c.id]
    for node, reward in ((node_a, 0.05), (node_b, -0.05), (node_c, 0.0)):
        node.times_exercised = 1
        node.latest_reward = reward

    for _ in range(50):
        selected = bandit.select_concept()
        assert selected in bandit.graph.eligible


def test_select_concept_favours_dominant_reward():
    a, b = Concept("A", 1), Concept("B", 2)
    bandit = make_bandit((a, 1), (b, 1))

    node_a = bandit.graph.concept_id_to_node[a.id]
    node_b = bandit.graph.concept_id_to_node[b.id]
    node_a.times_exercised = 1
    node_b.times_exercised = 1
    node_a.latest_reward = 10.0  # far larger than realistic reward_function output
    node_b.latest_reward = -10.0

    selections = [bandit.select_concept() for _ in range(200)]
    assert selections.count(node_a) > 190


def test_select_concept_roughly_uniform_when_rewards_equal():
    a, b = Concept("A", 1), Concept("B", 2)
    bandit = make_bandit((a, 1), (b, 1))

    node_a = bandit.graph.concept_id_to_node[a.id]
    node_b = bandit.graph.concept_id_to_node[b.id]
    node_a.times_exercised = 1
    node_b.times_exercised = 1
    node_a.latest_reward = 0.05
    node_b.latest_reward = 0.05

    selections = [bandit.select_concept() for _ in range(200)]
    count_a = selections.count(node_a)
    # With 200 draws at p=0.5 each, an even split should land well within a generous margin.
    assert 60 < count_a < 140
