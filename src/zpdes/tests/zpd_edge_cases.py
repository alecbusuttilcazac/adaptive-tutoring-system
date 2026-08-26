# ZPDGraph batch-construction edge case test plan:
#   1. A valid prerequisite chain constructs successfully.
#   2. A self-loop (concept requiring itself) raises GraphHasCycleError.
#   3. A cycle closed through several intermediate concepts raises GraphHasCycleError.
#   4. A pure cycle with no root concept at all (e.g. A->B->C->A, nothing outside it) is
#      still detected -- a root-only scan would miss this entirely, since there are zero
#      roots to start a traversal from.
#   5. A rejected (cyclic) batch never gets to construct any ZPDGraph object at all --
#      the whole point of validating before building any graph state.
#   6. A prerequisite referencing a concept never included in the batch at all raises
#      DanglingPrerequisiteError, instead of surfacing as an opaque KeyError.
#   7. A node whose level is lower than one of its prerequisites' levels raises
#      LevelInconsistencyError -- structurally fine (no cycle, no dangling reference)
#      but never actually reachable through level-gating.
#   8. complete_concept/advance_level end to end: a level only advances once every
#      node at that level is completed (not just one of several), and newly-eligible
#      nodes at the unlocked level are correctly promoted out of `unreachable`.
#
# ZPDGraph is now a batch-only constructor: list[tuple[Concept, int, set[Concept]]],
# where each tuple is (concept, difficulty_level, prerequisite_concepts). Mock Concepts
# stand in for what would normally come from Phase 2's content pipeline.

import pytest
from zpdes.zpd import (
    ZPDGraph,
    GraphHasCycleError,
    DanglingPrerequisiteError,
    LevelInconsistencyError,
)
from structures.structures import Concept


def test_valid_chain_constructs():
    a, b, c = Concept("A", 1), Concept("B", 2), Concept("C", 3)
    concepts = [
        (a, 1, set()),
        (b, 1, {a}),
        (c, 1, {b}),
    ]
    graph = ZPDGraph(concepts)

    node_a = graph.concept_id_to_node[a.id]
    node_b = graph.concept_id_to_node[b.id]
    node_c = graph.concept_id_to_node[c.id]
    assert node_a in node_b.prerequisites
    assert node_b in node_c.prerequisites
    assert len(node_a.prerequisites) == 0


def test_self_loop_raises():
    a = Concept("A", 1)
    concepts = [(a, 1, {a})]
    with pytest.raises(GraphHasCycleError):
        ZPDGraph(concepts)


def test_multi_node_cycle_raises():
    a, b, c, d = Concept("A", 1), Concept("B", 2), Concept("C", 3), Concept("D", 4)
    # A -> D -> C -> B -> A: a 4-node cycle, closed by A depending on D.
    concepts = [
        (a, 1, {d}),
        (b, 1, {a}),
        (c, 1, {b}),
        (d, 1, {c}),
    ]
    with pytest.raises(GraphHasCycleError):
        ZPDGraph(concepts)


def test_pure_cycle_with_no_root_raises():
    # Every concept has a prerequisite -- there is no root (no concept with an empty
    # prerequisite set) anywhere in this batch. A root-only traversal would find zero
    # starting points and silently report no cycle; this must still be caught.
    a, b, c = Concept("A", 1), Concept("B", 2), Concept("C", 3)
    concepts = [
        (a, 1, {b}),
        (b, 1, {c}),
        (c, 1, {a}),
    ]
    with pytest.raises(GraphHasCycleError):
        ZPDGraph(concepts)


def test_rejected_batch_constructs_nothing():
    a, b = Concept("A", 1), Concept("B", 2)
    concepts = [
        (a, 1, {b}),
        (b, 1, {a}),
    ]
    graph = None
    try:
        graph = ZPDGraph(concepts)
    except GraphHasCycleError:
        pass
    assert graph is None


def test_dangling_prerequisite_raises():
    a, b = Concept("A", 1), Concept("B", 2)
    x = Concept("X", 99)  # never included as a top-level entry in the batch
    concepts = [
        (a, 1, set()),
        (b, 1, {x}),
    ]
    with pytest.raises(DanglingPrerequisiteError):
        ZPDGraph(concepts)


def test_level_inconsistency_raises():
    # B is at level 1 but depends on A, which is at level 3 -- structurally fine
    # (no cycle, no dangling reference) but never reachable through level-gating,
    # since A's level would need to unlock before B's prerequisite could ever complete.
    a, b = Concept("A", 1), Concept("B", 2)
    concepts = [
        (a, 3, set()),
        (b, 1, {a}),
    ]
    with pytest.raises(LevelInconsistencyError):
        ZPDGraph(concepts)


def test_equal_level_prerequisite_is_allowed():
    # A node's level must be >= its prerequisites' levels, not strictly greater --
    # same-level prerequisites are valid.
    a, b = Concept("A", 1), Concept("B", 2)
    concepts = [(a, 1, set()), (b, 1, {a})]
    graph = ZPDGraph(concepts)
    assert graph.concept_id_to_node[a.id] in graph.concept_id_to_node[b.id].prerequisites


def test_level_advances_only_once_fully_cleared():
    # Two level-1 roots, one level-2 node depending on both. Completing only one
    # root must NOT advance current_level -- the level isn't cleared until every
    # node at that level is completed.
    a, b, c = Concept("A", 1), Concept("B", 2), Concept("C", 3)
    concepts = [
        (a, 1, set()),
        (b, 1, set()),
        (c, 2, {a, b}),
    ]
    graph = ZPDGraph(concepts)

    graph.complete_concept(a)
    assert graph.current_level == 1

    graph.complete_concept(b)
    assert graph.current_level == 2


def test_completing_level_promotes_dependents_to_eligible():
    # Once level 2 unlocks, C (needing both A and B) and D (needing only A) must
    # both move from `unreachable` to `eligible` -- exercises the subset check in
    # advance_level's promotion loop, not just membership of a single prerequisite.
    a, b, c, d = Concept("A", 1), Concept("B", 2), Concept("C", 3), Concept("D", 4)
    concepts = [
        (a, 1, set()),
        (b, 1, set()),
        (c, 2, {a, b}),
        (d, 2, {a}),
    ]
    graph = ZPDGraph(concepts)
    node_c = graph.concept_id_to_node[c.id]
    node_d = graph.concept_id_to_node[d.id]
    assert node_c in graph.unreachable
    assert node_d in graph.unreachable

    graph.complete_concept(a)
    graph.complete_concept(b)

    assert node_c in graph.eligible
    assert node_d in graph.eligible
    assert node_c not in graph.unreachable
    assert node_d not in graph.unreachable
