# ZPDGraph batch-construction edge case test plan:
#   1. A valid prerequisite chain constructs successfully.
#   2. A self-loop (concept requiring itself) raises GraphHasCycleError.
#   3. A cycle closed through several intermediate concepts raises GraphHasCycleError.
#   4. A pure cycle with no root concept at all (e.g. A->B->C->A, nothing outside it) is
#      still detected -- a root-only scan would miss this entirely, since there are zero
#      roots to start a traversal from.
#   5. A rejected (cyclic) batch never gets to construct any ZPDGraph object at all --
#      the whole point of validating before building any graph state.
#
# ZPDGraph is now a batch-only constructor: list[tuple[Concept, int, set[Concept]]],
# where each tuple is (concept, difficulty_level, prerequisite_concepts). Mock Concepts
# stand in for what would normally come from Phase 2's content pipeline.

import pytest
from zpdes.zpdes import ZPDGraph, GraphHasCycleError
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
