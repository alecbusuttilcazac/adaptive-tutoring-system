# ZPDES graph edge case test plan:
#   1. A valid prerequisite chain is added successfully.
#   2. A self-loop (node requiring itself) is rejected, graph left unchanged.
#   3. A cycle closed through several intermediate nodes is rejected, graph left unchanged.
#   4. Rejected attempts never partially commit edges -- checked BEFORE any edge is
#      added (see add_prerequisites), not added-then-rolled-back.

from zpdes.zpdes import ZPDNode, ZPDGraph
from structures.structures import Concept


def make_graph(names: list[str]) -> tuple[ZPDGraph, dict[str, ZPDNode]]:
    graph = ZPDGraph()
    nodes = {}
    for i, name in enumerate(names, start=1):
        node = ZPDNode(Concept(name, i))
        graph.concept_id_to_node[i] = node
        nodes[name] = node
    return graph, nodes


def test_valid_chain_is_added():
    graph, n = make_graph(["A", "B", "C"])
    assert graph.add_prerequisites(n["B"], {n["A"]})
    assert graph.add_prerequisites(n["C"], {n["B"]})
    assert n["A"] in n["B"].prerequisites
    assert n["B"] in n["C"].prerequisites


def test_self_loop_rejected():
    graph, n = make_graph(["A"])
    result = graph.add_prerequisites(n["A"], {n["A"]})
    assert result is False
    assert len(n["A"].prerequisites) == 0


def test_multi_node_cycle_rejected():
    graph, n = make_graph(["A", "B", "C", "D"])
    assert graph.add_prerequisites(n["D"], {n["C"]})
    assert graph.add_prerequisites(n["C"], {n["B"]})
    assert graph.add_prerequisites(n["B"], {n["A"]})

    # A -> D would close the cycle A -> D -> C -> B -> A
    result = graph.add_prerequisites(n["A"], {n["D"]})
    assert result is False


def test_rejected_cycle_leaves_graph_unchanged():
    graph, n = make_graph(["A", "B", "C"])
    assert graph.add_prerequisites(n["B"], {n["A"]})
    assert graph.add_prerequisites(n["C"], {n["B"]})

    graph.add_prerequisites(n["A"], {n["C"]})  # rejected: would cycle

    # The rejected call must not have added anything to A's prerequisites, and the
    # valid chain built beforehand must still be exactly as it was.
    assert len(n["A"].prerequisites) == 0
    assert n["B"].prerequisites == {n["A"]}
    assert n["C"].prerequisites == {n["B"]}
