import json

from engine.search_node import Journal, SearchNode, filter_on_path
from utils.metric import MetricValue
from utils.serialize import dumps_json, loads_json


def test_filter_on_path_deepcopies_bidirectional_node_cycles():
    root = SearchNode(
        code="",
        stage="root",
        metric=MetricValue(value=0.0, maximize=True),
        is_buggy=False,
    )
    child = SearchNode(
        code="solution",
        stage="draft",
        parent=root,
        metric=MetricValue(value=1.0, maximize=True),
        is_buggy=False,
        _term_out=["large output"],
    )
    child.children.add(root)

    filtered = filter_on_path(Journal([root, child]), [root.id, child.id])

    assert [node.id for node in filtered.nodes] == [root.id, child.id]
    assert filtered.nodes[0] is not root
    assert filtered.nodes[1] is not child
    assert filtered.nodes[1]._term_out == "<OMITTED>"
    assert filtered.nodes[1].parent is filtered.nodes[0]

    serialized = json.loads(dumps_json(filtered))
    assert [node["id"] for node in serialized["nodes"]] == [root.id, child.id]
    assert serialized["node2parent"] == {child.id: root.id}


def test_journal_roundtrip_restores_parent_and_local_best_links():
    root = SearchNode(
        id="root",
        code="",
        stage="root",
        metric=MetricValue(value=0.0, maximize=True),
        is_buggy=False,
    )
    child = SearchNode(
        id="child",
        code="solution",
        stage="draft",
        parent=root,
        metric=MetricValue(value=1.0, maximize=True),
        is_buggy=False,
    )
    child.local_best_node = child

    restored = loads_json(dumps_json(Journal([root, child])), Journal)

    assert restored[1].parent is restored[0]
    assert restored[1] in restored[0].children
    assert restored[1].local_best_node is restored[1]
