import copy
import json
from pathlib import Path
from typing import Type, TypeVar
from dataclasses import fields

import dataclasses_json


def _get_journal_cls():
    from engine.search_node import Journal
    return Journal


def dumps_json(obj: dataclasses_json.DataClassJsonMixin):
    """Serialize dataclasses (such as Journals) to JSON."""
    Journal = _get_journal_cls()
    if isinstance(obj, Journal):
        # Completed nodes can carry a live decision context that refers back
        # to the whole search graph. Deep-copying that graph before severing
        # links grows with the search and can exceed Python's recursion limit.
        # Only dataclass fields are part of the journal's persisted contract.
        node2parent = {n.id: n.parent.id for n in obj.nodes if n.parent is not None}
        node2best_local_node = {
            n.id: n.local_best_node.id for n in obj.nodes
            if n.local_best_node is not None and n.local_best_node.metric is not None
            and n.local_best_node.metric.value is not None
        }
        detached = copy.copy(obj)
        detached.nodes = []
        for node in obj.nodes:
            clone = type(node).__new__(type(node))
            for member in fields(node):
                value = getattr(node, member.name)
                if member.name in ("parent", "local_best_node", "child_count_lock"):
                    value = None
                elif member.name == "children":
                    value = set()
                else:
                    value = copy.deepcopy(value)
                setattr(clone, member.name, value)
            detached.nodes.append(clone)
        for member in fields(obj):
            if member.name != "nodes":
                setattr(detached, member.name, copy.deepcopy(getattr(obj, member.name)))
        obj = detached

    obj_dict = obj.to_dict()

    if isinstance(obj, Journal):
        obj_dict["node2parent"] = node2parent  # type: ignore
        obj_dict["node2best_local_node"] = node2best_local_node
        obj_dict["__version"] = "3"

    def _json_default(o):
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, set):
            return list(o)
        to_dict = getattr(o, "to_dict", None)
        if callable(to_dict):
            try:
                return to_dict()
            except Exception:
                pass
        return str(o)

    return json.dumps(obj_dict, separators=(",", ":"), default=_json_default)


def dump_json(obj: dataclasses_json.DataClassJsonMixin, path: Path):
    tmp_path = path.with_suffix(".json.tmp")
    data = dumps_json(obj)
    with open(tmp_path, "w") as f:
        f.write(data)
    tmp_path.replace(path)


G = TypeVar("G", bound=dataclasses_json.DataClassJsonMixin)


def loads_json(s: str, cls: Type[G]) -> G:
    """Deserialize JSON to dataclasses."""
    Journal = _get_journal_cls()
    obj_dict = json.loads(s)
    obj = cls.from_dict(obj_dict)

    if isinstance(obj, Journal):
        id2nodes = {n.id: n for n in obj.nodes}
        for child_id, parent_id in obj_dict["node2parent"].items():
            id2nodes[child_id].parent = id2nodes[parent_id]
            id2nodes[child_id].__post_init__()
        for node_id, local_best_id in obj_dict.get(
            "node2best_local_node", {}
        ).items():
            if node_id in id2nodes and local_best_id in id2nodes:
                id2nodes[node_id].local_best_node = id2nodes[local_best_id]
    return obj


def load_json(path: Path, cls: Type[G]) -> G:
    with open(path, "r") as f:
        return loads_json(f.read(), cls)
