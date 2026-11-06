import json
from types import SimpleNamespace

from agents.memory.global_memory import GlobalMemoryLayer


def test_resume_memory_loads_only_admitted_journal_nodes(tmp_path):
    records = [
        {
            "record_id": "node_keep",
            "title": "improve - keep",
            "description": "committed",
            "method": "method",
            "label": 1,
            "current_metric": 0.8,
        },
        {
            "record_id": "node_orphan",
            "title": "improve - orphan",
            "description": "not committed",
            "method": "method",
            "label": 1,
            "current_metric": 0.9,
        },
    ]
    (tmp_path / "records.json").write_text(json.dumps(records))
    indexed = {}

    layer = GlobalMemoryLayer.__new__(GlobalMemoryLayer)
    layer.memory_dir = tmp_path
    layer.admitted_record_ids = {"node_keep"}
    layer.records = []
    layer.node_metadata_map = {}
    layer.retriever = SimpleNamespace(
        build_index=lambda kept, texts: indexed.update(
            records=list(kept),
            texts=list(texts),
        )
    )

    layer._load_memory()

    assert [record.record_id for record in layer.records] == ["node_keep"]
    assert set(layer.node_metadata_map) == {"node_keep"}
    assert [record.record_id for record in indexed["records"]] == ["node_keep"]

    layer._save_memory()
    saved_ids = {
        item["record_id"]
        for item in json.loads((tmp_path / "records.json").read_text())
    }
    assert saved_ids == {"node_keep", "node_orphan"}
