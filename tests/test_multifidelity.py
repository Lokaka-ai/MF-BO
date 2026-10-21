import csv
from types import SimpleNamespace

import pytest

from engine.multifidelity import bind_view, full_node_ids, prepare_view


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def test_frozen_uniform_half_view_and_full_validation(tmp_path):
    protocol = tmp_path / "protocol"
    public = protocol / "public"
    write_csv(public / "train.csv", ["sample_id", "feature", "label"],
              [(str(i), f"asset/{i}.txt", i % 2) for i in range(7)])
    for name in ("validation.csv", "test.csv", "sample_submission.csv"):
        write_csv(public / name, ["sample_id", "feature"], [[name, "full"]])
    write_csv(protocol / "assets.csv", ["sample_id", "public_path", "role"],
              [(str(i), f"asset/{i}.txt", "feature") for i in range(7)]
              + [["v", "asset/v.txt", "feature"]])
    for name in [*(str(i) for i in range(7)), "v"]:
        path = public / "asset" / f"{name}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    view, manifest = prepare_view(protocol, tmp_path / "work", 42)
    with (view / "train.csv").open(newline="") as stream:
        selected = {row["sample_id"] for row in csv.DictReader(stream)}
    assert len(selected) == 3
    assert manifest["n_selected"] == 3
    assert (view / "validation.csv").read_bytes() == (public / "validation.csv").read_bytes()
    assert {p.stem for p in (view / "asset").iterdir()} == selected | {"v"}
    assert prepare_view(protocol, tmp_path / "work", 42)[1] == manifest
    with pytest.raises(ValueError, match="changed"):
        prepare_view(protocol, tmp_path / "work", 43)
    bind_view(tmp_path / "work", view)
    assert (tmp_path / "work" / "input").resolve() == view.resolve()
    bind_view(tmp_path / "work", public, (view,))
    assert (tmp_path / "work" / "input").resolve() == public.resolve()


def test_full_node_ids_ignore_cheap_and_missing_snapshots():
    records = [SimpleNamespace(child_node_id="cheap", metadata={"node_snapshot": {"node": {
                   "id": "cheap", "training_fraction": .5}}}),
               SimpleNamespace(child_node_id="full", metadata={"node_snapshot": {"node": {
                   "id": "full", "training_fraction": 1.0}}}),
               SimpleNamespace(child_node_id="missing", metadata={})]
    assert full_node_ids(records) == {"full"}
