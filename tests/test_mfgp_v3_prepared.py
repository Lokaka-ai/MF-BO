import copy
import json
import pytest
from engine.mf_gp.v3.prepared import PreparedPrograms


def ready():
    q = PreparedPrograms()
    q.reserve("a", {"utility": .2}, "improve")
    q.generated("a", "p", "print(1)")
    return q


def test_generation_and_ready_are_not_pending_measurements():
    q = PreparedPrograms()
    q.reserve("a", {"utility": .2}, "improve")
    assert q.pending_rows() == []
    q.generated("a", "p", "print(1)")
    assert q.pending_rows() == []
    q.start("a", 1.)
    assert q.pending_rows()[0]["fidelity"] == 1.
    with pytest.raises(ValueError): q.start("a", .2)


def test_restore_each_state_without_regeneration_or_reexecution():
    q = ready()
    for change in (lambda: None, lambda: q.defer("a", "unaffordable"),
                   lambda: q.reconsider("a"), lambda: q.start("a", .2),
                   lambda: q.finish("a", success=True)):
        change()
        recovered = PreparedPrograms()
        recovered.load_state_dict(json.loads(json.dumps(q.state_dict())))
        assert recovered.state_dict() == q.state_dict()
        with pytest.raises(ValueError): recovered.reserve("a", {}, "improve")
    with pytest.raises(ValueError): recovered.finish("a", success=True)


def test_defer_releases_capacity_without_erasing_source():
    q = ready()
    for t in ("b", "c"): q.reserve(t, {}, "draft")
    with pytest.raises(ValueError): q.reserve("d", {}, "draft")
    q.defer("a", "not_affordable")
    q.reserve("d", {}, "draft")
    with pytest.raises(ValueError): q.reconsider("a")
    q.expire_unexecuted()
    assert q.entries["a"]["code"] == "print(1)"
    assert q.pending_rows() == []
    assert all(e["state"] == "expired" for e in q.entries.values())


def test_tampered_checkpoint_rejected_without_partial_restore():
    q = ready()
    state = q.state_dict()
    state["entries"]["a"]["code"] = "print(2)"
    fresh = PreparedPrograms()
    with pytest.raises(ValueError, match="hash"): fresh.load_state_dict(state)
    assert fresh.entries == {}
    with pytest.raises(ValueError): q.finish("a", success=True)
