"""Durable decision states, independent of the scheduler and filesystem.

Caller holds the journal lock and persists state before dispatch. Reservations
are not observations: only executing entries enter the GP pending set.
"""
import copy
import hashlib
import math


def source_hash(code):
    return hashlib.sha256(code.encode()).hexdigest()


class PreparedPrograms:
    VERSION = 1
    STATES = {"generating", "ready", "executing", "completed", "failed", "deferred", "expired"}

    def __init__(self):
        self.entries = {}

    def reserve(self, ticket, raw_context, operator):
        if ticket in self.entries:
            raise ValueError("Duplicate generation reservation")
        if sum(e["state"] in ("generating", "ready") for e in self.entries.values()) >= 3:
            raise ValueError("Prepared program capacity exhausted")
        self.entries[ticket] = dict(state="generating", raw_context=copy.deepcopy(raw_context),
                                    operator=operator, program_id=None, code=None,
                                    code_sha256=None, fidelity=None, reason=None)

    def generated(self, ticket, program_id, code):
        e = self.entries[ticket]
        if e["state"] != "generating" or not code or not program_id:
            raise ValueError("Invalid generated-program transition")
        if any(x["program_id"] == program_id for x in self.entries.values()):
            raise ValueError("Duplicate program identity")
        e.update(state="ready", program_id=program_id, code=code, code_sha256=source_hash(code))

    def start(self, ticket, fidelity):
        e = self.entries[ticket]
        if e["state"] != "ready" or fidelity not in (.2, 1.):
            raise ValueError("Execution needs ready code and explicit fidelity")
        if source_hash(e["code"]) != e["code_sha256"]:
            raise ValueError("Prepared source changed")
        e.update(state="executing", fidelity=fidelity)

    def finish(self, ticket, *, success, reason=None):
        e = self.entries[ticket]
        if e["state"] not in ("generating", "ready", "executing"):
            raise ValueError("Attempt already terminal")
        if success and e["state"] != "executing":
            raise ValueError("Unexecuted code cannot be completed")
        e.update(state="completed" if success else "failed", reason=reason)

    def defer(self, ticket, reason):
        if not reason or self.entries[ticket]["state"] != "ready":
            raise ValueError("Only ready code can be deferred with a reason")
        self.entries[ticket].update(state="deferred", reason=reason)

    def reconsider(self, ticket):
        if self.entries[ticket]["state"] != "deferred":
            raise ValueError("Only deferred code can return to ready")
        if sum(e["state"] in ("generating", "ready") for e in self.entries.values()) >= 3:
            raise ValueError("Prepared program capacity exhausted")
        self.entries[ticket].update(state="ready", reason=None)

    def expire_unexecuted(self):
        for e in self.entries.values():
            if e["state"] in ("generating", "ready", "deferred"):
                e.update(state="expired", reason="active_budget_exhausted")

    def pending_rows(self):
        return [dict(program_id=e["program_id"], raw_context=copy.deepcopy(e["raw_context"]),
                     operator=e["operator"], fidelity=e["fidelity"])
                for e in self.entries.values() if e["state"] == "executing"]

    def state_dict(self):
        return dict(version=self.VERSION, entries=copy.deepcopy(self.entries))

    def load_state_dict(self, state):
        if state.get("version") != self.VERSION:
            raise ValueError("Prepared state version mismatch")
        entries = copy.deepcopy(state["entries"])
        ids = set()
        for e in entries.values():
            if e["state"] not in self.STATES:
                raise ValueError("Unknown prepared state")
            if e["code"] is not None:
                if source_hash(e["code"]) != e["code_sha256"]:
                    raise ValueError("Checkpoint source hash mismatch")
                if not e["program_id"] or e["program_id"] in ids:
                    raise ValueError("Duplicate or missing program identity")
                ids.add(e["program_id"])
            if e["state"] in ("ready", "executing", "completed", "deferred") and e["code"] is None:
                raise ValueError("Prepared state missing source")
            if e["state"] in ("executing", "completed") and e["fidelity"] not in (.2, 1.):
                raise ValueError("Execution missing fidelity")
        if sum(e["state"] in ("generating", "ready") for e in entries.values()) > 3:
            raise ValueError("Checkpoint exceeds prepared capacity")
        self.entries = entries
