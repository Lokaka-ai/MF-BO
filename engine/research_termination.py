"""Common v2 harness receipt; never infer scientific completion from exit code."""
import json
import os
from pathlib import Path
import threading
import time


class TerminationReceipt:
    def __init__(self, cfg):
        self.path = Path(os.environ["BO_TERMINATION_FILE"])
        self.lock = threading.RLock()
        self.data = {"version": 2, "status": "running", "steps": int(cfg.agent.steps),
                     "unexpected_errors": [], "exhaustion": [], "started": time.time()}
        self.save()

    def save(self):
        with self.lock:
            temp = self.path.with_suffix('.tmp')
            temp.write_text(json.dumps(self.data, indent=2) + '\n')
            temp.replace(self.path)

    def error(self, exc):
        with self.lock:
            self.data["unexpected_errors"].append(type(exc).__name__)
            self.save()

    def exhausted(self, reason):
        with self.lock:
            self.data["exhaustion"].append(str(reason))
            self.save()

    def search_finished(self, completed):
        self.data["completed_transitions"] = completed
        reasons = set(self.data["exhaustion"])
        legal_stop = completed >= self.data["steps"] or bool(reasons & {
            "Search step/time budget exhausted", "No legal normal, draft, or debug action remains"})
        self.data["search_finished_normally"] = legal_stop and not self.data["unexpected_errors"]
        self.save()
        return self.data["search_finished_normally"]

    def complete(self, *, no_valid=False):
        self.data.update(status="completed" if self.data.get("search_finished_normally") else "blocked",
                         algorithmic_no_valid=bool(no_valid), ended=time.time())
        self.save()

    def failed(self, exc):
        self.data.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "blocked",
                         exception_type=type(exc).__name__, ended=time.time())
        self.save()


def record_review_failure(agent, exc):
    """Common harness signal even when AQ03's optional rating is disabled."""
    if not os.environ.get("BO_TERMINATION_FILE") or isinstance(exc, (ValueError, TypeError, KeyError)):
        return  # malformed model content is not itself evidence of provider outage
    runtime = getattr(agent, "phase1_runtime", None)
    if runtime is not None:
        runtime.selection_logger._append("study7_failures.jsonl", {
            "status": "blocked/review_transport_failure", "exception_type": type(exc).__name__,
            "method_id": agent.cfg.study7.method_id})
