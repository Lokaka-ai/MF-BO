"""Live native OpenCode checks; no ML training, synthetic files only."""
from pathlib import Path
import argparse
import json
import sys
import tempfile
import time
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from llm import FunctionSpec, generate, query
    from llm.opencode import DEFAULT_MODEL, DEFAULT_BASE_URL, events
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".cache/opencode-check.json")
    parser.add_argument("--work-dir", type=Path, help="Parent for synthetic files/session DBs; use a shared cluster path to check persistence there")
    args = parser.parse_args()
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="mlevolve-opencode-check-", dir=args.work_dir) as directory:
        work = Path(directory)
        stage = NS(model=DEFAULT_MODEL, base_url=DEFAULT_BASE_URL, api_key="")
        cfg = NS(workspace_dir=work, log_dir=work / "logs", agent=NS(code=stage, feedback=stage))
        text = generate("Reply with exactly OPENCODE_TEXT_OK.", cfg=cfg)
        assert text.strip() == "OPENCODE_TEXT_OK", text
        spec = FunctionSpec(name="check_provider", description="Confirm native model access.",
            json_schema={"type": "object", "properties": {
                "ok": {"const": True}, "message": {"const": "OPENCODE_MUSE_OK"}},
                "required": ["ok", "message"], "additionalProperties": False})
        response = query("Return ok=true and message=OPENCODE_MUSE_OK.", None,
                         model=DEFAULT_MODEL, func_spec=spec, cfg=cfg)
        assert response == {"ok": True, "message": "OPENCODE_MUSE_OK"}, response
        report = {"model": DEFAULT_MODEL, "provider": DEFAULT_BASE_URL, "text": text,
                  "response": response, "calls": [json.loads(line) for line in
                      (cfg.log_dir / "llm_calls.jsonl").read_text().splitlines()]}
        if (ROOT / "llm/opencode_session.py").is_file():
            from llm.codex_session import RoleExecutionPolicy
            from llm.opencode_session import OpenCodeSession
            candidate = work / "candidate"
            candidate.mkdir()
            (candidate / "solution.py").write_text("def answer():\n    return 0\n")
            nonce = "memory_" + str(time.time_ns())
            kwargs = dict(session_root=work / "sessions", role="experiment", context_id="smoke",
                policy=RoleExecutionPolicy.for_role("experiment", candidate_workspace=candidate),
                model=DEFAULT_MODEL, model_provider="opencode", reasoning_effort="high",
                prompt_path=work / "role.md", timeout_seconds=180)
            schema = {"type": "object", "properties": {"ok": {"const": True}}, "required": ["ok"]}
            system = "You implement a synthetic Python function in the current workspace. Return only the requested JSON."
            first = OpenCodeSession(**kwargs).run_turn(system_prompt=system,
                assignment_prompt=f"Remember the token {nonce}. Read solution.py, use the edit tool to change answer() to return 42, "
                    "and run a Python assertion that answer() == 42. Return {\"ok\":true}.", output_schema=schema)
            assert "return 42" in (candidate / "solution.py").read_text()
            second = OpenCodeSession(**kwargs).run_turn(system_prompt=system,
                assignment_prompt="Read solution.py and change answer() to return 43. Run a Python assertion to check it. "
                    "Return JSON with nonce equal to the token I told you in the previous turn and ok=true.",
                output_schema={"type": "object", "properties": {
                    "nonce": {"type": "string"}, "ok": {"const": True}}, "required": ["nonce", "ok"]})
            assert json.loads(second.message)["nonce"] == nonce  # Expected value is not sent in the second prompt/schema.
            assert second.resumed and second.thread_id == first.thread_id
            assert "return 43" in (candidate / "solution.py").read_text()
            tool_names = [event.get("part", {}).get("tool") for turn in (first, second)
                          for event in events(turn.stdout) if event.get("type") == "tool_use"
                          and event.get("part", {}).get("state", {}).get("status") == "completed"]
            assert "bash" in tool_names and "read" in tool_names and "edit" in tool_names, tool_names
            report["stateful"] = {"resumed": second.resumed, "session_id": second.thread_id,
                "tools": tool_names, "memory_verified": True, "file_edit_verified": True,
                "turns": [json.loads((turn.turn_path / "usage.json").read_text()) for turn in (first, second)]}
            readonly = OpenCodeSession(session_root=work / "sessions", role="rating", context_id="smoke",
                policy=RoleExecutionPolicy.for_role("rating", scratch_workspace=candidate),
                model=DEFAULT_MODEL, model_provider="opencode", reasoning_effort="high",
                prompt_path=work / "role.md", timeout_seconds=180).run_turn(
                    system_prompt="Inspect the supplied file and return JSON.",
                    assignment_prompt="Read solution.py using the read tool. Report answer as the integer the function returns.",
                    output_schema={"type": "object", "properties": {"answer": {"const": 43}}, "required": ["answer"]})
            report["readonly"] = {"response": json.loads(readonly.message),
                "usage": json.loads((readonly.turn_path / "usage.json").read_text())}
        report["work_dir"] = str(work.parent.resolve())
        report["elapsed_seconds"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
