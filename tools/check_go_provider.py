"""Live text and structured-output checks using the campaign's Codex Go route."""
from pathlib import Path
import argparse
import json
import os
import sys
import tempfile
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from llm import FunctionSpec, generate, query
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if (args.model, args.provider) != ("muse-spark-1.3-contributor", "codex://opencode-go"):
        raise ValueError("Provider check must use the approved Muse Contributor Go route")
    if os.environ.get("MLEVOLVE_CODEX_REASONING_EFFORT") != "high":
        raise ValueError("Campaign requires high reasoning")
    with tempfile.TemporaryDirectory(prefix="bo-go-check-") as directory:
        work = Path(directory)
        stage = NS(model=args.model, base_url=args.provider, api_key="")
        cfg = NS(workspace_dir=work, log_dir=work / "logs",
                 agent=NS(code=stage, feedback=stage))
        text = generate("Reply with exactly OPENCODE_TEXT_OK.", cfg=cfg, max_retries=1)
        assert text.strip() == "OPENCODE_TEXT_OK", text
        schema = {"type": "object", "properties": {
            "ok": {"type": "boolean"}, "message": {"type": "string"}},
            "required": ["ok", "message"], "additionalProperties": False}
        spec = FunctionSpec(name="check_provider", description="Confirm model access.", json_schema=schema)
        response = query("Return ok=true and message=OPENCODE_MUSE_OK.", None,
                         model=args.model, func_spec=spec, cfg=cfg)
        assert response == {"ok": True, "message": "OPENCODE_MUSE_OK"}, response
        calls = [json.loads(line) for line in (cfg.log_dir / "llm_calls.jsonl").read_text().splitlines()]
        starts = [c for c in calls if c.get("event") == "started"]
        assert len(starts) == 2 and all(c["reasoning_effort"] == "high" and
            c["model"] == args.model and c["provider"] == "opencode-go" for c in starts)
        report = {"model": args.model, "provider": args.provider, "reasoning_effort": "high",
                  "text": text, "response": response, "calls": calls}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
