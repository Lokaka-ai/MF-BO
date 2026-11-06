"""Offline challenger on frozen v3 decisions; no counterfactual performance claim."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.mf_gp.v3.model import V3GaussianProcess
from engine.mf_gp.v3.mes import score_actions


def replay(source, output):
    with Path(output).open("x") as out:
        for line in Path(source).open():
            decision = json.loads(line)
            gp = V3GaussianProcess()
            gp.load_state_dict(decision["gp_state"])
            gp.fit(decision["observations"])
            if decision["terminal"]:
                result = dict(skipped="terminal_full_artifact_objective")
            else:
                scores, details = score_actions(gp, decision["targets"], decision["actions"], pending=decision["pending"])
                result = dict(scores=scores, chosen=max(range(len(scores)), key=scores.__getitem__), **details)
            out.write(json.dumps(dict(decision_id=decision["decision_id"], live_chosen=decision["chosen"],
                challenger=result, unknown_counterfactual_rewards=True), allow_nan=False)+"\n")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("decisions", type=Path)
    parser.add_argument("output", type=Path)
    args=parser.parse_args()
    replay(args.decisions, args.output)
