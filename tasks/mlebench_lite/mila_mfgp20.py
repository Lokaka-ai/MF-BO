#!/usr/bin/env python3
"""Separate MF-GP campaign. Never adds jobs to the existing two-mode campaign."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tasks.mlebench_lite import mila_multifidelity as campaign


def main():
    campaign.__doc__ = __doc__
    campaign.SPEC = ROOT / "config/mlebench_mfgp20.json"
    campaign.SCRIPT = Path(__file__).resolve()
    campaign.main(modes=("mf_gp",), default_campaign=ROOT / "runs/mf20_gp")


if __name__ == "__main__":
    main()
