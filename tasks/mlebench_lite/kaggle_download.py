#!/usr/bin/env python3
"""Download only; isolated from the benchmark runtime and upstream preparation."""
import argparse
from importlib.metadata import version
import os
from pathlib import Path


def download(task, destination, token_file):
    for package, expected in (("kaggle", "2.2.5"), ("kagglesdk", "0.1.38")):
        if version(package) != expected:
            raise RuntimeError("Download client version mismatch")
    token = token_file.read_text().strip()
    if not token.startswith("KGAT_") or any(c.isspace() for c in token):
        raise ValueError("Expected one Kaggle access token")
    # Set before importing Kaggle, which can authenticate during import.
    os.environ["KAGGLE_API_TOKEN"] = token
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    api.competition_download_files(task, path=str(destination), quiet=True, unzip=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    args = parser.parse_args()
    try:
        download(args.task, args.destination, args.token_file)
    except Exception as exc:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
        detail = f"HTTP {status}" if status is not None else type(exc).__name__
        if status == 403:
            detail += " (accept competition rules with the token's account and check access)"
        print("KAGGLE_DOWNLOAD_ERROR: " + detail)
        raise SystemExit(1) from None
    print("KAGGLE_DOWNLOAD_OK")


if __name__ == "__main__":
    main()
