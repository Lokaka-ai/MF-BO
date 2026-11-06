"""Record or verify the exact prepared corpus and tokenizer."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]


def checksum(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    cache = Path(os.environ.get("AUTORESEARCH_CACHE_DIR", ROOT / ".cache" / "autoresearch")).expanduser().resolve()
    required = [f"data/shard_{i:05d}.parquet" for i in [*range(10), 6542]]
    required += ["tokenizer/tokenizer.pkl", "tokenizer/token_bytes.pt"]
    actual_shards = sorted(p.name for p in (cache / "data").glob("*.parquet"))
    if actual_shards != sorted(Path(p).name for p in required if p.endswith(".parquet")):
        raise ValueError("This preset requires exactly training shards 00000–00009 and validation shard 06542")
    files = []
    for name in required:
        path = cache / name
        entry = {"path": name, "bytes": path.stat().st_size, "sha256": checksum(path)}
        if path.suffix == ".parquet":
            parquet = pq.ParquetFile(path)
            if "text" not in parquet.schema.names or parquet.metadata.num_rows < 1:
                raise ValueError(f"Invalid corpus shard: {path}")
            entry["rows"] = parquet.metadata.num_rows
        files.append(entry)
    manifest = {"dataset": "karpathy/climbmix-400b-shuffle", "train_shards": list(range(10)),
                "validation_shard": 6542, "vocab_size": 8192, "files": files}
    target = cache / "manifest.json"
    if args.verify:
        if json.loads(target.read_text()) != manifest:
            raise ValueError("Prepared data/tokenizer differs from the recorded manifest")
        print(f"Verified all {len(files)} data/tokenizer SHA256 hashes")
    else:
        if target.exists() and json.loads(target.read_text()) != manifest:
            raise ValueError("Existing manifest differs; use a new cache directory for a new dataset")
        target.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Recorded {len(files)} verified files in {target}")


if __name__ == "__main__":
    main()
