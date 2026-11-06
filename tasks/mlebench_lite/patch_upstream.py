#!/usr/bin/env python3
"""Put MLE-bench's disposable SQLite cache on node-local storage."""
from pathlib import Path

root = Path(__file__).resolve().parents[2]
path = root / "mle-bench/mlebench/data.py"
original = 'cache = dc.Cache("cache", size_limit=2**26)  # 64 MB'
replacement = (
    'cache = dc.Cache(os.environ.get("MLEBENCH_CACHE_DIR", '
    'f"/tmp/mlebench-{os.getuid()}"), size_limit=2**26)  # 64 MB, node-local SQLite'
)
source = path.read_text()
if original in source:
    path.write_text(source.replace(original, replacement, 1))
elif replacement not in source:
    raise SystemExit("Unexpected upstream cache declaration; inspect before applying the cache patch")
print("MLE-bench SQLite cache uses node-local /tmp (or MLEBENCH_CACHE_DIR).")
