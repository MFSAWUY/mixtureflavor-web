"""Verify the original project still matches its pre-implementation snapshot."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / "docs/source-integrity.json").read_text(encoding="utf-8"))
source = (ROOT / data["source"]).resolve()
changed = []
for item in data["files"]:
    p = source / item["path"]
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != item["sha256"]:
        changed.append(item["path"])
print(json.dumps({"source": str(source), "checked_files": len(data["files"]), "changed": changed}, ensure_ascii=False, indent=2))
raise SystemExit(1 if changed else 0)
