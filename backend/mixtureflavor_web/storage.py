"""Private prediction records; each operation owns its SQLite connection."""
from contextlib import contextmanager
from datetime import datetime, timezone
import csv
import io
import json
from pathlib import Path
import sqlite3
import uuid


def utcnow():
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS submissions (
                id TEXT PRIMARY KEY, created_at TEXT NOT NULL, mode TEXT NOT NULL,
                status TEXT NOT NULL, model_version TEXT NOT NULL, input_json TEXT NOT NULL,
                result_json TEXT, error_json TEXT, completed_at TEXT)""")
            db.execute("CREATE INDEX IF NOT EXISTS submissions_created ON submissions(created_at DESC)")
            # An interrupted worker must never leave a request looking as if it is still running forever.
            db.execute("UPDATE submissions SET status='interrupted', error_json=?, completed_at=? WHERE status='running'",
                       (json.dumps({"code": "service_restarted"}), utcnow()))

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=10000")
        db.execute("PRAGMA secure_delete=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def begin(self, mode, payload, version):
        record_id = str(uuid.uuid4())
        with self.connect() as db:
            db.execute("INSERT INTO submissions(id,created_at,mode,status,model_version,input_json) VALUES(?,?,?,?,?,?)",
                       (record_id, utcnow(), mode, "running", version, json.dumps(payload, ensure_ascii=False)))
        return record_id

    def finish(self, record_id, result=None, error=None):
        with self.connect() as db:
            db.execute("UPDATE submissions SET status=?,result_json=?,error_json=?,completed_at=? WHERE id=?",
                       ("failed" if error else "success", json.dumps(result, ensure_ascii=False) if result is not None else None,
                        json.dumps(error, ensure_ascii=False) if error is not None else None, utcnow(), record_id))

    @staticmethod
    def decode(row):
        if row is None:
            return None
        value = dict(row)
        for name in ["input", "result", "error"]:
            raw = value.pop(name + "_json")
            value[name] = json.loads(raw) if raw else None
        return value

    def list(self, limit=50, offset=0):
        with self.connect() as db:
            total = db.execute("SELECT COUNT(*) FROM submissions").fetchone()[0]
            rows = db.execute("SELECT * FROM submissions ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        records = []
        for row in rows:
            item = self.decode(row)
            # Results are fetched individually; the listing remains bounded for large batches.
            item.pop("result")
            records.append(item)
        return {"total": total, "records": records, "limit": limit, "offset": offset}

    def get(self, record_id):
        with self.connect() as db:
            return self.decode(db.execute("SELECT * FROM submissions WHERE id=?", (record_id,)).fetchone())

    def delete(self, record_id):
        with self.connect() as db:
            return db.execute("DELETE FROM submissions WHERE id=?", (record_id,)).rowcount > 0

    def export(self, format):
        # Read a consistent snapshot. Stream records so export does not grow with the entire database.
        with self.connect() as db:
            cursor = db.execute("SELECT * FROM submissions ORDER BY created_at DESC")
            if format == "json":
                yield "["
                first = True
                for row in cursor:
                    yield ("" if first else ",") + json.dumps(self.decode(row), ensure_ascii=False)
                    first = False
                yield "]"
            else:
                buffer = io.StringIO()
                writer = csv.writer(buffer)
                keys = ["id", "created_at", "mode", "status", "model_version", "input", "result", "error"]
                writer.writerow(keys)
                yield "\ufeff" + buffer.getvalue()
                for row in cursor:
                    buffer.seek(0); buffer.truncate(0)
                    item = self.decode(row)
                    writer.writerow([json.dumps(item[k], ensure_ascii=False) if k in {"input", "result", "error"} else item[k] for k in keys])
                    yield buffer.getvalue()
