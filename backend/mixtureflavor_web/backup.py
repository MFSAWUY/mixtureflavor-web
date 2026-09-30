from datetime import datetime, timezone
import logging
from pathlib import Path
import sqlite3
import threading

log = logging.getLogger(__name__)


def backup_database(database: Path, directory: Path, keep=7):
    if not database.is_file():
        raise FileNotFoundError(database)
    directory.mkdir(parents=True, exist_ok=True)
    name = datetime.now(timezone.utc).strftime("submissions-%Y%m%dT%H%M%S%fZ.sqlite3")
    destination = directory / name
    temporary = destination.with_suffix(".tmp")
    source = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True, timeout=10)
    target = sqlite3.connect(temporary)
    try:
        source.backup(target)
        if target.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("Backup integrity check failed")
    finally:
        source.close(); target.close()
    temporary.replace(destination)
    for old in sorted(directory.glob("submissions-*.sqlite3"), reverse=True)[keep:]:
        old.unlink(missing_ok=True)
    return destination


def start_backup_worker(settings):
    stop = threading.Event()
    def work():
        while not stop.wait(settings.backup_hours * 3600):
            try:
                backup_database(settings.database.resolve(), settings.backup_dir, settings.backup_keep)
            except Exception:
                log.exception("Scheduled database backup failed")
    thread = threading.Thread(target=work, name="daily-sqlite-backup", daemon=True)
    thread.start()
    return stop, thread
