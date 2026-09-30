"""Restore a verified SQLite backup into a NEW file, never overwrite a live database."""
import argparse
from contextlib import closing
from pathlib import Path
import sqlite3


def restore(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects existing databases, including their WAL state.
    with destination.open('xb'):
        pass
    try:
        with closing(sqlite3.connect(source.as_uri() + '?mode=ro', uri=True)) as src:
            if src.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise RuntimeError('Source backup failed its integrity check')
            with closing(sqlite3.connect(destination)) as dst:
                src.backup(dst)
                if dst.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                    raise RuntimeError('Restored database failed its integrity check')
                count = dst.execute('SELECT COUNT(*) FROM submissions').fetchone()[0]
        return count
    except Exception:
        # Only the new file exclusively created by this invocation is removed.
        destination.unlink(missing_ok=True)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('backup', type=Path)
    parser.add_argument('destination', type=Path, help='New database path; must not exist')
    args = parser.parse_args()
    count = restore(args.backup, args.destination)
    print(f'Restored {count} records to {args.destination.resolve()}. Stop the service before switching FLAVOR_DATABASE to this file.')
