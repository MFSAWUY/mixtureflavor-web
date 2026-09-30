"""Run acceptance tests using isolated dependencies and disposable local databases."""
from pathlib import Path
import os
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.python-deps'), str(ROOT / 'backend'), str(ROOT / 'scripts')]
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.chdir(ROOT)
import pytest

if __name__ == '__main__':
    temporary = ROOT / 'runtime' / ('test-' + uuid.uuid4().hex[:10])
    raise SystemExit(pytest.main(['tests', '-q', '--basetemp', str(temporary), '-o', 'cache_dir=runtime/pytest-cache', *sys.argv[1:]]))
