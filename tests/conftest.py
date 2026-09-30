from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.python-deps'), str(ROOT / 'backend'), str(ROOT / 'scripts')]

import pytest
from fastapi.testclient import TestClient
from mixtureflavor_web.api import create_app
from mixtureflavor_web.models import ModelService
from mixtureflavor_web.settings import Settings

PASSWORD = 'disposable-test-credentials-2026'


@pytest.fixture(scope='session')
def model():
    return ModelService(Settings(admin_password=PASSWORD))


@pytest.fixture
def settings(tmp_path):
    return Settings(database=tmp_path / 'records.sqlite3', backup_dir=tmp_path / 'backups', admin_password=PASSWORD)


@pytest.fixture
def client(settings, model):
    with TestClient(create_app(settings, lambda _: model)) as client:
        yield client


@pytest.fixture
def auth():
    return ('admin', PASSWORD)


def submission(smiles=('CCO', 'CC(=O)O'), mode='mixture'):
    return {'mode': mode, 'molecules': [{'smiles': s, 'name': f'Test {i + 1}'} for i, s in enumerate(smiles)], 'acknowledge_storage': True}
