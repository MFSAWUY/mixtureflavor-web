import csv
from dataclasses import replace
import io
import json
from pathlib import Path
import sqlite3
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import ROOT, submission
from mixtureflavor_web.api import create_app
from mixtureflavor_web.backup import backup_database, start_backup_worker
from mixtureflavor_web.chemistry import validate_rows
from mixtureflavor_web.models import ModelService
from mixtureflavor_web.settings import MAX_FILE_BYTES
from mixtureflavor_web.storage import Store
from restore_database import restore


def vector(scores):
    return np.array([x['score'] for x in scores])


def test_reference_fidelity_and_delta(client):
    response = client.post('/api/predictions', json=submission())
    assert response.status_code == 200
    result = response.json()
    fixture = json.loads((ROOT / 'vendor/flavorgnn/reference-predictions.json').read_text())
    assert result['dataset_kind'] == 'synthetic_benchmark'
    assert result['record_id'] and 'synthetic' in result['notice']['en'].lower()
    np.testing.assert_allclose([vector(x['scores']) for x in result['molecules']], fixture['single_scores'], atol=2e-5)
    mix = result['mixture']
    np.testing.assert_allclose(vector(mix['scores']), fixture['mixture_scores'], atol=2e-5)
    np.testing.assert_allclose(vector(mix['deltas']), vector(mix['scores']) - vector(mix['baseline_scores']), atol=1e-7)
    assert not np.allclose(vector(mix['deltas']), mix['raw_interaction_logits'])
    matrix = np.array(mix['attention_matrix'])
    assert matrix.shape == (2, 2) and np.isfinite(matrix).all()
    np.testing.assert_allclose(matrix.sum(axis=1), 1, atol=1e-5)
    assert 'not causal' in mix['attention_semantics']


def test_single_batch_five_components_and_permutation(client):
    for mode, values in [('single', ['CCO']), ('batch', ['CCO', 'CC(=O)O', 'c1ccccc1O'])]:
        result = client.post('/api/predictions', json=submission(values, mode)).json()
        assert result['mixture'] is None and result['mode'] == mode
        assert all(len(m['scores']) == 24 for m in result['molecules'])
    inputs = ['CCO', 'CC(=O)O', 'c1ccccc1O', 'CCOC(C)=O', 'CC(C)O']
    a = client.post('/api/predictions', json=submission(inputs)).json()
    b = client.post('/api/predictions', json=submission(inputs[::-1])).json()
    np.testing.assert_allclose(vector(a['mixture']['scores']), vector(b['mixture']['scores']), atol=2e-5)
    assert np.array(a['mixture']['attention_matrix']).shape == (5, 5)


def test_zero_bond_structure_and_100_molecule_batch(client):
    one = client.post('/api/predictions', json=submission(['[Na+]'], 'single'))
    assert one.status_code == 200 and len(one.json()['molecules'][0]['scores']) == 24
    hundred = client.post('/api/predictions', json=submission(['CCO'] * 100, 'batch'))
    assert hundred.status_code == 200 and len(hundred.json()['molecules']) == 100
    assert len(hundred.json()['duplicates']) == 99


@pytest.mark.parametrize('smiles,mode,code', [
    ([], 'batch', 'invalid_request'),
    (['CCO'] * 101, 'batch', 'invalid_request'),
    (['CCO'], 'mixture', 'mixture_count'),
    (['C', 'CC', 'CCC', 'CCCC', 'CCCCC', 'CCCCCC'], 'mixture', 'mixture_count'),
    (['CCO', 'OCC'], 'mixture', 'duplicate_components'),
    (['CCO', 'not-a-smiles'], 'mixture', 'invalid_records'),
    (['*'], 'single', 'invalid_records'),
    (['C' * 129], 'single', 'invalid_records'),
    (['C' * 100] * 42, 'batch', 'batch_too_large'),
    (['CCO', 'CC'], 'single', 'single_count'),
])
def test_rejected_predictions_are_recorded(client, auth, smiles, mode, code):
    response = client.post('/api/predictions', json=submission(smiles, mode))
    assert response.status_code == 422, response.text
    assert response.json()['error']['code'] == code
    record = client.get('/api/admin/records/' + response.json()['record_id'], auth=auth).json()
    assert record['status'] == 'failed' and record['result'] is None


def test_preview_record_positions_and_no_storage(client, auth):
    bad = client.post('/api/molecules/preview', json={'text': 'CCO\n\ninvalid\nCC(=O)O'})
    assert bad.status_code == 422
    assert bad.json()['error']['issues'][0]['record'] == 3
    good = client.post('/api/molecules/preview', json={'text': 'CCO\tEthanol\nOCC\tSame molecule'}).json()
    assert '<svg' in good['molecules'][0]['svg']
    assert good['molecules'][0]['formula'] == 'C2H6O'
    assert good['duplicates'] == [{'record': 2, 'same_as': 1}]
    assert client.get('/api/admin/records', auth=auth).json()['total'] == 0


@pytest.mark.parametrize('name,content,status', [
    ('good.csv', b'\xef\xbb\xbfname,smiles\nEthanol,CCO\nAcid,CC(=O)O\n', 200),
    ('bad.csv', b'name,other\nEthanol,CCO', 422),
    ('bad.csv', b'SMILES, smiles\nCCO,CC', 422),
    ('bad.csv', b'smiles\nCCO,unexpected', 422),
    ('bad.csv', b'smiles\nCCO\nnot-smiles', 422),
    ('bad.csv', b'smiles\n\xff', 422),
    ('bad.csv', b'', 422),
    ('bad.txt', b'CCO', 422),
    ('bad.sdf', b'invalid molecular record\n$$$$\n', 422),
])
def test_upload_validation(client, name, content, status):
    result = client.post('/api/molecules/upload', files={'file': (name, content)})
    assert result.status_code == status, result.text


def test_sdf_and_file_limit(client):
    content = (ROOT / 'frontend/public/example.sdf').read_bytes()
    result = client.post('/api/molecules/upload', files={'file': ('example.sdf', content)}).json()
    assert [m['canonical_smiles'] for m in result['molecules']] == ['CCO', 'CC(=O)O']
    too_large = client.post('/api/molecules/upload', files={'file': ('large.csv', b'x' * (MAX_FILE_BYTES + 1))})
    assert too_large.status_code == 413
    body_limit = client.post('/api/molecules/preview', content=b'x' * (MAX_FILE_BYTES + 65537))
    assert body_limit.status_code == 413


def test_missing_ack_and_sanitized_failure(client, auth, model, monkeypatch):
    payload = submission(['CCO'], 'single')
    payload['acknowledge_storage'] = False
    assert client.post('/api/predictions', json=payload).status_code == 422
    def fail(*_):
        raise RuntimeError('secret-internal-path-and-key')
    monkeypatch.setattr(model, 'predict', fail)
    failed = client.post('/api/predictions', json=submission())
    assert failed.status_code == 500 and 'secret-internal' not in failed.text
    record = client.get('/api/admin/records/' + failed.json()['record_id'], auth=auth).json()
    assert record['status'] == 'failed' and record['error']['code'] == 'prediction_failed'


def test_admin_export_delete_backup_restore_and_restart(client, settings, auth, model, tmp_path):
    result = client.post('/api/predictions', json=submission()).json()
    record_id = result['record_id']
    for method, path in [('get', '/records'), ('get', '/records/' + record_id), ('get', '/export'), ('delete', '/records/' + record_id), ('post', '/backup')]:
        assert getattr(client, method)('/api/admin' + path).status_code == 401
    assert client.get('/api/admin/records', auth=('admin', 'wrong')).status_code == 401
    assert client.get('/api/admin/records', auth=auth).json()['total'] == 1
    exported = client.get('/api/admin/export?format=json', auth=auth)
    assert exported.json()[0]['result']['record_id'] == record_id
    exported_csv = client.get('/api/admin/export?format=csv', auth=auth)
    rows = list(csv.DictReader(io.StringIO(exported_csv.text.lstrip('\ufeff'))))
    assert json.loads(rows[0]['result'])['dataset_kind'] == 'synthetic_benchmark'
    backup_name = client.post('/api/admin/backup', auth=auth).json()['backup']
    restored = tmp_path / 'restored.sqlite3'
    assert restore(settings.backup_dir / backup_name, restored) == 1
    with pytest.raises(FileExistsError):
        restore(settings.backup_dir / backup_name, restored)
    with TestClient(create_app(replace(settings, database=restored), lambda _: model)) as restarted:
        assert restarted.get('/api/admin/records/' + record_id, auth=auth).json()['result'] == result
    assert client.delete('/api/admin/records/' + record_id, auth=auth).status_code == 200
    assert client.get('/api/admin/records/' + record_id, auth=auth).status_code == 404
    assert Store(settings.database).list()['total'] == 0


def test_worker_backup_rotation_and_interrupted_status(settings):
    store = Store(settings.database)
    record = store.begin('single', {'smiles': 'CCO'}, 'test')
    assert Store(settings.database).get(record)['status'] == 'interrupted'
    for _ in range(4):
        backup_database(settings.database, settings.backup_dir, keep=2)
    assert len(list(settings.backup_dir.glob('*.sqlite3'))) == 2
    fast_settings = replace(settings, backup_hours=0.00001, backup_dir=settings.backup_dir / 'worker')
    stop, thread = start_backup_worker(fast_settings)
    try:
        deadline = time.monotonic() + 3
        while not list(fast_settings.backup_dir.glob('*.sqlite3')) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert list(fast_settings.backup_dir.glob('*.sqlite3'))
    finally:
        stop.set(); thread.join(timeout=2)


def test_missing_and_changed_assets_refuse_startup(settings, tmp_path):
    vendor = tmp_path / 'invalid-model'
    vendor.mkdir()
    manifest = {'sha256': {'missing.pt': '0' * 64}}
    (vendor / 'model-manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match='Missing or changed model asset'):
        with TestClient(create_app(replace(settings, vendor=vendor))):
            pass
    (vendor / 'missing.pt').write_bytes(b'invalid checkpoint')
    with pytest.raises(RuntimeError, match='Missing or changed model asset'):
        ModelService(replace(settings, vendor=vendor))


def test_incompatible_checkpoint_fails_strict_load(settings, monkeypatch):
    monkeypatch.setattr(ModelService, '_state', lambda *_: {})
    with pytest.raises(RuntimeError, match='Missing key'):
        ModelService(settings)


def test_feature_contract_rejects_wrong_rdkit(settings, monkeypatch):
    monkeypatch.setattr('mixtureflavor_web.models.rdBase.rdkitVersion', '0.0.0')
    with pytest.raises(RuntimeError, match='feature contract'):
        ModelService(settings)


def test_legacy_endpoints_and_metadata(client):
    response = client.post('/v1/predict', json={'smiles': 'CCO', 'top_k': 5, 'include_attention': True})
    assert response.status_code == 200
    assert len(response.json()['predictions'][0]['flavors']) <= 5
    assert len(response.json()['predictions'][0]['attention_weights']) == 3
    mix = client.post('/v2/predict-mixture', json={'mixture': ['CCO', 'CC(=O)O'], 'include_interaction_matrix': True}).json()
    assert len(mix['score_deltas']) == 24 and len(mix['interaction_matrix']) == 2
    assert client.get('/api/health').json()['status'] == 'ok'
    meta = client.get('/api/meta')
    assert meta.headers['cache-control'] == 'no-store'
    assert len(meta.json()['labels']) == 24
    assert meta.json()['metrics']['mixture']['macro_auroc'] == pytest.approx(0.9736348898555502)
    assert len(client.get('/api/examples').json()['examples']) == 3


def test_authentication_throttle(client, auth):
    for _ in range(20):
        assert client.get('/api/admin/records', auth=('admin', 'incorrect')).status_code == 401
    assert client.get('/api/admin/records', auth=auth).status_code == 429
