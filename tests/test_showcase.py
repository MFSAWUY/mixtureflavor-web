"""Public scores must retain agreement with the independent original predictors."""
from pathlib import Path
import hashlib
import json
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / 'frontend/public/showcase-data.json').read_text(encoding='utf-8'))
SOURCE = ROOT / 'examples/examples.json'
REFERENCES = json.loads(SOURCE.read_text(encoding='utf-8'))['examples']


@pytest.mark.parametrize('example,reference', zip(DATA['examples'], REFERENCES))
def test_precomputed_scores_match_original_predictors(example, reference):
    result = example['result']
    if example['mode'] == 'mixture':
        scores = result['mixture']['scores']
        expected = reference['reference_result']['mixture_prediction']['flavors']
        assert {s['label']: s['score'] for s in scores} == pytest.approx({s['label']: s['probability'] for s in expected}, abs=2e-5)
    else:
        for actual, expected in zip(result['molecules'], reference['reference_result'], strict=True):
            assert {s['label']: s['score'] for s in actual['scores']} == pytest.approx({s['label']: s['probability'] for s in expected['flavors']}, abs=2e-5)


def test_precomputed_asset_is_traceable_and_contains_no_submission_data():
    assert DATA['source_examples_sha256'] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert 'feature_contract' not in DATA['meta']
    text = json.dumps(DATA)
    for private_field in ['record_id', 'created_at', 'inference_ms', 'FLAVOR_ADMIN_PASSWORD', 'submissions.sqlite3']:
        assert private_field not in text
