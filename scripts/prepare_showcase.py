"""Create public, reproducible examples without reading or writing a submission database."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.python-deps'), str(ROOT / 'backend')]
sys.dont_write_bytecode = True

from mixtureflavor_web.chemistry import validate_rows
from mixtureflavor_web.models import ModelService
from mixtureflavor_web.settings import Settings

STORAGE = {
    'zh': '本页展示预计算示例，不接收或保存用户分子提交。语言偏好仅保存在当前浏览器。',
    'en': 'This site displays precomputed examples and does not accept or store molecular submissions. Language preference is saved only in this browser.',
}
PROVENANCE = {
    'zh': '由随项目保存的模型权重预先计算；分数已与复制的原始预测器示例核对。页面切换不会运行模型。',
    'en': 'Precomputed with the model checkpoints included in the project and checked against the copied original predictors. Switching examples does not run inference.',
}


def prepare():
    service = ModelService(Settings())
    source = ROOT / 'examples/examples.json'
    source_examples = json.loads(source.read_text(encoding='utf-8'))['examples']
    examples = []
    for item in source_examples:
        preview = validate_rows(item['molecules'], item['mode'])
        result = service.predict(preview['molecules'], item['mode'])
        reference = item['reference_result']
        if item['mode'] == 'mixture':
            expected = {x['label']: x['probability'] for x in reference['mixture_prediction']['flavors']}
            actual = result['mixture']['scores']
            assert all(abs(s['score'] - expected[s['label']]) < 2e-5 for s in actual)
        else:
            for actual, ref in zip(result['molecules'], reference, strict=True):
                expected = {x['label']: x['probability'] for x in ref['flavors']}
                assert all(abs(s['score'] - expected[s['label']]) < 2e-5 for s in actual['scores'])
        result.pop('inference_ms')
        result.update(result_kind='precomputed_example', example_id=item['id'], provenance=PROVENANCE, storage_notice=STORAGE)
        examples.append({k: item[k] for k in ('id', 'name', 'description', 'mode')} | {'structures': preview['molecules'], 'result': result})
    meta = service.metadata()
    meta.pop('feature_contract')
    meta['storage_notice'] = STORAGE
    payload = {'schema_version': 1, 'meta': meta, 'provenance': PROVENANCE,
               'source_examples_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'examples': examples}
    target = ROOT / 'frontend/public/showcase-data.json'
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'file': str(target), 'examples': len(examples), 'model_version': service.version, 'database_access': False}))


if __name__ == '__main__':
    prepare()
