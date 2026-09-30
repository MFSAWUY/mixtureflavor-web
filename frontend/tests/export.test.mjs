import { test } from 'node:test'
import assert from 'node:assert/strict'
import { predictionBlob } from '../src/api.ts'

const labels = ['sweet','bitter','sour','salty','umami','fruity','floral','spicy','herbal','fatty','smoky','nutty','caramel','vanilla','citrus','mint','woody','earthy','sulfurous','alcoholic','astringent','cooling','metallic','kokumi']
const scores = labels.map((label, i) => ({ label, score: i / 30 }))
const result = { record_id: 'test-record', created_at: '2026-09-29', model_version: 'test-version', dataset_kind: 'synthetic_benchmark', notice: { en: 'Synthetic demo; not measured intensities.', zh: '合成演示' }, molecules: [{ name: '=HYPERLINK("unsafe")', smiles: 'CCO', scores }, { name: 'Acid, sample', smiles: 'CC(=O)O', scores }], mixture: { scores, baseline_scores: scores.map(x => ({ ...x, score: x.score + 0.1 })), deltas: scores.map(x => ({ ...x, score: -0.1 })) } }

test('JSON preserves complete result, research notice and version', async () => {
  const blob = predictionBlob(result, 'json')
  assert.equal(blob.type, 'application/json')
  assert.deepEqual(JSON.parse(await blob.text()), result)
})

test('CSV includes all single and mixture scores with numeric signed deltas', async () => {
  const blob = predictionBlob(result, 'csv')
  const text = await blob.text()
  assert.equal(text.split('\r\n').length, 73)
  assert.match(text, /"baseline_score","score_delta","notice"/)
  assert.equal(text.match(/"mixture","Mixture"/g).length, 24)
  assert.equal(text.match(/"-0.1"/g).length, 24)
  assert.equal(text.match(/"synthetic_benchmark"/g).length, 72)
  assert.equal(text.match(/"test-version"/g).length, 72)
})

test('CSV escapes spreadsheet formulas, quotes and embedded commas in names', async () => {
  const text = await predictionBlob(result, 'csv').text()
  assert.ok(text.includes('"\'=HYPERLINK(""unsafe"")"'))
  assert.ok(text.includes('"Acid, sample"'))
  assert.ok(!text.includes('"\'-0.1"'))
})
