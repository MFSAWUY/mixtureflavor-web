import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { predictionBlob } from '../src/api.ts'

const data = JSON.parse(readFileSync(new URL('../public/showcase-data.json', import.meta.url), 'utf8'))

test('public examples have stable IDs and no fabricated submission or timing fields', () => {
  assert.deepEqual(data.examples.map(x => x.id), ['single', 'mixture', 'batch'])
  for (const example of data.examples) {
    assert.equal(example.result.example_id, example.id)
    assert.equal(example.result.result_kind, 'precomputed_example')
    for (const key of ['record_id', 'created_at', 'inference_ms']) assert.equal(key in example.result, false)
    assert.equal(example.structures.length, example.result.molecules.length)
    for (const molecule of example.structures) assert.match(molecule.svg, /<svg/)
  }
})

test('all mixture score deltas match final minus baseline and attention has the correct dimension', () => {
  const { result } = data.examples.find(x => x.id === 'mixture')
  for (const score of result.mixture.scores) {
    const baseline = result.mixture.baseline_scores.find(x => x.label === score.label).score
    const delta = result.mixture.deltas.find(x => x.label === score.label).score
    assert.ok(Math.abs(score.score - baseline - delta) < 1e-7)
  }
  assert.equal(result.mixture.attention_matrix.length, 2)
  result.mixture.attention_matrix.forEach(row => assert.ok(Math.abs(row.reduce((a, b) => a + b) - 1) < 1e-6))
})

test('example JSON keeps provenance without introducing a fake user record', async () => {
  for (const { result } of data.examples) {
    const exported = JSON.parse(await predictionBlob(result, 'json').text())
    assert.deepEqual(exported, result)
    assert.equal(exported.record_id, undefined)
    assert.match(exported.provenance.en, /Precomputed/)
  }
})

test('example CSV labels its source and exports every score', async () => {
  for (const { result } of data.examples) {
    const csv = await predictionBlob(result, 'csv').text()
    assert.match(csv, /^"example_id","result_kind"/)
    assert.ok(!csv.includes('"record_id"'))
    assert.match(csv, /precomputed_example/)
    assert.match(csv, /Precomputed with the model checkpoints/)
    assert.equal(csv.split('\r\n').length, 1 + 24 * (result.molecules.length + (result.mixture ? 1 : 0)))
  }
})
