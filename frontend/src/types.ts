export type Lang = 'zh' | 'en'
export type Localized = { zh: string; en: string }
export type Mode = 'single' | 'batch' | 'mixture'
export type Score = { label: string; score: number }
export type Molecule = { id: string; name: string; smiles: string; canonical_smiles: string; formula: string; molecular_weight: number; atom_count: number; fragment_count: number; svg: string }
export type Preview = { molecules: Molecule[]; duplicates: { record: number; same_as: number }[]; count: number }
export type Prediction = {
  mode: Mode; record_id: string; created_at: string; model_version: string; dataset_kind: string; inference_ms: number; notice: Localized; storage_notice: Localized
  molecules: { id: string; name: string; smiles: string; canonical_smiles: string; scores: Score[] }[]
  mixture: null | { scores: Score[]; baseline_scores: Score[]; deltas: Score[]; attention_matrix: number[][]; baseline_semantics: string; attention_semantics: string }
}
export type Meta = { model_version: string; dataset_kind: string; notice: Localized; storage_notice: Localized; labels: { id: string; en: string; zh: string }[]; metrics: { single: Record<string, number>; mixture: Record<string, number> }; device: string; parameter_count: number }
export type Example = { id: string; name: Localized; description: Localized; mode: Mode; molecules: { name: string; smiles: string }[] }
export type RecordItem = { id: string; created_at: string; mode: string; status: string; model_version: string; input: unknown; error: unknown; result?: Prediction }
export type PrecomputedPrediction = Omit<Prediction, 'record_id' | 'created_at' | 'inference_ms'> & {
  result_kind: 'precomputed_example'; example_id: string; provenance: Localized
}
export type DisplayPrediction = Prediction | PrecomputedPrediction
export type ShowcaseExample = Omit<Example, 'molecules'> & { structures: Molecule[]; result: PrecomputedPrediction }
export type ShowcaseData = { schema_version: number; meta: Meta; provenance: Localized; source_examples_sha256: string; examples: ShowcaseExample[] }
