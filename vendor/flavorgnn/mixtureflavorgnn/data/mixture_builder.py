"""Synthetic mixture dataset builder."""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from flavorgnn.constants import FLAVOR_LABELS
from flavorgnn.data.split import create_splits, parse_labels


@dataclass(frozen=True)
class InteractionRule:
    name: str
    interaction_type: str
    required_counts: dict[str, int]
    delta: dict[str, float]
    focus_label: str
    focus_component_label: str
    focus_expected_change: int
    note: str


RULES: tuple[InteractionRule, ...] = (
    InteractionRule(
        name="umami_kokumi_boost",
        interaction_type="synergy",
        required_counts={"umami": 1, "kokumi": 1},
        delta={"umami": 1.20, "kokumi": 0.75},
        focus_label="umami",
        focus_component_label="kokumi",
        focus_expected_change=-1,
        note="Kokumi-rich peptides reinforce umami persistence.",
    ),
    InteractionRule(
        name="sweet_vanilla_rounding",
        interaction_type="synergy",
        required_counts={"sweet": 2, "vanilla": 1},
        delta={"sweet": 0.95, "vanilla": 0.45},
        focus_label="sweet",
        focus_component_label="vanilla",
        focus_expected_change=-1,
        note="Vanilla rounds and amplifies multi-sweet blends.",
    ),
    InteractionRule(
        name="citrus_fruity_lift",
        interaction_type="synergy",
        required_counts={"citrus": 1, "fruity": 1},
        delta={"citrus": 0.55, "fruity": 0.85},
        focus_label="fruity",
        focus_component_label="citrus",
        focus_expected_change=-1,
        note="Citrus esters brighten fruity perception.",
    ),
    InteractionRule(
        name="mint_cooling_amplification",
        interaction_type="synergy",
        required_counts={"mint": 1, "cooling": 1},
        delta={"mint": 0.40, "cooling": 1.00},
        focus_label="cooling",
        focus_component_label="mint",
        focus_expected_change=-1,
        note="Minty compounds amplify cooling sensations.",
    ),
    InteractionRule(
        name="woody_floral_bouquet",
        interaction_type="synergy",
        required_counts={"woody": 1, "floral": 1},
        delta={"floral": 0.45, "herbal": 0.65},
        focus_label="herbal",
        focus_component_label="floral",
        focus_expected_change=-1,
        note="Woody support increases bouquet complexity.",
    ),
    InteractionRule(
        name="sweet_masks_bitter",
        interaction_type="antagonism",
        required_counts={"sweet": 1, "bitter": 1},
        delta={"bitter": -1.00},
        focus_label="bitter",
        focus_component_label="sweet",
        focus_expected_change=1,
        note="Sweeteners suppress perceived bitterness.",
    ),
    InteractionRule(
        name="sulfur_suppresses_citrus",
        interaction_type="mixed",
        required_counts={"sulfurous": 1, "citrus": 1},
        delta={"citrus": -1.10, "fruity": -0.55, "metallic": 0.45},
        focus_label="citrus",
        focus_component_label="sulfurous",
        focus_expected_change=1,
        note="Sulfurous notes mute citrus and shift mixtures metallic.",
    ),
    InteractionRule(
        name="astringent_cuts_sweet",
        interaction_type="antagonism",
        required_counts={"astringent": 1, "sweet": 1},
        delta={"sweet": -0.85},
        focus_label="sweet",
        focus_component_label="astringent",
        focus_expected_change=1,
        note="Astringent components reduce sweet intensity.",
    ),
    InteractionRule(
        name="smoky_floral_conflict",
        interaction_type="mixed",
        required_counts={"smoky": 1, "floral": 1},
        delta={"floral": -0.70, "smoky": 0.35},
        focus_label="floral",
        focus_component_label="smoky",
        focus_expected_change=1,
        note="Smoky compounds suppress delicate floral notes.",
    ),
    InteractionRule(
        name="salty_dampens_sweet",
        interaction_type="antagonism",
        required_counts={"salty": 1, "sweet": 1},
        delta={"sweet": -0.55, "salty": 0.20},
        focus_label="sweet",
        focus_component_label="salty",
        focus_expected_change=1,
        note="Salt can flatten sweetness in simple blends.",
    ),
)


def _load_single_records(single_dataset_csv: str | Path) -> list[dict[str, object]]:
    frame = pd.read_csv(single_dataset_csv)
    records: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        labels = parse_labels(str(row["labels"]))
        records.append(
            {
                "molecule_id": str(row.get("molecule_id", "")),
                "smiles": str(row["smiles"]).strip(),
                "labels": labels,
                "label_set": set(labels),
            }
        )
    return records


def _label_to_indices(records: list[dict[str, object]]) -> dict[str, list[int]]:
    mapping = {label: [] for label in FLAVOR_LABELS}
    for idx, record in enumerate(records):
        for label in record["labels"]:
            mapping[label].append(idx)
    return mapping


def _applicable_rules(component_label_sets: list[set[str]]) -> list[InteractionRule]:
    counts = Counter(label for labels in component_label_sets for label in labels)
    active: list[InteractionRule] = []
    for rule in RULES:
        if all(counts.get(label, 0) >= count for label, count in rule.required_counts.items()):
            active.append(rule)
    return active


def _sample_index_for_label(
    label: str,
    label_index: dict[str, list[int]],
    used: set[int],
    rng: random.Random,
) -> int:
    candidates = [idx for idx in label_index[label] if idx not in used]
    if not candidates:
        candidates = list(label_index[label])
    if not candidates:
        raise ValueError(f"No molecules available for label {label}")
    choice = rng.choice(candidates)
    used.add(choice)
    return choice


def _target_mixture_size(rng: random.Random, min_components: int, max_components: int) -> int:
    sizes = [size for size in range(min_components, max_components + 1)]
    weights = [0.38, 0.34, 0.18, 0.10][: len(sizes)]
    weight_total = sum(weights)
    normalized = [w / weight_total for w in weights]
    return rng.choices(sizes, weights=normalized, k=1)[0]


def _finalize_labels(
    baseline_labels: set[str],
    interaction_targets: dict[str, float],
) -> list[str]:
    final = set(baseline_labels)
    for label, delta in interaction_targets.items():
        if delta >= 0.45:
            final.add(label)
        elif delta <= -0.55 and label in final:
            final.remove(label)
    return sorted(final)


def _choose_focus_component(smiles_records: list[dict[str, object]], focus_label: str) -> int:
    for idx, record in enumerate(smiles_records):
        if focus_label in record["label_set"]:
            return idx
    return 0


def _build_additive_sample(
    records: list[dict[str, object]],
    rng: random.Random,
    min_components: int,
    max_components: int,
) -> list[dict[str, object]]:
    for _ in range(200):
        size = _target_mixture_size(rng, min_components=min_components, max_components=max_components)
        picks = rng.sample(records, k=size)
        if not _applicable_rules([record["label_set"] for record in picks]):
            return picks
    return rng.sample(records, k=min_components)


def _build_rule_sample(
    records: list[dict[str, object]],
    label_index: dict[str, list[int]],
    rule: InteractionRule,
    rng: random.Random,
    min_components: int,
    max_components: int,
) -> list[dict[str, object]]:
    used: set[int] = set()
    chosen_indices: list[int] = []

    for label, count in rule.required_counts.items():
        for _ in range(count):
            chosen_indices.append(_sample_index_for_label(label, label_index, used, rng))

    target_size = max(_target_mixture_size(rng, min_components=min_components, max_components=max_components), len(chosen_indices))
    attempts = 0
    while len(chosen_indices) < target_size and attempts < 400:
        candidate_idx = rng.randrange(len(records))
        attempts += 1
        if candidate_idx in used:
            continue
        candidate_indices = chosen_indices + [candidate_idx]
        candidate_sets = [records[idx]["label_set"] for idx in candidate_indices]
        active = _applicable_rules(candidate_sets)
        if any(active_rule.name != rule.name for active_rule in active):
            continue
        used.add(candidate_idx)
        chosen_indices.append(candidate_idx)

    return [records[idx] for idx in chosen_indices]


def build_mixture_dataframe(
    single_dataset_csv: str | Path,
    num_mixtures: int = 4096,
    seed: int = 42,
    min_components: int = 2,
    max_components: int = 5,
    interaction_fraction: float = 0.65,
) -> pd.DataFrame:
    rng = random.Random(seed)
    records = _load_single_records(single_dataset_csv)
    label_index = _label_to_indices(records)

    rows: list[dict[str, object]] = []
    interaction_count = int(num_mixtures * interaction_fraction)
    additive_count = num_mixtures - interaction_count

    for sample_idx in range(interaction_count):
        rule = rng.choice(RULES)
        mixture = _build_rule_sample(
            records=records,
            label_index=label_index,
            rule=rule,
            rng=rng,
            min_components=min_components,
            max_components=max_components,
        )
        baseline_labels = set(label for record in mixture for label in record["label_set"])
        mixture_labels = _finalize_labels(baseline_labels, dict(rule.delta))
        focus_component_index = _choose_focus_component(mixture, rule.focus_component_label)

        rows.append(
            {
                "mixture_id": f"MIX_{sample_idx:05d}",
                "source": "interaction_rule",
                "components": json.dumps([record["smiles"] for record in mixture]),
                "component_labels": json.dumps([sorted(record["label_set"]) for record in mixture]),
                "labels": ";".join(mixture_labels),
                "mixture_labels": ";".join(mixture_labels),
                "baseline_labels": ";".join(sorted(baseline_labels)),
                "interaction_type": rule.interaction_type,
                "interaction_targets": json.dumps(rule.delta, sort_keys=True),
                "focus_component_index": focus_component_index,
                "focus_flavor": rule.focus_label,
                "focus_expected_change": rule.focus_expected_change,
                "notes": rule.note,
                "component_count": len(mixture),
            }
        )

    for additive_idx in range(additive_count):
        mixture = _build_additive_sample(
            records=records,
            rng=rng,
            min_components=min_components,
            max_components=max_components,
        )
        baseline_labels = sorted(set(label for record in mixture for label in record["label_set"]))
        row_idx = interaction_count + additive_idx
        rows.append(
            {
                "mixture_id": f"MIX_{row_idx:05d}",
                "source": "additive",
                "components": json.dumps([record["smiles"] for record in mixture]),
                "component_labels": json.dumps([sorted(record["label_set"]) for record in mixture]),
                "labels": ";".join(baseline_labels),
                "mixture_labels": ";".join(baseline_labels),
                "baseline_labels": ";".join(baseline_labels),
                "interaction_type": "additive",
                "interaction_targets": json.dumps({}, sort_keys=True),
                "focus_component_index": -1,
                "focus_flavor": "",
                "focus_expected_change": 0,
                "notes": "No explicit interaction rule applied.",
                "component_count": len(mixture),
            }
        )

    frame = pd.DataFrame(rows)
    frame = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return frame


def build_and_save_mixture_dataset(
    single_dataset_csv: str | Path,
    output_csv: str | Path,
    split_dir: str | Path,
    num_mixtures: int = 4096,
    seed: int = 42,
    min_components: int = 2,
    max_components: int = 5,
    interaction_fraction: float = 0.65,
    make_splits: bool = True,
) -> pd.DataFrame:
    frame = build_mixture_dataframe(
        single_dataset_csv=single_dataset_csv,
        num_mixtures=num_mixtures,
        seed=seed,
        min_components=min_components,
        max_components=max_components,
        interaction_fraction=interaction_fraction,
    )
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)

    if make_splits:
        create_splits(
            dataset_csv=output_path,
            output_dir=split_dir,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            random_seed=seed,
        )
    return frame
