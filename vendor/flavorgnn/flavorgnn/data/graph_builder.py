"""SMILES to graph featurization."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any


ATOM_NUMBERS = {
    "H": 1,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Br": 35,
    "I": 53,
    "Na": 11,
    "K": 19,
    "Mg": 12,
    "Ca": 20,
    "Zn": 30,
    "Fe": 26,
}

HYBRIDIZATION_INDEX = {
    "SP": 0,
    "SP2": 1,
    "SP3": 2,
    "SP3D": 3,
    "SP3D2": 4,
    "OTHER": 5,
}

BOND_SYMBOL_TO_INDEX = {"-": 0, "=": 1, "#": 2, ":": 3}

STEREO_INDEX = {
    "STEREONONE": 0,
    "STEREOANY": 1,
    "STEREOZ": 2,
    "STEREOE": 3,
    "STEREOCIS": 4,
    "STEREOTRANS": 5,
}


@dataclass
class ParsedSmiles:
    atom_symbols: list[str]
    aromatic_flags: list[int]
    bonds: list[tuple[int, int, str, int]]
    ring_nodes: set[int]


def _one_hot(index: int, size: int) -> list[float]:
    out = [0.0] * size
    if 0 <= index < size:
        out[index] = 1.0
    return out


def _bounded_one_hot(value: int, max_value: int) -> list[float]:
    clamped = min(max(value, 0), max_value)
    return _one_hot(clamped, max_value + 1)


def _safe_ratio(a: float, b: float) -> float:
    return a / b if b else 0.0


def _extract_element(bracket_token: str) -> tuple[str, int]:
    match = re.search(r"([A-Z][a-z]?)", bracket_token)
    if not match:
        return "C", 0
    symbol = match.group(1)
    aromatic = 1 if symbol.islower() else 0
    return symbol.capitalize(), aromatic


def _tokenize_smiles(smiles: str) -> list[tuple[str, Any]]:
    tokens: list[tuple[str, Any]] = []
    i = 0
    while i < len(smiles):
        ch = smiles[i]
        if ch == "[":
            end = smiles.find("]", i + 1)
            if end == -1:
                break
            tokens.append(("atom_bracket", smiles[i + 1 : end]))
            i = end + 1
            continue

        if ch in "-=#:":
            tokens.append(("bond", ch))
            i += 1
            continue

        if ch in "()":
            tokens.append(("branch", ch))
            i += 1
            continue

        if ch.isdigit():
            tokens.append(("ring", ch))
            i += 1
            continue

        if ch in "/\\@":
            i += 1
            continue

        if ch.isalpha():
            if ch.isupper():
                if i + 1 < len(smiles) and smiles[i + 1].islower():
                    two = smiles[i : i + 2]
                    if two in {"Cl", "Br", "Na", "Mg", "Ca", "Zn", "Fe"}:
                        tokens.append(("atom", (two, 0)))
                        i += 2
                        continue
                tokens.append(("atom", (ch, 0)))
            else:
                tokens.append(("atom", (ch.upper(), 1)))
            i += 1
            continue

        i += 1

    return tokens


def parse_smiles_fallback(smiles: str) -> ParsedSmiles:
    tokens = _tokenize_smiles(smiles)
    atom_symbols: list[str] = []
    aromatic_flags: list[int] = []
    bonds: list[tuple[int, int, str, int]] = []
    ring_nodes: set[int] = set()

    branch_stack: list[int] = []
    ring_closures: dict[str, tuple[int, str]] = {}
    prev_idx: int | None = None
    pending_bond = "-"

    for token_type, payload in tokens:
        if token_type in {"atom", "atom_bracket"}:
            if token_type == "atom_bracket":
                symbol, aromatic = _extract_element(payload)
            else:
                symbol, aromatic = payload

            atom_symbols.append(symbol)
            aromatic_flags.append(aromatic)
            current_idx = len(atom_symbols) - 1

            if prev_idx is not None:
                bonds.append((prev_idx, current_idx, pending_bond, 0))
            prev_idx = current_idx
            pending_bond = "-"
            continue

        if token_type == "bond":
            pending_bond = payload
            continue

        if token_type == "branch":
            if payload == "(" and prev_idx is not None:
                branch_stack.append(prev_idx)
            elif payload == ")" and branch_stack:
                prev_idx = branch_stack.pop()
            continue

        if token_type == "ring" and prev_idx is not None:
            key = payload
            if key in ring_closures:
                start_idx, bond_symbol = ring_closures.pop(key)
                bond = pending_bond if pending_bond != "-" else bond_symbol
                bonds.append((start_idx, prev_idx, bond, 1))
                ring_nodes.add(start_idx)
                ring_nodes.add(prev_idx)
            else:
                ring_closures[key] = (prev_idx, pending_bond)
            pending_bond = "-"

    return ParsedSmiles(
        atom_symbols=atom_symbols,
        aromatic_flags=aromatic_flags,
        bonds=bonds,
        ring_nodes=ring_nodes,
    )


def _node_feature_from_fallback(
    atom_symbol: str,
    aromatic: int,
    degree: int,
    in_ring: int,
    max_atomic_number: int,
) -> list[float]:
    atomic_num = ATOM_NUMBERS.get(atom_symbol, 0)
    atomic_index = atomic_num if 1 <= atomic_num <= max_atomic_number else max_atomic_number
    atom_one_hot = _one_hot(atomic_index, max_atomic_number + 1)
    degree_one_hot = _bounded_one_hot(degree, 10)
    formal_charge = [0.0]
    hydrogen_one_hot = _one_hot(0, 5)
    hybridization = _one_hot(HYBRIDIZATION_INDEX["OTHER"], 6)
    aromaticity = [float(aromatic)]
    ring = [float(in_ring)]
    chirality = _one_hot(2, 3)
    partial_charge = [0.0]

    return (
        atom_one_hot
        + degree_one_hot
        + formal_charge
        + hydrogen_one_hot
        + hybridization
        + aromaticity
        + ring
        + chirality
        + partial_charge
    )


def _edge_feature_from_bond(bond_symbol: str, in_ring: int, stereo: str = "STEREONONE") -> list[float]:
    bond_type = [0.0] * 5
    index = BOND_SYMBOL_TO_INDEX.get(bond_symbol, 4)
    bond_type[index] = 1.0
    conjugated = [1.0 if bond_symbol in {"=", ":"} else 0.0]
    ring = [float(in_ring)]
    stereo_one_hot = _one_hot(STEREO_INDEX.get(stereo, 0), 6)
    bond_length = [0.0]
    return bond_type + conjugated + ring + stereo_one_hot + bond_length


def _global_features_from_fallback(parsed: ParsedSmiles) -> list[float]:
    atoms = parsed.atom_symbols
    atom_count = len(atoms)
    bond_count = len(parsed.bonds)
    aromatic_count = sum(parsed.aromatic_flags)
    hetero_count = sum(1 for atom in atoms if atom not in {"C", "H"})
    oxygen_count = atoms.count("O")
    nitrogen_count = atoms.count("N")
    sulfur_count = atoms.count("S")
    halogen_count = sum(1 for atom in atoms if atom in {"F", "Cl", "Br", "I"})
    ring_bonds = sum(b[-1] for b in parsed.bonds)
    mean_degree = _safe_ratio(2 * bond_count, max(atom_count, 1))
    smiles = "".join(atoms)

    has_acid = 1.0 if "COO" in smiles else 0.0
    has_amine = 1.0 if "N" in smiles else 0.0
    has_alcohol = 1.0 if "O" in smiles else 0.0
    double_ratio = _safe_ratio(sum(1 for _, _, b, _ in parsed.bonds if b == "="), max(bond_count, 1))
    triple_ratio = _safe_ratio(sum(1 for _, _, b, _ in parsed.bonds if b == "#"), max(bond_count, 1))

    return [
        atom_count / 150.0,
        bond_count / 200.0,
        _safe_ratio(aromatic_count, max(atom_count, 1)),
        _safe_ratio(hetero_count, max(atom_count, 1)),
        oxygen_count / 20.0,
        nitrogen_count / 20.0,
        sulfur_count / 20.0,
        halogen_count / 20.0,
        _safe_ratio(ring_bonds, max(bond_count, 1)),
        mean_degree / 4.0,
        0.0,
        has_acid,
        has_amine,
        has_alcohol,
        double_ratio,
        triple_ratio,
    ]


def _featurize_with_rdkit(smiles: str, max_atomic_number: int) -> dict[str, Any] | None:
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors
    except Exception:
        return None

    RDLogger.DisableLog("rdApp.error")
    mol = Chem.MolFromSmiles(smiles)
    RDLogger.EnableLog("rdApp.error")
    if mol is None:
        return None

    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        pass

    atom_features: list[list[float]] = []
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        atomic_idx = atomic_num if 1 <= atomic_num <= max_atomic_number else max_atomic_number
        atom_one_hot = _one_hot(atomic_idx, max_atomic_number + 1)
        degree_one_hot = _bounded_one_hot(atom.GetDegree(), 10)
        formal_charge = [max(min(atom.GetFormalCharge() / 4.0, 1.0), -1.0)]
        hydrogen_one_hot = _bounded_one_hot(atom.GetTotalNumHs(), 4)

        hyb_name = str(atom.GetHybridization())
        hyb_lookup = {
            "SP": "SP",
            "SP2": "SP2",
            "SP3": "SP3",
            "SP3D": "SP3D",
            "SP3D2": "SP3D2",
        }
        hyb = _one_hot(HYBRIDIZATION_INDEX.get(hyb_lookup.get(hyb_name, "OTHER"), 5), 6)
        aromatic = [float(atom.GetIsAromatic())]
        ring = [float(atom.IsInRing())]

        chiral_tag = str(atom.GetChiralTag())
        if "CCW" in chiral_tag:
            chirality = _one_hot(0, 3)
        elif "CW" in chiral_tag:
            chirality = _one_hot(1, 3)
        else:
            chirality = _one_hot(2, 3)

        charge_str = atom.GetProp("_GasteigerCharge") if atom.HasProp("_GasteigerCharge") else "0"
        try:
            partial_charge = float(charge_str)
            if math.isnan(partial_charge) or math.isinf(partial_charge):
                partial_charge = 0.0
        except Exception:
            partial_charge = 0.0

        atom_features.append(
            atom_one_hot
            + degree_one_hot
            + formal_charge
            + hydrogen_one_hot
            + hyb
            + aromatic
            + ring
            + chirality
            + [partial_charge]
        )

    edge_index: list[list[int]] = [[], []]
    edge_attr: list[list[float]] = []

    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        bond_type_name = str(bond.GetBondType())
        if bond_type_name == "SINGLE":
            bond_symbol = "-"
        elif bond_type_name == "DOUBLE":
            bond_symbol = "="
        elif bond_type_name == "TRIPLE":
            bond_symbol = "#"
        elif bond_type_name == "AROMATIC":
            bond_symbol = ":"
        else:
            bond_symbol = "?"

        stereo_key = str(bond.GetStereo())
        feat = _edge_feature_from_bond(
            bond_symbol=bond_symbol,
            in_ring=int(bond.IsInRing()),
            stereo=stereo_key if stereo_key in STEREO_INDEX else "STEREONONE",
        )

        edge_index[0].extend([begin, end])
        edge_index[1].extend([end, begin])
        edge_attr.extend([feat, feat])

    heavy_atoms = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    aromatic_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    hetero_atoms = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6))
    ring_count = rdMolDescriptors.CalcNumRings(mol)
    atom_count = mol.GetNumAtoms()
    bond_count = mol.GetNumBonds()

    global_features = [
        Descriptors.MolWt(mol) / 1000.0,
        Crippen.MolLogP(mol) / 10.0,
        rdMolDescriptors.CalcTPSA(mol) / 200.0,
        Lipinski.NumRotatableBonds(mol) / 20.0,
        Lipinski.NumHDonors(mol) / 10.0,
        Lipinski.NumHAcceptors(mol) / 20.0,
        rdMolDescriptors.CalcFractionCSP3(mol),
        heavy_atoms / 150.0,
        ring_count / 20.0,
        _safe_ratio(hetero_atoms, max(atom_count, 1)),
        _safe_ratio(aromatic_atoms, max(atom_count, 1)),
        1.0 if mol.HasSubstructMatch(Chem.MolFromSmarts("C(=O)O")) else 0.0,
        1.0 if mol.HasSubstructMatch(Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]")) else 0.0,
        1.0 if mol.HasSubstructMatch(Chem.MolFromSmarts("[OX2H]")) else 0.0,
        _safe_ratio(sum(1 for b in mol.GetBonds() if str(b.GetBondType()) == "DOUBLE"), max(bond_count, 1)),
        _safe_ratio(sum(1 for b in mol.GetBonds() if str(b.GetBondType()) == "TRIPLE"), max(bond_count, 1)),
    ]

    return {
        "x": atom_features,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "global_features": global_features,
    }


class SmilesGraphFeaturizer:
    """Converts SMILES strings into graph tensors.

    It uses RDKit when available and falls back to a pure-Python parser otherwise.
    """

    def __init__(self, use_rdkit_if_available: bool = True, max_atomic_number: int = 43) -> None:
        self.use_rdkit_if_available = use_rdkit_if_available
        self.max_atomic_number = max_atomic_number

    def featurize(self, smiles: str) -> dict[str, Any]:
        smiles = (smiles or "").strip()
        if not smiles:
            raise ValueError("SMILES is empty.")

        if self.use_rdkit_if_available:
            rdkit_graph = _featurize_with_rdkit(smiles, self.max_atomic_number)
            if rdkit_graph is not None:
                return rdkit_graph

        parsed = parse_smiles_fallback(smiles)
        if not parsed.atom_symbols:
            raise ValueError(f"Could not parse SMILES: {smiles}")

        num_atoms = len(parsed.atom_symbols)
        degrees = [0] * num_atoms
        for u, v, _, _ in parsed.bonds:
            degrees[u] += 1
            degrees[v] += 1

        node_in_ring = [0] * num_atoms
        for idx in parsed.ring_nodes:
            node_in_ring[idx] = 1

        x = [
            _node_feature_from_fallback(
                atom_symbol=parsed.atom_symbols[i],
                aromatic=parsed.aromatic_flags[i],
                degree=degrees[i],
                in_ring=node_in_ring[i],
                max_atomic_number=self.max_atomic_number,
            )
            for i in range(num_atoms)
        ]

        edge_index: list[list[int]] = [[], []]
        edge_attr: list[list[float]] = []
        for u, v, bond_symbol, in_ring in parsed.bonds:
            feat = _edge_feature_from_bond(bond_symbol=bond_symbol, in_ring=in_ring)
            edge_index[0].extend([u, v])
            edge_index[1].extend([v, u])
            edge_attr.extend([feat, feat])

        if not edge_attr:
            edge_index = [[0], [0]]
            edge_attr = [_edge_feature_from_bond("-", 0)]

        global_features = _global_features_from_fallback(parsed)
        return {
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "global_features": global_features,
        }
