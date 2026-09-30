"""Bounded, strict RDKit parsing. Never use the demo's permissive parser."""
import csv
from io import BytesIO, StringIO
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

from .schemas import InputProblem
from .settings import MAX_ATOMS, MAX_BATCH_ATOMS, MAX_FILE_BYTES, MAX_MOLECULES, MAX_SMILES_LENGTH


def parse_molecule(smiles: str, name: str = "", index: int = 1, draw=True):
    smiles = smiles.strip()
    if not smiles or len(smiles) > MAX_SMILES_LENGTH or any(c.isspace() for c in smiles):
        raise InputProblem("invalid_smiles", "SMILES 为空、过长或包含空白字符。", "SMILES is empty, too long, or contains whitespace.")
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None or not mol.GetNumAtoms() or any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
        raise InputProblem("invalid_smiles", "无法识别该分子结构，请检查 SMILES。", "Invalid molecular structure. Check the SMILES string.")
    if mol.GetNumAtoms() > MAX_ATOMS:
        raise InputProblem("molecule_too_large", f"每个分子最多 {MAX_ATOMS} 个原子。", f"At most {MAX_ATOMS} atoms are allowed per molecule.")
    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    # Redraw from SMILES, discarding uploaded SDF drawing/display properties.
    clean = Chem.MolFromSmiles(canonical)
    svg = ""
    if draw:
        drawer = rdMolDraw2D.MolDraw2DSVG(320, 190)
        drawer.drawOptions().clearBackground = False
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, clean)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
    return {
        "id": f"M{index:03d}", "name": (name.strip()[:120] or f"M{index:03d}"),
        "smiles": smiles, "canonical_smiles": canonical,
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "molecular_weight": round(Descriptors.MolWt(mol), 2),
        "atom_count": mol.GetNumAtoms(), "fragment_count": len(Chem.GetMolFrags(mol)), "svg": svg,
    }


def validate_rows(rows, mode=None, draw=True):
    if not rows:
        raise InputProblem("empty_input", "请至少输入一个分子。", "Enter at least one molecule.")
    if len(rows) > MAX_MOLECULES:
        raise InputProblem("too_many_molecules", "一次最多提交 100 个分子。", "Submit at most 100 molecules.")
    if mode == "single" and len(rows) != 1:
        raise InputProblem("single_count", "单分子模式只能提交一个分子。", "Single mode requires exactly one molecule.")
    if mode == "mixture" and not 2 <= len(rows) <= 5:
        raise InputProblem("mixture_count", "混合物需要 2–5 种不同分子。", "A mixture requires 2–5 distinct molecules.")
    molecules, issues, seen, duplicates = [], [], {}, []
    for i, row in enumerate(rows, 1):
        try:
            mol = parse_molecule(row.get("smiles", ""), row.get("name", ""), i, draw)
            if mol["canonical_smiles"] in seen:
                duplicates.append({"record": i, "same_as": seen[mol["canonical_smiles"]]})
            seen[mol["canonical_smiles"]] = i
            molecules.append(mol)
        except InputProblem as exc:
            issues.append({"record": row.get("record", i), "code": exc.code, "message": {"zh": exc.zh, "en": exc.en}})
    if issues:
        raise InputProblem("invalid_records", "部分分子无效，请修正后重新提交。", "Some records are invalid. Correct them before continuing.", issues)
    if mode == "mixture" and duplicates:
        raise InputProblem("duplicate_components", "混合物中存在重复分子；请移除重复项，重复次数不能表示配比。", "Remove duplicate components; repetition does not specify concentration.", duplicates)
    if sum(m["atom_count"] for m in molecules) > MAX_BATCH_ATOMS:
        raise InputProblem("batch_too_large", f"本次输入的原子总数不能超过 {MAX_BATCH_ATOMS}。", f"At most {MAX_BATCH_ATOMS} total atoms per request.")
    return {"molecules": molecules, "duplicates": duplicates, "count": len(molecules)}


def text_rows(text: str):
    # One SMILES per line; optional name after a tab. No guessing at chemical names.
    rows = []
    for line_no, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", 1)
        rows.append({"smiles": parts[0].strip(), "name": parts[1].strip() if len(parts) > 1 else "", "record": line_no})
        if len(rows) > MAX_MOLECULES:
            raise InputProblem("too_many_molecules", "一次最多提交 100 个分子。", "Submit at most 100 molecules.")
    return rows


def file_rows(filename: str, content: bytes):
    if len(content) > MAX_FILE_BYTES:
        raise InputProblem("file_too_large", "文件不得超过 5 MB。", "Files must not exceed 5 MB.", status=413)
    if not content.strip():
        raise InputProblem("empty_file", "上传文件为空。", "The uploaded file is empty.")
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        try:
            text = content.decode("utf-8-sig")
            reader = csv.DictReader(StringIO(text), strict=True)
            fields = reader.fieldnames or []
            mapping = {f.strip().lower(): f for f in fields if f}
            if "smiles" not in mapping or len(mapping) != len(fields):
                raise InputProblem("csv_header", "CSV 必须含唯一的 smiles 列，可选 name 列。", "CSV requires a unique smiles column; name is optional.")
            rows = []
            for row in reader:
                if None in row:
                    raise InputProblem("csv_row", "CSV 某行的列数与表头不一致。", "A CSV row does not match the header.")
                rows.append({"smiles": row.get(mapping["smiles"]) or "", "name": row.get(mapping.get("name")) or "", "record": reader.line_num})
                if len(rows) > MAX_MOLECULES:
                    raise InputProblem("too_many_molecules", "一次最多上传 100 个分子。", "Upload at most 100 molecules.")
            return rows
        except (UnicodeDecodeError, csv.Error) as exc:
            raise InputProblem("csv_encoding", "请使用 UTF-8 编码的有效 CSV 文件。", "Use a valid UTF-8 CSV file.") from exc
    if suffix == ".sdf":
        rows = []
        with rdBase.BlockLogs():
            supplier = Chem.ForwardSDMolSupplier(BytesIO(content), sanitize=True, removeHs=True, strictParsing=True)
            for i, mol in enumerate(supplier, 1):
                if i > MAX_MOLECULES:
                    raise InputProblem("too_many_molecules", "一次最多上传 100 个分子。", "Upload at most 100 molecules.")
                rows.append({"smiles": Chem.MolToSmiles(mol) if mol is not None else "", "name": mol.GetProp("_Name") if mol is not None and mol.HasProp("_Name") else "", "record": i})
        return rows
    raise InputProblem("file_type", "仅支持 CSV 或 SDF 文件。", "Only CSV and SDF files are supported.")
