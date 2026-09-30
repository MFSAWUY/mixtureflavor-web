from dataclasses import dataclass
import os
from pathlib import Path

from . import ROOT, VENDOR

MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_MOLECULES = 100
MAX_ATOMS = 128
MAX_BATCH_ATOMS = 4096
MAX_SMILES_LENGTH = 2000
DATASET_KIND = "synthetic_benchmark"
NOTICE = {
    "zh": "合成数据演示模型。分数并非实测风味强度；增强/减弱为相对模型内部基线的预测变化。注意力不代表因果作用或受体竞争。",
    "en": "Synthetic-benchmark demo. Scores are not measured flavor intensities. Changes are relative to the model's internal baseline. Attention is not evidence of causality or receptor competition.",
}
STORAGE_NOTICE = {
    "zh": "提交预测即表示知悉：输入分子、结果、时间、模型版本和运行状态会保存，用于课题使用分析及结果复现；不公开，保留至管理员主动删除。结构预览不保存。",
    "en": "Prediction submissions (molecules, results, time, model version and status) are stored privately for research usage analysis and reproducibility until deleted by the administrator. Structure previews are not stored.",
}


@dataclass(frozen=True)
class Settings:
    database: Path = ROOT / "runtime" / "submissions.sqlite3"
    backup_dir: Path = ROOT / "runtime" / "backups"
    vendor: Path = VENDOR
    admin_user: str = "admin"
    admin_password: str = ""
    threads: int = 2
    backup_hours: float = 24
    backup_keep: int = 7

    @classmethod
    def from_env(cls):
        return cls(
            database=Path(os.getenv("FLAVOR_DATABASE", str(ROOT / "runtime" / "submissions.sqlite3"))),
            backup_dir=Path(os.getenv("FLAVOR_BACKUP_DIR", str(ROOT / "runtime" / "backups"))),
            vendor=Path(os.getenv("FLAVOR_VENDOR", str(VENDOR))),
            admin_user=os.getenv("FLAVOR_ADMIN_USER", "admin"),
            admin_password=os.getenv("FLAVOR_ADMIN_PASSWORD", ""),
            threads=int(os.getenv("FLAVOR_THREADS", "2")),
            backup_hours=float(os.getenv("FLAVOR_BACKUP_HOURS", "24")),
            backup_keep=int(os.getenv("FLAVOR_BACKUP_KEEP", "7")),
        )

    def validate(self):
        if len(self.admin_password) < 16:
            raise RuntimeError("Set FLAVOR_ADMIN_PASSWORD to at least 16 characters; no default admin password is allowed.")
        if not 1 <= self.threads <= 16:
            raise RuntimeError("FLAVOR_THREADS must be between 1 and 16.")
        if self.backup_hours <= 0 or self.backup_keep < 1:
            raise RuntimeError("Backup interval and retention must be positive.")
