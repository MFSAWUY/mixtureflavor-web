"""Data utilities."""

from flavorgnn.data.graph_builder import SmilesGraphFeaturizer
from flavorgnn.data.split import create_splits

__all__ = ["SmilesGraphFeaturizer", "create_splits"]

try:
    from flavorgnn.data.dataset import FlavorGraphDataset, collate_graph_batch

    __all__.extend(["FlavorGraphDataset", "collate_graph_batch"])
except Exception:
    # Allows non-torch utilities (e.g., data split generation) to run before torch is installed.
    pass
