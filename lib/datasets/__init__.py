from lib.utils.config import CN

from ..utils.builder import build_dataset
from .mpm_synthetic import MPM_Synthetic
from .real_capture import Real_Capture


def create_dataset(cfg: CN, data_preset: CN):
    """
    Create a dataset instance.
    """
    return build_dataset(cfg, data_preset=data_preset)


def load_dataset(cfg: CN):
    """
    Create a dataset instance.
    """
    return build_dataset(cfg)