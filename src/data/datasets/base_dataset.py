import torch
from torch.utils.data import Dataset
from pathlib import Path


class BaseDataset(Dataset):
    """The base class for all datasets.
    Args:
        type (str): The type of the dataset including train, valid and test.
        data_root (str): The root directory of the saved data.
    """
    def __init__(self, type, data_root, transforms=None):
        super().__init__()
        self.type = type
        self.data_root = Path(data_root)
