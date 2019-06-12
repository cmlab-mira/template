import csv
import glob
import torch
import numpy as np
from box import Box
import nibabel as nib
from pathlib import Path

from src.data.datasets.base_dataset import BaseDataset
from src.data.transforms import compose


class KitsSegDataset(BaseDataset):
    """ 2019 MICCAI challenge (ref: https://kits19.grand-challenge.org/). The kits dataset for two stage methods.
    Args:
        data_split_csv (str): the csv path of the training / validation data split file
        transforms (Box): the preprocessing and augmentation techiques applied to the data
    """
    def __init__(self, data_split_csv, train_transforms, valid_transforms, **kwargs):
        super().__init__(**kwargs)
        self.data_split_csv = data_split_csv
        self.train_transforms = compose(train_transforms)
        self.valid_transforms = compose(valid_transforms)
        self.image_path, self.label_path = [], []
        self.data = []

        # Collect the data paths according to the dataset split csv
        with open(self.data_split_csv, "r") as f:
            split_type = 'Training' if self.type == 'train' else 'Validation'
            rows = csv.reader(f)
            for row in rows:
                if row[1] == split_type:
                    image_paths = [path for path in (self.data_root / row[0]).iterdir() \
                            if path.is_file() and 'imaging' in path.parts[-1]]
                    label_paths = [path for path in (self.data_root / row[0]).iterdir() \
                            if path.is_file() and 'segmentation' in path.parts[-1]]
                    self.image_path.extend(image_paths)
                    self.label_path.extend(label_paths)

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        image, label = nib.load(str(self.image_path[index])).get_data(), nib.load(str(self.label_path[index])).get_data()
        image, label = image.transpose((1, 2, 0)), label.transpose((1, 2, 0))
        image, label = image.astype(np.float64), label.astype(np.uint8)

        if self.type == 'train':
            image, label = self.train_transforms(image, label, normalize_tags=[True, False], dtypes=[torch.float, torch.long])
        elif self.type == 'valid':
            image, label = self.valid_transforms(image, label, normalize_tags=[True, False], dtypes=[torch.float, torch.long])

        image, label = image.permute(2, 0, 1).contiguous(), label.permute(2, 0, 1).contiguous()

        return {"image": image, "label": label}
