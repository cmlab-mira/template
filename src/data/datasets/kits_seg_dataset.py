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
    """The Kidney Tumor Segmentation (KiTS) Challenge dataset (ref: https://kits19.grand-challenge.org/) for the 2D segmentation method (do not exclude the slices that does not contain the kidney or the tumer).
    Args:
        data_split_csv (str): The path of the training and validation data split csv file.
        train_transforms (Box): The preprocessing and augmentation techiques applied to the training data.
        valid_transforms (Box): The preprocessing and augmentation techiques applied to the validation data.
    """
    def __init__(self, data_split_csv, train_transforms, valid_transforms, **kwargs):
        super().__init__(**kwargs)
        self.data_split_csv = data_split_csv
        self.train_transforms = compose(train_transforms)
        self.valid_transforms = compose(valid_transforms)
        self.data_paths = []

        # Collect the data paths according to the dataset split csv.
        with open(self.data_split_csv, "r") as f:
            type_ = 'Training' if self.type == 'train' else 'Validation'
            rows = csv.reader(f)
            for case_name, split_type in rows:
                if split_type == type_:
                    _image_paths = sorted(list((self.data_root / case_name).glob('imaging*.nii.gz')))
                    _label_paths = sorted(list((self.data_root / case_name).glob('segmentation*.nii.gz')))
                    self.data_paths.extend([(image_path, label_path) for image_path, label_path in zip(_image_paths, _label_paths)])

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        image_path, label_path = self.data_paths[index]
        image, label = nib.load(str(image_path)).get_data(), nib.load(str(label_path)).get_data()
        if self.type == 'train':
            image, label = self.train_transforms(image, label, normalize_tags=[True, False], dtypes=[torch.float, torch.long])
        elif self.type == 'valid':
            image, label = self.valid_transforms(image, label, normalize_tags=[True, False], dtypes=[torch.float, torch.long])
        image, label = image.permute(2, 0, 1).contiguous(), label.permute(2, 0, 1).contiguous()
        return {"image": image, "label": label}
