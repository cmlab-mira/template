import csv
import glob
import torch
import numpy as np
import nibabel as nib
from pathlib import Path

from src.data.datasets.base_dataset import BaseDataset
from src.data.transforms import compose


class KitsClfSegDataset(BaseDataset):
    """The Kidney Tumor Segmentation (KiTS) Challenge dataset (ref: https://kits19.grand-challenge.org/) for the two-staged method.
    Args:
        data_split_csv (str): The path of the training and validation data split csv file.
        task (str): The task name ('clf' or 'seg').
        transforms (Box): The preprocessing and augmentation techiques applied to the data.
    """
    def __init__(self, data_split_csv, task, transforms, **kwargs):
        super().__init__(**kwargs)
        self.data_split_csv = data_split_csv
        self.task = task
        self.image_transforms = compose(transforms) # for images and 'seg' labels
        self.label_transforms = compose() # for 'clf' labels
        self.data_paths = []

        # Collect the data paths according to the dataset_split_csv.
        with open(self.data_split_csv, "r") as f:
            type_ = 'Training' if self.type == 'train' else 'Validation'
            rows = csv.reader(f)
            for case_name, split_type in rows:
                if split_type == type_:
                    _image_paths = glob.glob(self.data_root / case_name / 'imaging*.nii.gz').sort()
                    _clf_label_paths = glob.glob(self.data_root / case_name / 'classification*.nii.gz').sort()
                    _seg_label_paths = glob.glob(self.data_root / case_name / 'segmentation*.nii.gz').sort()
                    if self.task == 'clf':
                        self.data_paths.extend([(image_path, clf_label_path) for image_path, clf_label_path in zip(_image_paths, _clf_label_paths)])
                    elif self.task == 'seg':
                        # Exclude the slice that does not contain the kidney or the tumer (foreground).
                        self.data_paths.extend([(image_path, seg_label_path) for image_path, clf_label_path, seg_label_path in zip(_image_paths, _clf_label_paths, _seg_label_paths)] if nib.load(str(clf_label_path)).get_data() != 0)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        image_path, label_path = self.data_paths[index]
        image, label = nib.load(str(image_path)).get_data(), nib.load(str(label_path)).get_data()
        if self.task == 'clf':
            image = self.image_transforms(image, dtypes=[torch.float])
            label = self.label_transforms(label, dtypes=[torch.long])
            image = image.permute(2, 0, 1).contiguous()
        elif self.task == 'seg':
            image, label = self.image_transforms(image, label, normalize_tags=[True, False], dtypes=[torch.float, torch.long])
            image, label = image.permute(2, 0, 1).contiguous(), label.permute(2, 0, 1).contiguous()
        return {"image": image, "label": label}
