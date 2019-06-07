import csv
import glob
import torch
import numpy as np
import nibabel as nib
from pathlib import Path

from src.data.datasets.base_dataset import BaseDataset
from src.data.transforms import compose


class KitsDataset(BaseDataset):
    """ 2019 MICCAI challenge (ref: https://kits19.grand-challenge.org/). The kits dataset for two stage methods.
    Args:
        data_split_csv (str): the csv path of the training / validation data split file
        task (str): the task name, i.e., `classification` or `segmentation`
        transforms (Box): the preprocessing and augmentation techiques applied to the data
    """
    def __init__(self, data_split_csv, task, transforms, **kwargs):
        super().__init__(**kwargs)
        self.data_split_csv = data_split_csv
        self.task = task
        self.image_transforms = compose(transforms)
        self.label_transforms = compose()
        self.image_path, self.label_path = [], []
        self.data = []

        # Collect the data paths according to the dataset split csv
        with open(self.data_split_csv, "r") as f:
            split_type = 'Training' if self.type == 'train' else 'Validation'
            rows = csv.reader(f)
            for row in rows:
                if row[1] == split_type:
                    self.image_path.append(self.data_root / row[0] / 'imaging.nii.gz')
                    self.label_path.append(self.data_root / row[0] / 'segmentation.nii.gz')


        # Build the image look up table
        for i in range(len(self.image_path)):
            metadata = nib.load(str(self.label_path[i]))
            img = metadata.get_data()
            num_slice = metadata.header.get_data_shape()[0]
            for j in range(num_slice):
                # Append the list: [# image_path, # slice]
                if self.task == 'clf':
                    self.data.append([i, j])
                elif self.task == 'seg':
                    # Need to verify whether the slice contains the kidney or the tumer
                    if np.sum(img[j]) != 0:
                        self.data.append([i, j])
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path_index, slice_index = self.data[index]
        image, label = nib.load(str(self.image_path[path_index])).get_data(), nib.load(str(self.label_path[path_index])).get_data()
        image, label = image[slice_index:slice_index+1].transpose((1, 2, 0)), label[slice_index:slice_index+1].transpose((1, 2, 0))

        if self.task == 'clf':
            label = np.array([np.sum(label) != 0])
            image = self.image_transforms(image, dtypes=[torch.float])
            label = self.label_transforms(label, dtypes=[torch.long])
            image = image.permute(2, 0, 1).contiguous()
        elif self.task == 'seg':
            image, label = self.image_transforms(image, label, normalize_tags=[True, False], dtypes=[torch.float, torch.long])
            image, label = image.permute(2, 0, 1).contiguous(), label.permute(2, 0, 1).contiguous()

        return {"image": image, "label": label}
