import csv
import glob
import torch
import numpy as np
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
    def __init__(self, data_split_csv, transforms, **kwargs):
        super().__init__(**kwargs)
        self.data_split_csv = data_split_csv
        self.transforms = compose(transforms)
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
                self.data.append([i, j])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path_index, slice_index = self.data[index]
        image, label = nib.load(str(self.image_path[path_index])).get_data(), nib.load(str(self.label_path[path_index])).get_data()
        image, label = image[slice_index:slice_index+1].transpose((1, 2, 0)), label[slice_index:slice_index+1].transpose((1, 2, 0))
        image, label = self.transforms(image, label, normalize_tags=[True, False], dtypes=[torch.float, torch.long])
        image, label = image.permute(2, 0, 1).contiguous(), label.permute(2, 0, 1).contiguous()

        return {"image": image, "label": label}
