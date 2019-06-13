import logging
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path


def main(args):
    data_dir = args.data_dir
    output_dir = args.output_dir
    paths = [path for path in data_dir.iterdir() if path.is_dir()]

    for path in paths:
        logging.info(f'Process {path.parts[-1]}.')

        # Create output directory
        if not (output_dir / path.parts[-1]).is_dir():
            (output_dir / path.parts[-1]).mkdir(parents=True)

        # Read in the CT scans
        image = nib.load(str(path / 'imaging.nii.gz')).get_data().astype(np.float32)
        label = nib.load(str(path / 'segmentation.nii.gz')).get_data().astype(np.uint8)

        # Save each slice of the scan into single file
        for s in range(image.shape[0]):
            _image = image[s:s+1].transpose((1, 2, 0)) # (C, H, W) --> (H, W, C)
            nib.save(nib.Nifti1Image(_image, np.eye(4)), str(output_dir / path.parts[-1] / f'imaging_{s}.nii.gz'))

            # The label for segmentation task.
            _seg_label = label[s:s+1].transpose((1, 2, 0)) # (C, H, W) --> (H, W, C)
            nib.save(nib.Nifti1Image(_seg_label, np.eye(4)), str(output_dir / path.parts[-1] / f'segmentation_{s}.nii.gz'))

            # The label for classification task. If the slice has kidney or tumor (foreground), the label is set to 1, otherwise is 0.
            _clf_label = np.array(1) if np.count_nonzero(_seg_label) > 0 else np.array(0)
            np.save(output_dir / path.parts[-1] / f'classification_{s}.npy', _clf_label)


def _parse_args():
    parser = argparse.ArgumentParser(description="The data preprocessing.")
    parser.add_argument('data_dir', type=Path, help='The directory of the dataset.')
    parser.add_argument('output_dir', type=Path, help='The directory of the processed data.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    args = _parse_args()
    main(args)
