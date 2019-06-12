import logging
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path


def main(args):
    data_path = args.data_path
    output_path = args.output_path
    paths = [path for path in data_path.iterdir() if path.is_dir()]

    for path in paths:
        logging.info(f'Process {path.parts[-1]}.')
        # Create output directory
        if not (output_path / path.parts[-1]).is_dir():
            (output_path / path.parts[-1]).mkdir(parents=True)

        # Read in the CT scans
        image = nib.load(str(path / 'imaging.nii.gz')).get_data()
        label = nib.load(str(path / 'segmentation.nii.gz')).get_data()
        # Save each slice of the scan into single file
        for s in range(image.shape[0]):
            _image = image[s:s+1]
            _image = nib.Nifti1Image(_image, np.eye(4))
            nib.save(_image, str(output_path / path.parts[-1] / f'imaging_{s}.nii.gz'))
            _label = label[s:s+1]
            _label = nib.Nifti1Image(_label, np.eye(4))
            nib.save(_label, str(output_path / path.parts[-1] / f'segmentation_{s}.nii.gz'))

def _parse_args():
    parser = argparse.ArgumentParser(description="The data preprocessing.")
    parser.add_argument('data_path', type=Path, help='The path of the dataset.')
    parser.add_argument('output_path', type=Path, help='The path of the processed data.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    args = _parse_args()
    main(args)
