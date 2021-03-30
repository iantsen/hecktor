import argparse
import yaml
import os
import pathlib

import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

from utils import read_nifti, write_nifti, get_attributes, resample_sitk_image


def main(args):
    path_to_config = pathlib.Path(args.path)
    with open(path_to_config) as f:
        config = yaml.safe_load(f)

    path_to_input = pathlib.Path(config['path_to_input'])
    path_to_bb = pathlib.Path(config['path_to_bb'])
    path_to_output = pathlib.Path(config['path_to_output'])
    is_mask_available = config['is_mask_available']
    verbose = config['verbose']

    bb = pd.read_csv(path_to_bb)
    patients = list(bb.PatientID)
    print(f"Total number of patients: {len(patients)}")

    print(f"Resampled images will be saved in {path_to_output}")
    if not os.path.exists(path_to_output):
        os.makedirs(path_to_output, exist_ok=True)

    for p in tqdm(patients) if verbose else patients:
        # Read images:
        img_ct = read_nifti(path_to_input / p / (p + '_ct.nii.gz'))
        img_pt = read_nifti(path_to_input / p / (p + '_pt.nii.gz'))
        if is_mask_available:
            mask = read_nifti(path_to_input / p / (p + '_ct_gtvt.nii.gz'))

        # Get bounding boxes:
        pt1 = bb.loc[bb.PatientID == p, ['x1', 'y1', 'z1']]
        pt2 = bb.loc[bb.PatientID == p, ['x2', 'y2', 'z2']]
        pt1, pt2 = tuple(*pt1.values), tuple(*pt2.values)

        # Convert physcial points into array indexes:
        pt1_ct = img_ct.TransformPhysicalPointToIndex(pt1)
        pt1_pt = img_pt.TransformPhysicalPointToIndex(pt1)
        if is_mask_available:
            pt1_mask = mask.TransformPhysicalPointToIndex(pt1)

        pt2_ct = img_ct.TransformPhysicalPointToIndex(pt2)
        pt2_pt = img_pt.TransformPhysicalPointToIndex(pt2)
        if is_mask_available:
            pt2_mask = mask.TransformPhysicalPointToIndex(pt2)

        # Exctract the patch:
        cr_img_ct = img_ct[pt1_ct[0]: pt2_ct[0], pt1_ct[1]: pt2_ct[1], pt1_ct[2]: pt2_ct[2]]
        cr_img_pt = img_pt[pt1_pt[0]: pt2_pt[0], pt1_pt[1]: pt2_pt[1], pt1_pt[2]: pt2_pt[2]]
        if is_mask_available:
            cr_mask = mask[pt1_mask[0]: pt2_mask[0], pt1_mask[1]: pt2_mask[1], pt1_mask[2]: pt2_mask[2]]

        # Resample all images using CT attributes:
        # CT:
        cr_img_ct = resample_sitk_image(
            cr_img_ct,
            new_spacing=[1, 1, 1],
            new_size=[144, 144, 144],
            interpolator=sitk.sitkLinear)
        target_size = list(cr_img_ct.GetSize())
        attributes = get_attributes(cr_img_ct)

        # PT:
        cr_img_pt = resample_sitk_image(
            cr_img_pt,
            new_spacing=[1, 1, 1],
            new_size=target_size,
            attributes=attributes,
            interpolator=sitk.sitkLinear
        )

        # Mask:
        if is_mask_available:
            cr_mask = resample_sitk_image(
                cr_mask,
                new_spacing=[1, 1, 1],
                new_size=target_size,
                attributes=attributes,
                interpolator=sitk.sitkNearestNeighbor
            )

        # Save resampled images:
        if not os.path.exists(path_to_output / p):
            os.makedirs(path_to_output / p, exist_ok=True)

        write_nifti(cr_img_ct, path_to_output / p / (p + '_ct.nii.gz'))
        write_nifti(cr_img_pt, path_to_output / p / (p + '_pt.nii.gz'))
        if is_mask_available:
            write_nifti(cr_mask, path_to_output / p / (p + '_ct_gtvt.nii.gz'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Preprocessing Script')
    parser.add_argument("-p", "--path", type=str, required=True, help="path to the config file")
    args = parser.parse_args()
    main(args)
