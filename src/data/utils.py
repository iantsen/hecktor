import os
import pathlib
import json
import numpy as np
import SimpleITK as sitk
import torch
from torch.nn import functional as F


def get_paths_to_patient_files(path_to_imgs, append_mask=True):
    """
    Get paths to all data samples, i.e., CT & PET images (and a mask) for each patient.

    Parameters
    ----------
    path_to_imgs : str
        A path to a directory with patients' data. Each folder in the directory must corresponds to a single patient.
    append_mask : bool
        Used to append a path to a ground truth mask.

    Returns
    -------
    list of tuple
        A list wherein each element is a tuple with two (three) `pathlib.Path` objects for a single patient.
        The first one is the path to the CT image, the second one - to the PET image. If `append_mask` is True,
        the path to the ground truth mask is added.
    """
    path_to_imgs = pathlib.Path(path_to_imgs)

    patients = [p for p in os.listdir(path_to_imgs) if os.path.isdir(path_to_imgs / p)]
    paths = []
    for p in patients:
        path_to_ct = path_to_imgs / p / (p + '_ct.nii.gz')
        path_to_pt = path_to_imgs / p / (p + '_pt.nii.gz')

        if append_mask:
            path_to_mask = path_to_imgs / p / (p + '_ct_gtvt.nii.gz')
            paths.append((path_to_ct, path_to_pt, path_to_mask))
        else:
            paths.append((path_to_ct, path_to_pt))
    return paths


def get_train_val_paths(all_paths, path_to_train_val_pkl):
    """"
    Split a list of all paths to patients' data into train & validation parts using patients' IDs.

    Parameters
    ----------
    all_paths: list
        An output of `get_paths_to_patient_files`.
    path_to_train_val_pkl: str
        A path to a pkl file storing train & validation IDs.

    Returns
    -------
    (list, list)
        Two lists of paths to train & validation data samples.
    """
    path_to_train_val_pkl = pathlib.Path(path_to_train_val_pkl)
    with open(path_to_train_val_pkl) as f:
        train_val_split = json.load(f)

    train_paths = [path for path in all_paths
                   if any(patient_id + '_ct.nii.gz' in str(path[0]) for patient_id in train_val_split['train'])]

    val_paths = [path for path in all_paths
                 if any(patient_id + '_ct.nii.gz' in str(path[0]) for patient_id in train_val_split['val'])]

    return train_paths, val_paths


def read_nifti(path):
    """Read a NIfTI image. Return a SimpleITK Image."""
    nifti = sitk.ReadImage(str(path))
    return nifti


def write_nifti(sitk_img, path):
    """Save a SimpleITK Image to disk in NIfTI format."""
    writer = sitk.ImageFileWriter()
    writer.SetImageIO("NiftiImageIO")
    writer.SetFileName(str(path))
    writer.Execute(sitk_img)


def get_attributes(sitk_image):
    """Get physical space attributes (meta-data) of the image."""
    attributes = {}
    attributes['orig_pixelid'] = sitk_image.GetPixelIDValue()
    attributes['orig_origin'] = sitk_image.GetOrigin()
    attributes['orig_direction'] = sitk_image.GetDirection()
    attributes['orig_spacing'] = np.array(sitk_image.GetSpacing())
    attributes['orig_size'] = np.array(sitk_image.GetSize(), dtype=np.int)
    return attributes


def resample_sitk_image(sitk_image,
                        new_spacing=[1, 1, 1],
                        new_size=None,
                        attributes=None,
                        interpolator=sitk.sitkLinear,
                        fill_value=0):
    """
    Resample a SimpleITK Image.

    Parameters
    ----------
    sitk_image : sitk.Image
        An input image.
    new_spacing : list of int
        A distance between adjacent voxels in each dimension given in physical units (mm) for the output image.
    new_size : list of int or None
        A number of pixels per dimension of the output image. If None, `new_size` is computed based on the original
        input size, original spacing and new spacing.
    attributes : dict or None
        The desired output image's spatial domain (its meta-data). If None, the original image's meta-data is used.
    interpolator
        Available interpolators:
            - sitk.sitkNearestNeighbor : nearest
            - sitk.sitkLinear : linear
            - sitk.sitkGaussian : gaussian
            - sitk.sitkLabelGaussian : label_gaussian
            - sitk.sitkBSpline : bspline
            - sitk.sitkHammingWindowedSinc : hamming_sinc
            - sitk.sitkCosineWindowedSinc : cosine_windowed_sinc
            - sitk.sitkWelchWindowedSinc : welch_windowed_sinc
            - sitk.sitkLanczosWindowedSinc : lanczos_windowed_sinc
    fill_value : int or float
        A value used for padding, if the output image size is less than `new_size`.

    Returns
    -------
    sitk.Image
        The resampled image.

    Notes
    -----
    This implementation is based on https://github.com/deepmedic/SimpleITK-examples/blob/master/examples/resample_isotropically.py
    """
    sitk_interpolator = interpolator

    # provided attributes:
    if attributes:
        orig_pixelid = attributes['orig_pixelid']
        orig_origin = attributes['orig_origin']
        orig_direction = attributes['orig_direction']
        orig_spacing = attributes['orig_spacing']
        orig_size = attributes['orig_size']

    else:
        # use original attributes:
        orig_pixelid = sitk_image.GetPixelIDValue()
        orig_origin = sitk_image.GetOrigin()
        orig_direction = sitk_image.GetDirection()
        orig_spacing = np.array(sitk_image.GetSpacing())
        orig_size = np.array(sitk_image.GetSize(), dtype=np.int)

    # new image size:
    if not new_size:
        new_size = orig_size * (orig_spacing / new_spacing)
        new_size = np.ceil(new_size).astype(np.int)  # Image dimensions are in integers
        new_size = [int(s) for s in new_size]  # SimpleITK expects lists, not ndarrays

    resample_filter = sitk.ResampleImageFilter()
    resampled_sitk_image = resample_filter.Execute(sitk_image,
                                                   new_size,
                                                   sitk.Transform(),
                                                   sitk_interpolator,
                                                   orig_origin,
                                                   new_spacing,
                                                   orig_direction,
                                                   fill_value,
                                                   orig_pixelid)

    return resampled_sitk_image
