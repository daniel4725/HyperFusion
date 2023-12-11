# -*- coding: utf-8 -*-
"""
intensity_normalization.normalize.fcm

use fuzzy c-means to find a mask for the white matter
given a T1w image and it's brain mask. Create a WM mask
from that T1w image's FCM WM mask. Then we can use that
WM mask as input to the func again, where the WM mask is
used to find an approximate mean of the WM intensity in
another target contrast, move it to some standard value.

Author: Blake Dewey (blake.dewey@jhu.edu),
        Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Apr 24, 2018


intensity_normalization.utilities.mask

create a tissue class mask of a target image

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: May 01, 2018
"""

from __future__ import print_function, division

import logging

import nibabel as nib

import logging
import warnings

import nibabel as nib
import numpy as np
from scipy.ndimage.morphology import (binary_closing, binary_fill_holes, generate_binary_structure, iterate_structure,
                                      binary_dilation)
from skfuzzy import cmeans
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture



def fcm_normalize(img, wm_mask, norm_value=1):
    """
    Use FCM generated mask to normalize the WM of a target image

    Args:
        img (nibabel.nifti1.Nifti1Image): target MR brain image
        wm_mask (nibabel.nifti1.Nifti1Image): white matter mask for img
        norm_value (float): value at which to place the WM mean

    Returns:
        normalized (nibabel.nifti1.Nifti1Image): img with WM mean at norm_value
    """

    img_data = img.get_data()
    wm_mask_data = wm_mask.get_data()
    wm_mean = img_data[wm_mask_data == 1].mean()
    normalized = nib.Nifti1Image((img_data / wm_mean) * norm_value,
                                 img.affine, img.header)
    return normalized


def find_wm_mask(img, brain_mask, threshold=0.8):
    """
    find WM mask using FCM with a membership threshold

    Args:
        img (nibabel.nifti1.Nifti1Image): target img
        brain_mask (nibabel.nifti1.Nifti1Image): brain mask for img
        threshold (float): membership threshold

    Returns:
        wm_mask (nibabel.nifti1.Nifti1Image): white matter mask for img
    """
    t1_mem = fcm_class_mask(img, brain_mask)
    wm_mask = t1_mem[..., 2] > threshold
    wm_mask_nifti = nib.Nifti1Image(wm_mask, img.affine, img.header)
    return wm_mask_nifti

def fcm_class_mask(img, brain_mask=None, hard_seg=False):
    """
    creates a mask of tissue classes for a target brain with fuzzy c-means

    Args:
        img (nibabel.nifti1.Nifti1Image): target image (must be T1w)
        brain_mask (nibabel.nifti1.Nifti1Image): mask covering the brain of img
            (none if already skull-stripped)
        hard_seg (bool): pick the maximum membership as the true class in output

    Returns:
        mask (np.ndarray): membership values for each of three classes in the image
            (or class determinations w/ hard_seg)
    """
    img_data = img.get_data()
    if brain_mask is not None:
        mask_data = brain_mask.get_data() > 0
    else:
        mask_data = img_data > img_data.mean()
    [t1_cntr, t1_mem, _, _, _, _, _] = cmeans(img_data[mask_data].reshape(-1, len(mask_data[mask_data])),
                                              3, 2, 0.005, 50)
    t1_mem_list = [t1_mem[i] for i, _ in sorted(enumerate(t1_cntr), key=lambda x: x[1])]  # CSF/GM/WM
    mask = np.zeros(img_data.shape + (3,))
    for i in range(3):
        mask[..., i][mask_data] = t1_mem_list[i]
    if hard_seg:
        tmp_mask = np.zeros(img_data.shape)
        tmp_mask[mask_data] = np.argmax(mask[mask_data], axis=1) + 1
        mask = tmp_mask
    return mask