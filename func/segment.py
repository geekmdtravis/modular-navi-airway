#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides functions for segmenting airways from CT scans.

@author: Travis Nesbit, MD (tnesbi2@emory.edu, tnesbit7@gatech.edu)
"""

from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

from .model_arch import SegAirwayModel
from .model_run import semantic_segment_crop_and_cat
from .post_process import post_process
from .ulti import load_one_CT_img

# TODO: Investigate.
# Reviewing original code, crop and stride values provided by the pmutha
# implementation are not aligned with the original code. The cubu/stride are
# intended to be provided as a single integer and is turned into a a cube.
CROP_CUBE_SIZE = [32, 128, 128]
STRIDE = [16, 64, 64]
WINDOW_MIN = -1000
WINDOW_MAX = 600

# --- Model Initialization ---
_this_file = Path(__file__).resolve()
_checkpoint_dir = _this_file.parent.parent / "checkpoint"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
threshold = 0.5

model = SegAirwayModel(in_channels=1, out_channels=2)
model.to(device)
load_path = _checkpoint_dir / "checkpoint.pkl"
checkpoint = torch.load(load_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

model_ssl = SegAirwayModel(in_channels=1, out_channels=2)
model_ssl.to(device)
load_path = _checkpoint_dir / "checkpoint_semi_supervise_learning.pkl"
checkpoint = torch.load(load_path, map_location=device)
model_ssl.load_state_dict(checkpoint["model_state_dict"])
model_ssl.eval()


def _bbox2_3D(mask):
    # Computes a bounding box for the mask
    r = np.any(mask, axis=(1, 2))
    c = np.any(mask, axis=(0, 2))
    z = np.any(mask, axis=(0, 1))
    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]
    return rmin, rmax, cmin, cmax, zmin, zmax


def segment_airway(raw_img_path: str, lung_path: str, savepath: str):
    """
    Segments the airway from a single CT scan.
    """
    sitkim = sitk.ReadImage(raw_img_path)
    in_img = load_one_CT_img(raw_img_path)

    # The image is already segmented, so we don't need to do this.
    # ---- This is the original code for generating the lung mask ----
    # if not os.path.isfile(lung_path):
    #     print(f"  Lung mask not found at {lung_path}, generating...")
    #     inferer = LMInferer()
    #     segmentation = inferer.apply(sitkim)
    #     segmentation = np.uint8(segmentation > 0)
    #     lungmask = sitk.GetImageFromArray(segmentation)
    #     lungmask.CopyInformation(sitkim)
    #     sitk.WriteImage(lungmask, lung_path)
    #     print(f"  Lung mask saved to {lung_path}")
    # ---- End of original code ----

    lung_mask = load_one_CT_img(lung_path)
    rmin, rmax, cmin, cmax, zmin, zmax = _bbox2_3D(lung_mask)
    cropped_ct = in_img[rmin:rmax, cmin:cmax, zmin:zmax]

    seg_result_semi_supervise_learning = semantic_segment_crop_and_cat(
        cropped_ct,
        model_ssl,
        device,
        crop_cube_size=CROP_CUBE_SIZE,  # pyright: ignore
        stride=STRIDE,  # pyright: ignore
        windowMin=WINDOW_MIN,
        windowMax=WINDOW_MAX,
    )
    seg_onehot_semi_supervise_learning = np.array(
        seg_result_semi_supervise_learning > threshold, dtype=np.uint8
    )

    seg_result = semantic_segment_crop_and_cat(
        cropped_ct,
        model,
        device,
        crop_cube_size=CROP_CUBE_SIZE,  # pyright: ignore
        stride=STRIDE,  # pyright: ignore
        windowMin=WINDOW_MIN,
        windowMax=WINDOW_MAX,
    )
    seg_onehot = np.array(seg_result > threshold, dtype=np.uint8)

    seg_onehot_comb = np.array(
        (seg_onehot + seg_onehot_semi_supervise_learning) > 0, dtype=np.uint8
    )
    # Ignoring error for post_process, since borrowed from pmutha working.
    seg_processed, _ = post_process(seg_onehot_comb, threshold=threshold)  # pyright: ignore

    op = np.zeros_like(lung_mask)
    op[rmin:rmax, cmin:cmax, zmin:zmax] = seg_processed
    # Ignoring error for sitk.GetImageFromArray, since borrowed from pmutha working.
    zz = sitk.GetImageFromArray(np.uint8(op > 0))  # pyright: ignore
    zz.CopyInformation(sitkim)
    sitk.WriteImage(zz, savepath)


def run_segmentation(
    raw_ct_dir: str,
    lung_mask_dir: str,
    output_dir: str,
    filename: str,
    verbose: bool = False,
):
    """
    Runs airway segmentation on all patient folders in an input directory.
    Each patient folder is expected to contain a CT scan file.

    Args:
        raw_ct_path (str): Path to the raw CT scan file.
        lung_mask_path (str): Path to the lung mask file.
        output_path (str): Path to the output directory.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        None

    Raises:
        Exception: If the raw CT directory does not exist.
        Exception: If the lung mask directory does not exist.
    """
    _raw_ct_dir = Path(raw_ct_dir)
    if not _raw_ct_dir.exists():
        print(f"  Raw CT directory not found at {_raw_ct_dir}")
        return

    raw_ct_path = Path(raw_ct_dir)
    if not raw_ct_path.exists():
        print(f"  Raw CT file not found at {raw_ct_path}")
        return

    _lung_mask_dir = Path(lung_mask_dir)
    if not _lung_mask_dir.exists():
        print(f"  Lung mask directory not found at {_lung_mask_dir}")
        return

    lung_mask_path = Path(lung_mask_dir)
    if not lung_mask_path.exists():
        print(f"  Lung mask file not found at {lung_mask_path}")
        return

    _output_dir = Path(output_dir)
    _output_dir.mkdir(parents=True, exist_ok=True)
    _savepath = _output_dir / filename

    print("Starting segmentation process...")
    print(f"Input directory: {_raw_ct_dir.resolve()}")
    print(f"Output directory: {_output_dir.resolve()}")

    try:
        segment_airway(
            raw_img_path=str(raw_ct_path),
            lung_path=str(lung_mask_path),
            savepath=str(_savepath),
        )
        print(f"  Successfully processed {_savepath}")
    except Exception as e:
        print(f"  Error processing {_savepath}: {e}")
