#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides functions for segmenting airways from CT scans.

@author: Travis Nesbit, MD (tnesbi2@emory.edu, tnesbit7@gatech.edu)
"""

import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from lungmask import mask as LMInferer

from .model_arch import SegAirwayModel
from .model_run import semantic_segment_crop_and_cat
from .post_process import post_process
from .ulti import load_one_CT_img

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

model_semi_supervise_learning = SegAirwayModel(in_channels=1, out_channels=2)
model_semi_supervise_learning.to(device)
load_path = _checkpoint_dir / "checkpoint_semi_supervise_learning.pkl"
checkpoint = torch.load(load_path, map_location=device)
model_semi_supervise_learning.load_state_dict(checkpoint["model_state_dict"])
model_semi_supervise_learning.eval()


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

    if not os.path.isfile(lung_path):
        print(f"  Lung mask not found at {lung_path}, generating...")
        inferer = LMInferer()
        segmentation = inferer.apply(sitkim)
        segmentation = np.uint8(segmentation > 0)
        lungmask = sitk.GetImageFromArray(segmentation)
        lungmask.CopyInformation(sitkim)
        sitk.WriteImage(lungmask, lung_path)
        print(f"  Lung mask saved to {lung_path}")

    lmg = load_one_CT_img(lung_path)
    rmin, rmax, cmin, cmax, zmin, zmax = _bbox2_3D(lmg)
    raw_img = in_img[rmin:rmax, cmin:cmax, zmin:zmax]

    seg_result_semi_supervise_learning = semantic_segment_crop_and_cat(
        raw_img,
        model_semi_supervise_learning,
        device,
        crop_cube_size=[32, 128, 128],
        stride=[16, 64, 64],
        windowMin=-1000,
        windowMax=600,
    )
    seg_onehot_semi_supervise_learning = np.array(
        seg_result_semi_supervise_learning > threshold, dtype=np.uint8
    )

    seg_result = semantic_segment_crop_and_cat(
        raw_img,
        model,
        device,
        crop_cube_size=[32, 128, 128],
        stride=[16, 64, 64],
        windowMin=-1000,
        windowMax=600,
    )
    seg_onehot = np.array(seg_result > threshold, dtype=np.uint8)

    seg_onehot_comb = np.array(
        (seg_onehot + seg_onehot_semi_supervise_learning) > 0, dtype=np.uint8
    )
    seg_processed, _ = post_process(seg_onehot_comb, threshold=threshold)

    op = np.zeros_like(lmg)
    op[rmin:rmax, cmin:cmax, zmin:zmax] = seg_processed
    zz = sitk.GetImageFromArray(np.uint8(op > 0))
    zz.CopyInformation(sitkim)
    sitk.WriteImage(zz, savepath)


def run_segmentation(
    input_dir: str,
    output_dir: str,
    ct_filename: str,
    lung_mask_filename: str,
    output_filename: str,
):
    """
    Runs airway segmentation on all patient folders in an input directory.
    Each patient folder is expected to contain a CT scan file.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Starting segmentation process...")
    print(f"Input directory: {input_path.resolve()}")
    print(f"Output directory: {output_path.resolve()}")

    patient_folders = [f for f in input_path.iterdir() if f.is_dir()]
    if not patient_folders:
        print("No patient subdirectories found in the input directory.")
        # Check if the input_dir itself is a patient folder
        if (input_path / ct_filename).exists():
            print("Treating input directory as a single patient folder.")
            patient_folders = [input_path]
        else:
            return

    for patient_folder in patient_folders:
        print(f"Processing patient: {patient_folder.name}")

        ct_path = patient_folder / ct_filename
        if not ct_path.exists():
            print(f"  Skipping: CT file not found at {ct_path}")
            continue

        patient_output_dir = output_path / patient_folder.name
        patient_output_dir.mkdir(exist_ok=True)

        lung_path = patient_output_dir / lung_mask_filename
        airway_path = patient_output_dir / output_filename

        print(f"  Input CT: {ct_path}")
        print(f"  Lung Mask: {lung_path}")
        print(f"  Output Airway: {airway_path}")

        try:
            segment_airway(str(ct_path), str(lung_path), str(airway_path))
            print(f"  Successfully processed {patient_folder.name}")
        except Exception as e:
            print(f"  Error processing {patient_folder.name}: {e}")
