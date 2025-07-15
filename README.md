# NaviAirway
Travis Nesbit, MD (travis@geekmd.io, tnesbit7@gatech.edu, tnesbi2@emory.edu)

This repository is an adaptation of the original NaviAirway project, refactored for ease of use as a modular library that can easily be incorporated into a larger data processing pipeline.

## Original Work

> [**NaviAirway: a Bronchiole-sensitive Deep Learning-based Airway Segmentation Pipeline**](https://arxiv.org/abs/2203.04294), ***Preliminary version presented at RSNA2021***.
>
> Airway segmentation is essential for chest CT image analysis. However, it remains a challenging task because of the intrinsic complex tree-like structure and imbalanced sizes of airway branches. Current deep learning-based methods focus on model structure design while the potential of training strategy and loss function have not been fully explored. Therefore, we present a simple yet effective airway segmentation pipeline, denoted NaviAirway, which finds finer bronchioles with a bronchiole-sensitive loss function and a human-vision-inspired iterative training strategy. Experimental results show that NaviAirway outperforms existing methods, particularly in identification of higher generation bronchioles and robustness to new CT scans.

The original implementation and research were conducted by Anton Wang.

## Adaptations

The version of the code in this repository was adapted by the user previously associated with `pmutha@emory.edu` to use the `lungmask` library for lung segmentation, simplifying the preprocessing pipeline.

Further modifications have been made to refactor the core logic from a standalone script into a modular, importable function. This allows the segmentation pipeline to be easily integrated into other projects and to process entire directories of CT imaging data without modifying the source code.

## Model Weights

The required model weights can be downloaded from [this OneDrive link](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/wangad_connect_hku_hk/EquVeqhZkGdPoRZ6Ay0gDSgBHeaV2uP94ajR4MEj3B3fjw?e=eJgSta) (password: `2333`).

Place the downloaded `.pkl` files into the `checkpoint/` directory.

## Dependencies

- See the `environment.yaml` file for required Python packages.

## Usage

The segmentation logic has been wrapped into a convenient function. It can be used to process a directory containing CT imaging data. The script will automatically find subdirectories, process the `CT.nii.gz` file within each, and save the results to a specified output directory.

**Example:**

```python
from func import run_segmentation
import os

# --- Configuration ---
# Directory containing CT imaging folders (e.g., /data/LIDC-IDRI/)
# Each CT imaging folder should contain a 'CT.nii.gz' file.
input_data_dir = "/path/to/your/ct_imaging_data"

# Directory where segmentation results will be saved
output_dir = "/path/to/your/output_folder"

# --- Run Segmentation ---
if __name__ == "__main__":
    if not os.path.isdir(input_data_dir):
        print(f"Error: Input directory not found at {input_data_dir}")
    else:
        # The function will create the output directory if it doesn't exist.
        run_segmentation(input_data_dir, output_dir)
        print("Segmentation complete.")
```

The script will produce an `Airway.nii.gz` file for each CT imaging in the corresponding output subfolder. If a lung mask is not present, it will be generated and saved as `LungMask.nii.gz`.

## Contact

- For questions about the original NaviAirway research, please contact `wangad@connect.hku.hk`.
- For questions about the `lungmask` adaptation, please refer to the previous maintainer at `pmutha@emory.edu`.
