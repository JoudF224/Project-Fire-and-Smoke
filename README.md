<img width="590" height="52" alt="image" src="https://github.com/user-attachments/assets/d6119724-747f-4e70-aaf1-7c5d099134f5" /># Fire and Smoke Detection with RT-DETR, YOLO11 and YOLOv10
### A Comparison on Original and Cleaned Data

**Course:** CSC 403 – Selected Topics in Computer Science  
**Assignment:** #2 – Methodology  
**Supervisor:** Dr. Atheer  
**Group:** 2 | **Section:** 4  

## Overview

This project compares three real-time object detectors for fire and smoke detection: one transformer-based model (RT-DETR-L) and two CNN-based models (YOLO11s and YOLOv10s). The public MyData_Fire dataset was first audited and cleaned, because it contained exact duplicates, duplicates shared between the training, validation and test splits, and conflicting annotations. Each model was trained with two random seeds on both the original dataset (to compare with a previous benchmark study) and the cleaned dataset (to measure performance without data leakage), under an identical protocol: 30 epochs, AdamW, mixed precision, on a free Colab Tesla T4 GPU.

## Team and Contributions

| Member | Role |
|---|---|
| Maryam Abu Qurain | Sections 1 and 4, pipeline figure, training notebooks, data check after cleaning, YOLOv10s (seed 1) |
| Joud Alsaba | Sections 2 and 5, data audit and cleaning, RT-DETR-L (seed 1) |
| Joud Fayez Althagafi | Section 3.1 (RT-DETR-L), annotator A, RT-DETR-L (seed 0) |
| Lina Bashamakh | Section 3.2 (YOLO11s), annotator B, YOLO11s (seed 0) |
| Renad Ghazi AlYami | Section 3.3 (YOLOv10s), adjudication, agreement and FP/FN analysis, YOLOv10s (seed 0) |
| Retal Alharbi | Section 6, results analysis, threshold selection for the FP/FN analysis, YOLO11s (seed 1) |

## Datasets

| Version | Images (train / valid / test) | MD5 | Link |
|---|---|---|---|
| Original MyData_Fire | 8,826 / 1,088 / 1,107 | `fd9ed42cd67fd5f7c016ca127985e245` | [Figshare (official)](https://doi.org/10.25946/28747046.v1) |
| Cleaned (v8) | 7,115 / 761 / 782 | `e4f0e4fc1bbe96ce7ac61d91151edf21` | [Google Drive](https://drive.google.com/file/d/1nBbOcdJcH4HDuX8eBl1lyeXgw09k_jwI/view?usp=sharing) |

The original dataset was released by CQUniversity (Partheepan, 2025). The cleaned version is derived from it: exact duplicates and augmented copies in the validation and test splits were removed, 240 label files were corrected, and the data was re-split so that related scenes stay in the same split.

## Repository Structure

```
├── data/
│   ├── data_cleaning.ipynb              # data audit and cleaning
│   └── data_check_after_cleaning.ipynb  # independent check of the cleaned dataset
├── training/
│   ├── train_orgData.ipynb              # training template, original dataset
│   ├── train_cleanData.ipynb            # training template, cleaned dataset
│   └── rtdetr_batch16_test.ipynb        # preliminary memory test (batch 16)
├── runs/
│   ├── original_data/                   # 6 runs: notebook, results.csv, report.txt
│   ├── cleaned_data/                    # 6 runs: notebook, results.csv, report.txt
│   └── cleaned_data_failed/             # first attempt of RT-DETR-L seed 0 (diverged)
└── analysis/
    ├── Results_Analysis.ipynb           # results, agreement (Cohen's kappa), challenging conditions
    ├── FPFN_Analysis.ipynb              # false positives and false negatives (Methods A and B)
    ├── analysis_inputs/                 # annotator files
    ├── results_analysis/                # outputs of Results_Analysis
    └── results_fpfn/                    # outputs of FPFN_Analysis
```

## How to Reproduce a Run

1. Open `training/train_orgData.ipynb` or `training/train_cleanData.ipynb` in Google Colab.
2. Runtime > Change runtime type > **T4 GPU**.
3. In the first code cell, set `MODEL_TO_RUN` ("RT-DETR-L", "YOLO11s" or "YOLOv10s") and `SEED` (0 or 1).
4. Runtime > **Run all**. The dataset is downloaded and its MD5 is verified automatically.

All runs used Ultralytics 8.4.173, which each notebook installs automatically.

## Results

*To be added.*
| Model | Dataset | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|
| RT-DETR-L | Original | 0.000 ± 0.000 | 0.000 ± 0.000 |
| YOLO11s | Original | 0.000 ± 0.000 | 0.000 ± 0.000 |
| YOLOv10s | Original | 0.000 ± 0.000 | 0.000 ± 0.000 |
| RT-DETR-L | Cleaned | 0.000 ± 0.000 | 0.000 ± 0.000 |
| YOLO11s | Cleaned | 0.000 ± 0.000 | 0.000 ± 0.000 |
| YOLOv10s | Cleaned | 0.000 ± 0.000 | 0.000 ± 0.000 |

Values are the mean ± standard deviation over two seeds on the test split.

## Large Files (Google Drive)

The complete training outputs and large analysis outputs are not stored in this repository because of GitHub's file size limits. They are available here:

- Full training outputs of all 12 runs (model weights, training plots, test evaluation and logs): [Google Drive](https://drive.google.com/drive/folders/1HfcdjD6BbsgZaPKoxmnTubIWYAXzDcOK?usp=sharing)
- Full analysis outputs: [Google Drive](ADD_LINK_HERE)

## Notes

- Some runs were interrupted by Colab and resumed from the last checkpoint. In these runs, the notebook shows only the last session, while `results.csv` contains all 30 epochs.
- The notebook of `rt_detr_l_seed1_orgData` is an evaluation-only session; training was completed in earlier sessions.
- The first attempt of RT-DETR-L with seed 0 on the cleaned data diverged at epoch 6 and was repeated with the same settings (`runs/cleaned_data_failed/`). The repeated run diverged at epoch 27, and the checkpoint from epoch 26 was used for testing.
- A final rotation- and mirror-aware check found one test image with a 180°-rotated copy in the training split. It was kept and is reported as a limitation.
