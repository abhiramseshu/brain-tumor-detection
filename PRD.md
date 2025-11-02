# Brain Tumor Detection & Classification - PRD

## Project Overview

**Goal:** Build a deep learning system for brain tumor analysis from MRI scans using PyTorch.

**Three Core Tasks:**
1. **Segmentation** - Locate and measure tumors using 3D U-Net (Target: Dice >85%)
2. **Grading** - Classify as High-Grade (HGG) or Low-Grade (LGG) glioma (Target: Accuracy >90%)
3. **Classification** - Identify tumor type: glioma, meningioma, pituitary, or no tumor (Target: Accuracy >95%)

**Tech Stack:** PyTorch 2.0+, Python 3.12, NiBabel, SimpleITK

---

## Datasets

### BraTS 2023 (Segmentation + Grading)
- **Source:** https://www.kaggle.com/datasets/bkb2024/brats-2023-training
- **Size:** ~50GB, 1,250 patients
- **Format:** NIfTI files (.nii.gz)
- **Contents:** 
  - 4 MRI modalities per patient: T1, T1ce, T2, FLAIR
  - Ground truth segmentation masks
  - Grade labels (HGG/LGG)

### Kaggle Brain Tumor MRI (Classification)
- **Source:** https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri
- **Size:** ~250MB, 3,000 images
- **Format:** JPEG/PNG
- **Classes:** glioma, meningioma, pituitary, no_tumor

---

## Model Architectures

### 1. Segmentation Model
- **Architecture:** 3D U-Net
- **Input:** (Batch, 4, 128, 128, 128) - 4 modalities stacked
- **Output:** (Batch, 4, 128, 128, 128) - 4 segmentation classes
- **Components:** 
  - Encoder: 4 conv3d blocks with max pooling
  - Bottleneck: 512 channels
  - Decoder: 4 upconv blocks with skip connections

### 2. Grading Model
- **Architecture:** CNN with ResNet50 backbone
- **Input:** (Batch, 4, 128, 128, 128) - same as segmentation
- **Output:** (Batch, 2) - HGG vs LGG
- **Modifications:** 
  - First conv layer adapted for 4 input channels
  - FC layers: 512 → 256 → 2

### 3. Classification Model
- **Architecture:** CNN with EfficientNet-B0 backbone
- **Input:** (Batch, 1, 224, 224) - 2D grayscale images
- **Output:** (Batch, 4) - 4 tumor types
- **FC Layers:** 1280 → 512 → 4

---

## Data Preprocessing

### BraTS Pipeline
1. Load NIfTI files using nibabel
2. Stack 4 modalities into single array
3. Z-score normalization per modality
4. Resize to 128×128×128
5. Augmentation: rotation, flip, elastic deformation

### Kaggle Pipeline
1. Load JPEG/PNG images
2. Resize to 224×224
3. Normalize to [0, 1]
4. Convert to grayscale if needed
5. Augmentation: rotation, flip, brightness/contrast

---

## Training Configuration

| Task | Loss Function | Optimizer | Batch Size | Epochs |
|------|--------------|-----------|------------|--------|
| Segmentation | Dice + CrossEntropy | Adam (1e-4) | 4 | 100 |
| Grading | Weighted CrossEntropy | Adam (1e-4) | 8 | 100 |
| Classification | Focal Loss | Adam (1e-4) | 16 | 100 |

**Additional:**
- Early stopping: patience=15
- Learning rate scheduler: ReduceLROnPlateau
- Gradient clipping for stability
- TensorBoard logging

---

## Project Structure

```
brain-tumor-detection/
├── src/
│   ├── main.py                 # Main pipeline
│   ├── inference.py            # Inference script
│   ├── models/
│   │   ├── unet_model.py      # 3D U-Net for segmentation
│   │   └── cnn_model.py       # CNNs for grading/classification
│   ├── data/
│   │   ├── data_loader.py     # PyTorch Dataset classes
│   │   └── preprocessing.py   # Preprocessing utilities
│   ├── training/
│   │   ├── train.py           # Training loops
│   │   ├── validate.py        # Validation functions
│   │   └── losses.py          # Dice, Focal losses
│   ├── evaluation/
│   │   ├── metrics.py         # Dice, IoU, accuracy, F1
│   │   └── visualize.py       # Plotting functions
│   └── utils/
│       └── helpers.py         # Config loading, model saving
├── data/
│   ├── raw/                   # Downloaded datasets (gitignored)
│   │   ├── BraTS/
│   │   └── kaggle/
│   └── processed/             # Preprocessed data (gitignored)
├── notebooks/
│   ├── exploratory_data_analysis.ipynb
│   └── model_evaluation.ipynb
├── configs/
│   └── config.yaml           # Hyperparameters and paths
├── models/                    # Saved model weights (gitignored)
├── tests/                     # Unit tests
├── requirements.txt
├── TASKS.md                  # Development checklist
└── README.md
```

---

## Success Metrics

### Performance Targets
- **Segmentation:** Dice Score >85%, Hausdorff Distance <5mm
- **Grading:** Accuracy >90%, F1-Score >88%
- **Classification:** Accuracy >95%, Per-class F1 >93%

### Technical Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA GPU (recommended)
- 16GB+ RAM
- 100GB+ storage

---

## Development Phases

1. **Setup** ✅ - Project structure, dependencies
2. **Data** - Download and organize datasets
3. **EDA** - Explore and visualize data
4. **Preprocessing** - Build data pipelines
5. **Models** - Implement U-Net and CNNs
6. **Training** - Train all three models
7. **Evaluation** - Calculate metrics, visualize results
8. **Integration** - Build end-to-end pipeline
9. **Testing** - Unit tests and bug fixes
10. **Documentation** - README and code docs

See TASKS.md for detailed checklist.

---

## Implementation Notes

**Key Principles:**
- PyTorch only (no TensorFlow)
- Config-driven design (YAML files)
- Modular, testable code
- Version control with Git
- Exclude large datasets from Git

**AI-Assisted Development:**
- Using GitHub Copilot for implementation
- Phase-by-phase incremental approach
- Continuous testing during development

---

**Version:** 1.0  
**Last Updated:** November 3, 2025  
**Status:** Phase 1 Complete ✅
