# Brain Tumor Detection and Classification

This project aims to develop a deep learning model for the detection and classification of brain tumors using MRI images. The project utilizes Convolutional Neural Networks (CNN) for classification tasks and U-Net architecture for segmentation tasks, particularly focusing on the BraTS and Kaggle datasets.

## Project Structure

```
brain-tumor-detection
├── src
│   ├── main.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── cnn_model.py
│   │   └── unet_model.py
│   ├── data
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   └── preprocessing.py
│   ├── training
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── validate.py
│   ├── evaluation
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── visualize.py
│   └── utils
│       ├── __init__.py
│       └── helpers.py
├── data
│   ├── raw
│   │   ├── BraTS
│   │   └── kaggle
│   ├── processed
│   └── .gitkeep
├── notebooks
│   ├── exploratory_data_analysis.ipynb
│   └── model_evaluation.ipynb
├── configs
│   ├── config.yaml
│   └── model_config.yaml
├── tests
│   ├── __init__.py
│   ├── test_data_loader.py
│   └── test_preprocessing.py
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md
```

## Dataset Storage

1. **BraTS Dataset**: 
   - Store the BraTS dataset in `data/raw/BraTS`.
   - Organize the NIfTI files by patient ID and modality.

2. **Kaggle Dataset**: 
   - Store the Kaggle dataset in `data/raw/kaggle`.
   - Ensure images and labels are organized appropriately.

## Preprocessing Guidance

- **Preprocessing Steps**:
  - Load NIfTI files and convert them to NumPy arrays.
  - Normalize pixel values to a range of [0, 1].
  - Resize images to a consistent size (e.g., 128x128 or 256x256).
  - Apply data augmentation techniques (e.g., rotation, flipping) to increase dataset variability.

- **Segmentation and Grading**: 
  - Use BraTS labels for grading and volumetric analysis.
  - Implement functions in `src/data/preprocessing.py` to extract and process segmentation masks.

- **Classification**: 
  - Use the Kaggle dataset for tumor type classification.
  - Ensure the data loader can handle both datasets and their respective labels.

## Usage

1. Install the required dependencies listed in `requirements.txt`.
2. Prepare the datasets by placing them in the appropriate directories.
3. Run the main script `src/main.py` to initialize the model, load the data, and start the training and evaluation processes.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.