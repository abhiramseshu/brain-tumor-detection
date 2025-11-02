from pathlib import Path
import nibabel as nib
import numpy as np
import os
import cv2

def load_brats_data(data_dir):
    images = []
    masks = []
    
    for patient_dir in Path(data_dir).iterdir():
        if patient_dir.is_dir():
            # Load images
            for img_file in patient_dir.glob('*.nii.gz'):
                img = nib.load(str(img_file)).get_fdata()
                images.append(img)
                
            # Load masks
            for mask_file in patient_dir.glob('*_seg.nii.gz'):
                mask = nib.load(str(mask_file)).get_fdata()
                masks.append(mask)
    
    return np.array(images), np.array(masks)

def load_kaggle_data(data_dir):
    images = []
    labels = []
    
    for img_file in Path(data_dir).glob('*.jpg'):
        img = cv2.imread(str(img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        
        # Assuming labels are stored in a CSV file
        label_file = data_dir / 'labels.csv'
        if label_file.exists():
            labels_df = pd.read_csv(label_file)
            label = labels_df[labels_df['filename'] == img_file.name]['label'].values[0]
            labels.append(label)
    
    return np.array(images), np.array(labels)

def preprocess_images(images, target_size=(256, 256)):
    processed_images = []
    
    for img in images:
        img = cv2.resize(img, target_size)
        img = img / 255.0  # Normalize to [0, 1]
        processed_images.append(img)
    
    return np.array(processed_images)

def preprocess_masks(masks, target_size=(256, 256)):
    processed_masks = []
    
    for mask in masks:
        mask = cv2.resize(mask, target_size)
        processed_masks.append(mask)
    
    return np.array(processed_masks)