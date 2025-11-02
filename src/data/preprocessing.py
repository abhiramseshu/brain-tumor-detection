from skimage import io, transform
import numpy as np
import os
import nibabel as nib
from sklearn.model_selection import train_test_split
import cv2

def load_nifti_image(file_path):
    img = nib.load(file_path)
    return img.get_fdata()

def preprocess_nifti_image(image, target_size=(256, 256)):
    image = (image - np.min(image)) / (np.max(image) - np.min(image))  # Normalize to [0, 1]
    image = transform.resize(image, target_size, mode='reflect', anti_aliasing=True)
    return image

def load_kaggle_image(file_path):
    image = io.imread(file_path)
    return image

def preprocess_kaggle_image(image, target_size=(256, 256)):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = image / 255.0  # Normalize to [0, 1]
    image = cv2.resize(image, target_size)  # Resize to target size
    return image

def preprocess_brats_data(data_dir, target_size=(256, 256)):
    images = []
    labels = []
    for patient_dir in os.listdir(data_dir):
        patient_path = os.path.join(data_dir, patient_dir)
        for modality in os.listdir(patient_path):
            modality_path = os.path.join(patient_path, modality)
            image = load_nifti_image(modality_path)
            processed_image = preprocess_nifti_image(image, target_size)
            images.append(processed_image)
            # Assuming labels are stored in a specific way, adjust as necessary
            labels.append(get_label_from_patient_id(patient_dir))
    return np.array(images), np.array(labels)

def preprocess_kaggle_data(data_dir, target_size=(256, 256)):
    images = []
    labels = []
    for image_file in os.listdir(data_dir):
        image_path = os.path.join(data_dir, image_file)
        image = load_kaggle_image(image_path)
        processed_image = preprocess_kaggle_image(image, target_size)
        images.append(processed_image)
        labels.append(get_label_from_filename(image_file))  # Adjust as necessary
    return np.array(images), np.array(labels)

def get_label_from_patient_id(patient_id):
    # Implement logic to extract label from patient ID
    pass

def get_label_from_filename(filename):
    # Implement logic to extract label from filename
    pass

def split_data(images, labels, test_size=0.2):
    return train_test_split(images, labels, test_size=test_size, random_state=42)