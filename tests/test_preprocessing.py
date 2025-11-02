import unittest
import numpy as np
from src.data.preprocessing import preprocess_image

class TestPreprocessing(unittest.TestCase):

    def test_preprocess_image_shape(self):
        # Test if the image is resized correctly
        input_image = np.random.rand(240, 240, 3)  # Simulating a random image
        processed_image = preprocess_image(input_image)
        self.assertEqual(processed_image.shape, (128, 128, 3), "Image shape should be (128, 128, 3) after preprocessing.")

    def test_preprocess_image_normalization(self):
        # Test if the image is normalized correctly
        input_image = np.random.rand(240, 240, 3) * 255  # Simulating a random image with pixel values [0, 255]
        processed_image = preprocess_image(input_image)
        self.assertTrue(np.all(processed_image >= 0) and np.all(processed_image <= 1), "Image pixel values should be normalized to [0, 1].")

    def test_preprocess_image_augmentation(self):
        # Test if augmentation is applied (this is a placeholder, actual implementation may vary)
        input_image = np.random.rand(240, 240, 3)
        processed_image = preprocess_image(input_image, augment=True)
        self.assertEqual(processed_image.shape, (128, 128, 3), "Augmented image shape should be (128, 128, 3) after preprocessing.")

if __name__ == '__main__':
    unittest.main()