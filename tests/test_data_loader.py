from src.data.data_loader import load_brats_data, load_kaggle_data

def test_load_brats_data():
    # Test loading BraTS data
    images, masks = load_brats_data('data/raw/BraTS')
    assert len(images) > 0, "BraTS images should be loaded"
    assert len(masks) > 0, "BraTS masks should be loaded"
    assert images[0].shape == (128, 128, 3), "Image shape should be (128, 128, 3)"
    assert masks[0].shape == (128, 128, 1), "Mask shape should be (128, 128, 1)"

def test_load_kaggle_data():
    # Test loading Kaggle data
    images, labels = load_kaggle_data('data/raw/kaggle')
    assert len(images) > 0, "Kaggle images should be loaded"
    assert len(labels) > 0, "Kaggle labels should be loaded"
    assert images[0].shape == (128, 128, 3), "Image shape should be (128, 128, 3)"
    assert len(labels) == len(images), "Number of labels should match number of images"