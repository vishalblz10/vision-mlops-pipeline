import numpy as np
import torch

from vision_mlops.data import CLASSES, IMG_SIZE, extract_features, make_dataset


def test_dataset_shapes_and_types():
    X, y = make_dataset(30, seed=0)
    assert X.shape == (30, 3, IMG_SIZE, IMG_SIZE)
    assert X.dtype == torch.float32
    assert y.shape == (30,)
    assert y.dtype == torch.int64


def test_pixel_range():
    X, _ = make_dataset(20, seed=1)
    assert X.min() >= 0.0
    assert X.max() <= 1.0


def test_labels_cover_all_classes():
    _, y = make_dataset(60, seed=2)
    assert set(y.tolist()) == set(range(len(CLASSES)))


def test_determinism():
    X1, y1 = make_dataset(25, seed=42)
    X2, y2 = make_dataset(25, seed=42)
    assert torch.equal(X1, X2)
    assert torch.equal(y1, y2)


def test_different_seeds_differ():
    X1, _ = make_dataset(25, seed=1)
    X2, _ = make_dataset(25, seed=2)
    assert not torch.equal(X1, X2)


def test_brightness_shift_raises_mean():
    X_ref, _ = make_dataset(50, seed=3)
    X_shifted, _ = make_dataset(50, seed=3, brightness_shift=0.3)
    assert X_shifted.mean() > X_ref.mean()


def test_extract_features_keys_and_shapes():
    X, _ = make_dataset(15, seed=4)
    feats = extract_features(X)
    assert set(feats) == {"brightness", "contrast"}
    for values in feats.values():
        assert isinstance(values, np.ndarray)
        assert values.shape == (15,)
