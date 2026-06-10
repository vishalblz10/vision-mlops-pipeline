import pytest

from vision_mlops.data import make_dataset
from vision_mlops.train import train_model


@pytest.fixture(scope="session")
def shapes_data():
    X_train, y_train = make_dataset(1200, seed=42)
    X_test, y_test = make_dataset(300, seed=7)
    return X_train, y_train, X_test, y_test


@pytest.fixture(scope="session")
def trained_model(shapes_data):
    X_train, y_train, _, _ = shapes_data
    model, history = train_model(X_train, y_train, epochs=5, seed=42)
    return model, history
