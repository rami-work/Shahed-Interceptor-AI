import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture
def model():
    from ultralytics import YOLO

    return YOLO("models/best.pt")


def test_model_loads(model):
    assert model is not None
    assert model.names is not None


def test_model_classes(model):
    assert "shahed" in model.names.values()


def test_model_predict(model):
    results = model.predict("assets/detect1.gif", stream=True, verbose=False)
    first = next(results, None)
    assert first is not None
