import pytest
import numpy as np
import pandas as pd
from assg.tasks import task_1_load_data


@pytest.fixture
def data():
    X, y = task_1_load_data()
    return X, y

def test_loaded_types(data):
    X, y = data
    assert isinstance(X, pd.core.frame.DataFrame)
    assert isinstance(y, np.ndarray)

def test_X_properties(data):
    X, _ = data
    assert X.shape == (19020, 10)
    assert list(X.columns) == ['fLength:', 'fWidth:', 'fSize:', 'fConc:', 'fConc1:', 'fAsym:', 'fM3Long:', 'fM3Trans:', 'fAlpha:', 'fDist:']

def test_y_properties(data):
    _, y = data
    assert y.shape == (19020,)
    assert y.sum() == 12332
