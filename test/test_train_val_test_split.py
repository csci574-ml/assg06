import pytest
import numpy as np
import pandas as pd
from assg.tasks import task_1_load_data
from assg.tasks import train_val_test_split


def test_generated_data():
    np.random.seed(42)
    X = np.random.rand(1000, 5)
    y = np.random.randint(0, 2, size=(1000,))
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y, train_size=500, val_size=300)

    # test train split is as expected
    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert X_train.shape == (500, 5)
    assert y_train.shape == (500,)

    # if random shuffling uses correct seed we should pass this, index 18351
    # ends up first training sample and we end up with 6475 positive targets
    assert X_train[0,0] == pytest.approx(0.06725591396736785)
    assert y_train.sum() == 257

    # test validation split
    assert isinstance(X_val, np.ndarray)
    assert isinstance(y_val, np.ndarray)
    assert X_val.shape == (300, 5)
    assert y_val.shape == (300,)
    assert X_val[0,0] == pytest.approx(0.4978125080342902)
    assert y_val.sum() == 159

    # test the test split
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert X_test.shape == (200, 5)
    assert y_test.shape == (200,)
    assert X_test[0,0] == pytest.approx(0.12786680122241756)
    assert y_test.sum() == 95

def test_assg_data():
    X, y = task_1_load_data()
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y, train_size=10000, val_size=4510)

    # test train split is as expected
    assert isinstance(X_train, pd.core.frame.DataFrame)
    assert isinstance(y_train, np.ndarray)
    assert X_train.shape == (10000, 10)
    assert y_train.shape == (10000,)

    # if random shuffling uses correct seed we should pass this, index 18351
    # ends up first training sample and we end up with 6475 positive targets
    assert X_train.iloc[0].name == 18351
    assert y_train.sum() == 6475

    # test validation split
    assert isinstance(X_val, pd.core.frame.DataFrame)
    assert isinstance(y_val, np.ndarray)
    assert X_val.shape == (4510, 10)
    assert y_val.shape == (4510,)
    assert X_val.iloc[0].name == 9394
    assert y_val.sum() == 2918

    # test the test split
    assert isinstance(X_test, pd.core.frame.DataFrame)
    assert isinstance(y_test, np.ndarray)
    assert X_test.shape == (4510, 10)
    assert y_test.shape == (4510,)
    assert X_test.iloc[0].name == 8156
    assert y_test.sum() == 2939

