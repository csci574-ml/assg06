import pytest
import numpy as np
import pandas as pd
import sklearn
from assg.tasks import task_1_load_data
from assg.tasks import train_val_test_split
from assg.tasks import task_3_voting_ensemble
from assg.tasks import create_stacked_data


@pytest.fixture
def data():
    X, y = task_1_load_data()
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y, train_size=10000, val_size=4510)
    vc = task_3_voting_ensemble(X_train, y_train, voting='soft')
    return vc, X_train, y_train, X_val, y_val, X_test, y_test

def test_stack_validation_data(data):
    vc, X_train, y_train, X_val, y_val, X_test, y_test = data
    X_stacked = create_stacked_data(vc, X_val)

    # we expect 5 estimators each generating 2 outputs
    assert X_stacked.shape == (4510, 10)

    # first two columns should be from first estimator
    X_0 = vc.estimators_[0].predict_proba(X_val)
    assert np.allclose(X_0, X_stacked[:,0:2])

    # last two columns should come from last estimator
    X_4 = vc.estimators_[4].predict_proba(X_val)
    assert np.allclose(X_4, X_stacked[:,8:10])

def test_stack_random_data(data):
    vc, X_train, y_train, X_val, y_val, X_test, y_test = data

    # a random set of 500 inputs with 10 numeric features
    data = np.random.rand(500, 10)
    X = pd.DataFrame(data, columns=X_train.columns)
    X_stacked = create_stacked_data(vc, X)

    # we expect 5 estimators each generating 2 outputs
    assert X_stacked.shape == (500, 10)

    # first two columns should be from first estimator
    X_0 = vc.estimators_[0].predict_proba(X)
    assert np.allclose(X_0, X_stacked[:,0:2])

    # last two columns should come from last estimator
    X_4 = vc.estimators_[4].predict_proba(X)
    assert np.allclose(X_4, X_stacked[:,8:10])

