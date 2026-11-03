import pytest
import numpy as np
import pandas as pd
import sklearn
from assg.tasks import task_1_load_data
from assg.tasks import train_val_test_split
from assg.tasks import task_6_stacked_ensemble


@pytest.fixture
def data():
    X, y = task_1_load_data()
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y, train_size=10000, val_size=4510)
    blending_estimator, voting_ensemble = task_6_stacked_ensemble(X_train, y_train, X_val, y_val)
    return blending_estimator, voting_ensemble

def test_voting_ensemble(data):
    blending_estimator, voting_ensemble = data

    assert isinstance(voting_ensemble, sklearn.ensemble._voting.VotingClassifier)
    assert voting_ensemble.get_params()['voting'] == 'soft'
    # did they fit with the training data? if so should fit on 10000 samples
    assert voting_ensemble.named_estimators_['knn'].n_samples_fit_ == 10000
    
def test_blending_estimator(data):
    blending_estimator, voting_ensemble = data

    assert isinstance(blending_estimator, sklearn.svm._classes.SVC)
    # did they fit with the training data? if so should fit on 10000 samples
    assert blending_estimator.shape_fit_ == (4510, 10)
