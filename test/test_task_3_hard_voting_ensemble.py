import pytest
import numpy as np
import pandas as pd
import sklearn
from assg.tasks import task_1_load_data
from assg.tasks import train_val_test_split
from assg.tasks import task_3_voting_ensemble


@pytest.fixture
def data():
    X, y = task_1_load_data()
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y, train_size=10000, val_size=4510)
    vc = task_3_voting_ensemble(X_train, y_train, voting='hard')
    return vc, X_val, y_val

def test_knn_estimator(data):
    vc, X_val, y_val = data
    estimator_names = list(vc.named_estimators_.keys())
    assert 'knn' in estimator_names
    model = vc.named_estimators_['knn']
    assert isinstance(model, sklearn.neighbors._classification.KNeighborsClassifier)
    assert model.score(X_val, y_val) > 0.75

def test_dt_estimator(data):
    vc, X_val, y_val = data
    estimator_names = list(vc.named_estimators_.keys())
    assert 'dt' in estimator_names
    model = vc.named_estimators_['dt']
    assert isinstance(model, sklearn.tree._classes.DecisionTreeClassifier)
    assert model.score(X_val, y_val) > 0.75

def test_lr_estimator(data):
    vc, X_val, y_val = data
    estimator_names = list(vc.named_estimators_.keys())
    assert 'lr' in estimator_names
    model = vc.named_estimators_['lr']
    assert isinstance(model, sklearn.linear_model._logistic.LogisticRegression)
    assert model.score(X_val, y_val) > 0.75

def test_svc_estimator(data):
    vc, X_val, y_val = data
    estimator_names = list(vc.named_estimators_.keys())
    assert 'svc' in estimator_names
    model = vc.named_estimators_['svc']
    assert isinstance(model, sklearn.svm._classes.SVC)
    assert model.score(X_val, y_val) > 0.75

def test_mlp_estimator(data):
    vc, X_val, y_val = data
    estimator_names = list(vc.named_estimators_.keys())
    assert 'mlp' in estimator_names
    model = vc.named_estimators_['mlp']
    assert isinstance(model, sklearn.neural_network._multilayer_perceptron.MLPClassifier)
    assert model.score(X_val, y_val) > 0.75

def test_voting_ensemble(data):
    vc, X_val, y_val = data
    assert isinstance(vc, sklearn.ensemble._voting.VotingClassifier)
    assert vc.get_params()['voting'] == 'hard'
    assert vc.score(X_val, y_val) > 0.75
