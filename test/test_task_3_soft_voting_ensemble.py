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
    vc = task_3_voting_ensemble(X_train, y_train, voting='soft')
    return vc, X_val, y_val

def test_voting_ensemble(data):
    # we won't retest individual models, just check the voting ensemble uses soft voting
    # correctly when asked
    vc, X_val, y_val = data
    assert isinstance(vc, sklearn.ensemble._voting.VotingClassifier)
    assert vc.get_params()['voting'] == 'soft'
    assert vc.score(X_val, y_val) > 0.75