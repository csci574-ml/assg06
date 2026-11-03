import pytest
import numpy as np
import pandas as pd
import sklearn
from assg.tasks import task_1_load_data
from assg.tasks import train_val_test_split
from assg.tasks import task_4_bag_of_trees_ensemble


@pytest.fixture
def data():
    X, y = task_1_load_data()
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y, train_size=10000, val_size=4510)
    bag = task_4_bag_of_trees_ensemble(X_train, y_train)
    return bag, X_val, y_val

def test_bag_of_trees_ensemble(data):
    bag, X_val, y_val = data
    assert (isinstance(bag, sklearn.ensemble._forest.RandomForestClassifier) or
            isinstance(bag, sklearn.ensemble._forest.ExtraTreesClassifier))
    assert bag.score(X_val, y_val) > 0.85
