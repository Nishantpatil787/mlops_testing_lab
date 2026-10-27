# tests/test_model_outputs.py - validation of model outputs
import numpy as np
import pytest
from train import run_pipeline
 
@pytest.fixture(scope="module")
def result():
    return run_pipeline()          # train once, use in all tests of this file
 
def test_predictions_are_valid(result):
    model, metrics, X_test, y_test = result
    pred = model.predict(X_test)
    assert len(pred) == len(X_test)
    assert set(np.unique(pred)) <= {0, 1, 2}
 
def test_probabilities_are_valid(result):
    model, metrics, X_test, y_test = result
    prob = model.predict_proba(X_test)
    assert ((prob >= 0) & (prob <= 1)).all()
    assert np.allclose(prob.sum(axis=1), 1.0)
 
def test_no_nan_in_output(result):
    model, metrics, X_test, y_test = result
    assert not np.isnan(model.predict_proba(X_test)).any()
 
def test_accuracy_is_good_enough(result):
    assert result[1]["accuracy"] >= 0.90
 
def test_f1_score_is_good_enough(result):
    assert result[1]["f1_score"] >= 0.90
 
def test_roc_auc_is_good_enough(result):
    assert result[1]["roc_auc"] >= 0.95
