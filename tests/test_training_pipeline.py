# tests/test_training_pipeline.py - tests of the training pipeline
import pytest
from train import train_model, run_pipeline
from data_processing import load_data, split_data, scale_features
 
def test_train_learns_weights():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train, X_test = scale_features(X_train, X_test)
    model = train_model(X_train, y_train)
    assert model.coef_.shape == (3, 13)
 
def test_train_rejects_wrong_input():
    X, y = load_data()
    with pytest.raises(ValueError):
        train_model(X, y[:10])          # X and y have different lengths
 
def test_pipeline_runs_end_to_end():
    model, metrics, X_test, y_test = run_pipeline()
    assert hasattr(model, "predict")
    assert set(metrics) == {"accuracy", "f1_score", "roc_auc"}
 
def test_pipeline_is_repeatable():
    _, first, _, _ = run_pipeline()
    _, second, _, _ = run_pipeline()
    assert first == second
