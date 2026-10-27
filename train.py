# train.py - training pipeline that will be tested
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from data_processing import (load_data, fill_missing_values,
                             split_data, scale_features)
 
def train_model(X_train, y_train):
    """Train a Logistic Regression model."""
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    return model
 
def evaluate_model(model, X_test, y_test):
    """Return accuracy, F1 score and ROC-AUC as a dictionary."""
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "f1_score": f1_score(y_test, pred, average="macro"),
        "roc_auc": roc_auc_score(y_test, prob, multi_class="ovr"),
    }
 
def run_pipeline():
    """Load -> clean -> split -> scale -> train -> evaluate."""
    X, y = load_data()
    X = fill_missing_values(X)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train, X_test = scale_features(X_train, X_test)
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    return model, metrics, X_test, y_test
