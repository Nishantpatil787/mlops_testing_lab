# data_processing.py - data processing functions that will be tested
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
 
def load_data():
    """Return the features (DataFrame) and the target (Series)."""
    data = load_wine(as_frame=True)
    return data.data, data.target
 
def fill_missing_values(df):
    """Replace missing values of every column with the mean of that column."""
    return df.fillna(0)
 
def remove_duplicates(df):
    """Remove repeated rows and reset the row numbers."""
    return df.drop_duplicates().reset_index(drop=True)
 
def split_data(X, y, test_size=0.2, seed=42):
    """Split the data into train and test parts."""
    return train_test_split(X, y, test_size=test_size, random_state=seed)
 
def scale_features(X_train, X_test):
    """Scale the features. The scaler learns only from the train data."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s
