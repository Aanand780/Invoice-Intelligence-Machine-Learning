
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split


# Load invoice data from SQLite database
def load_invoice_data(db_path: str) -> pd.DataFrame:

    query = "SELECT * FROM vendor_invoice"

    with sqlite3.connect(db_path) as connection:
        df = pd.read_sql_query(query, connection)

    return df


# Select input features and target variable
def prepare_features(df: pd.DataFrame):

    X = df[["Dollars"]]
    # Keep the target one-dimensional for scikit-learn estimators.
    y = df["Freight"]

    return X, y


# Split dataset into training and testing sets
def split_data(X, y, test_size=0.2, random_state=42):

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )
