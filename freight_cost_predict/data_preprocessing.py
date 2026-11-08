# Import SQLite database connection module
import sqlite3

# Import function to split dataset into training and testing sets
from sklearn.model_selection import train_test_split

# Import pandas library for data handling and manipulation
import pandas as pd


# Function to load vendor invoice data from SQLite database
def load_vendor_invoice_data(db_path: str):

    """
    Load vendor invoice data from SQLite database.
    """

    # Establish connection with the SQLite database
    connection = sqlite3.connect(db_path)

    # SQL query to fetch all records from vendor_invoice table
    query = "SELECT * FROM vendor_invoice"

    # Execute SQL query and store the results in a DataFrame
    df = pd.read_sql_query(query, connection)

    # Close the database connection
    connection.close()

    # Return the loaded dataset
    return df


# Function to prepare input features and target variable
def prepare_features(df: pd.DataFrame):

    """
    Select features and target variable.
    """

    # Select Dollars column as the input feature (independent variable)
    X = df[["Dollars"]]

    # Select Freight column as the target (dependent variable)
    y = df[["Freight"]]

    # Return input features and target variable
    return X,y


# Function to split the dataset into training and testing sets
def split_data(X, y, test_size=0.2, random_state = 42):

    """
    Split dataset into train and test sets.
    """

    # Split data into training (80%) and testing (20%) sets
    # random_state=42 ensures reproducible results
    return train_test_split(
        X, y, test_size = test_size, random_state = random_state
    )