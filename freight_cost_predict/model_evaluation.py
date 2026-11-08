# Import math module for mathematical operations
import math

# Import Linear Regression algorithm
from sklearn.linear_model import LinearRegression

# Import Decision Tree Regression algorithm
from sklearn.tree import DecisionTreeRegressor

# Import Random Forest Regression algorithm
from sklearn.ensemble import RandomForestRegressor

# Import evaluation metrics for regression models
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Function to train a Linear Regression model
def train_linear_regression(x_train, y_train):

  # Create Linear Regression model
  model = LinearRegression()

  # Train the model using training data
  model.fit(x_train, y_train)

  # Return the trained model
  return model


# Function to train a Decision Tree Regression model
def train_decision_tree(x_train, y_train):

  # Create Decision Tree model with maximum depth of 5
  # random_state=42 ensures reproducible results
  model = DecisionTreeRegressor(max_depth=5, random_state=42)

  # Train the model using training data
  model.fit(x_train, y_train)

  # Return the trained model
  return model


# Function to train a Random Forest Regression model
def train_random_forest(x_train, y_train):

  # Create Random Forest model with maximum depth of 5
  # random_state=42 ensures reproducible results
  model = RandomForestRegressor(max_depth=5, random_state=42)

  # Train the model using training data
  model.fit(x_train, y_train)

  # Return the trained model
  return model


# Function to evaluate model performance using test data
def evaluate_model(model, x_test, y_test, model_name: str) -> dict:

  """
  Evaluate regression model and return performance metrics
  """

  # Predict target values using the trained model
  preds = model.predict(x_test)

  # Calculate Mean Absolute Error (MAE)
  mae = mean_absolute_error(y_test, preds)

  # Calculate Root Mean Squared Error (RMSE)
  mse = math.sqrt(mean_squared_error(y_test, preds))

  # Calculate R-squared score in percentage
  r2 = r2_score(y_test, preds) * 100

  # Display model performance
  print(f"\n{model_name} Performance:")
  print(f"MAE: {mae:.2f}")
  print(f"RMSE: {mse:.2f}")
  print(f"R2: {r2:.2f}")

  # Return model name and evaluation metrics as a dictionary
  return {
      'model_name': model_name,
      'mae': mae,
      'mse': mse,
      'r2': r2
  }