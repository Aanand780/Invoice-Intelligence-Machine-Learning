
import joblib
from pathlib import Path

from data_preprocessing import (
    load_invoice_data,
    apply_labels,
    split_data,
    scale_features
)
from model_evaluation import train_random_forest, evaluate_classifier


FEATURES = [
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars",
    "days_po_to_invoice",
    "days_to_pay"
]

TARGET = "flag_invoice"


def main():
    # Save the model in the shared models folder
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and label invoice data
    df = load_invoice_data()
    print("Columns found in database:", df.columns.tolist())
    df = apply_labels(df)

    # Prepare the training and testing data
    X_train, X_test, y_train, y_test = split_data(
        df, FEATURES, TARGET
    )

    # Scale the features
    X_train_scaled, X_test_scaled = scale_features(
        X_train,
        X_test,
        str(output_dir / "scaler.pkl")
    )

    # Train and evaluate the Random Forest model
    grid_search = train_random_forest(X_train_scaled, y_train)

    evaluate_classifier(
        grid_search.best_estimator_,
        X_test_scaled,
        y_test,
        "Random Forest Model"
    )

    # Save the trained model
    model_save_path = output_dir / "predict_flag_invoice.pkl"
    joblib.dump(grid_search.best_estimator_, model_save_path)

    print(f"Success! Model saved to: {model_save_path}")


if __name__ == "__main__":
    main()
