import sqlite3
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_invoice_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(base_dir, "data", "inventory.db")

    query = """
    WITH purchase_agg AS (
        SELECT
            p.PONumber,
            COUNT(DISTINCT p.Brand) AS total_brands,
            SUM(p.Quantity) AS total_item_quantity,
            SUM(p.Dollars) AS total_item_dollars,
            AVG(julianday(p.ReceivingDate) - julianday(p.PODate))
                AS avg_receiving_delay
        FROM purchases p
        GROUP BY p.PONumber
    )
    SELECT
        vi.PONumber,
        vi.Dollars AS invoice_dollars,
        vi.Freight,
        julianday(vi.InvoiceDate) - julianday(vi.PODate)
            AS days_po_to_invoice,
        julianday(vi.PayDate) - julianday(vi.InvoiceDate)
            AS days_to_pay,
        pa.total_brands,
        pa.total_item_quantity,
        pa.total_item_dollars,
        pa.avg_receiving_delay
    FROM vendor_invoice vi
    LEFT JOIN purchase_agg pa
        ON vi.PONumber = pa.PONumber
    """

    # Load invoice and purchase data from SQLite
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn)

    return df


def create_invoice_risk_label(row):
    # Flag invoices with a large difference in total cost
    if pd.notna(row["total_item_dollars"]) and row["total_item_dollars"] > 0:
        dollar_diff_pct = (
            abs(row["invoice_dollars"] - row["total_item_dollars"])
            / row["total_item_dollars"]
        )
        if dollar_diff_pct > 0.20:
            return 1

    # Flag invoices with a receiving delay above 30 days
    if pd.notna(row["avg_receiving_delay"]) and row["avg_receiving_delay"] > 30:
        return 1

    # Flag invoices where freight is more than 20% of invoice value
    if pd.notna(row["Freight"]) and row["invoice_dollars"] > 0:
        freight_pct = row["Freight"] / row["invoice_dollars"]
        if freight_pct > 0.20:
            return 1

    # Flag invoices with a payment delay above 90 days
    if pd.notna(row["days_to_pay"]) and row["days_to_pay"] > 90:
        return 1

    return 0


def apply_labels(df):
    df = df.copy()
    df["flag_invoice"] = df.apply(create_invoice_risk_label, axis=1)
    return df


def split_data(df, features, target):
    X = df[features]
    y = df[target]

    return train_test_split(
        X, y, test_size=0.2, random_state=42
    )


def scale_features(X_train, X_test, scaler_path):
    scaler = StandardScaler()

    # Fit the scaler on training data and apply it to both datasets
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, scaler_path)

    return X_train_scaled, X_test_scaled
