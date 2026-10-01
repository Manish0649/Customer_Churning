
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Columns used to identify records or excluded from model input
EXCLUDED_COLUMNS = ["customer_id", "signup_date"]
TARGET_COLUMN = "churn"


def load_data(file_path):
    """Load the customer churn dataset from a CSV file."""
    df = pd.read_csv(file_path)

    print("Dataset loaded successfully.")
    print("Dataset shape:", df.shape)

    return df


def check_data_quality(df):
    """Report basic data-quality information without modifying the data."""
    print("Dataset shape:", df.shape)
    print("Total missing values:", df.isnull().sum().sum())
    print("Duplicate rows:", df.duplicated().sum())
    print(
        "Duplicate customer IDs:",
        df["customer_id"].duplicated().sum()
    )
    print(
        "Invalid churn values:",
        (~df["churn"].isin([0, 1])).sum()
    )

    print("\nMissing values by column:")
    print(df.isnull().sum())


def engineer_features(df):
    """Create date-based features from the signup date."""
    df = df.copy()

    df["signup_date"] = pd.to_datetime(df["signup_date"])
    df["signup_year"] = df["signup_date"].dt.year
    df["signup_month"] = df["signup_date"].dt.month

    return df


def prepare_features_and_target(df):
    """Separate model features from the churn target."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Required target column '{TARGET_COLUMN}' is missing."
        )

    df = engineer_features(df)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].copy()

    # Remove customer identifier and raw date.
    # signup_year and signup_month remain available as features.
    X = X.drop(columns=EXCLUDED_COLUMNS)

    return X, y


def build_preprocessor(X):
    """Create preprocessing pipelines for numerical and categorical data."""
    numeric_cols = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_cols = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols)
    ])

    print("Preprocessing pipeline created successfully.")
    print("Numerical columns:", len(numeric_cols))
    print("Categorical columns:", len(categorical_cols))
    print("Total input features:", X.shape[1])

    return preprocessor


def prepare_dataset(file_path):
    """Load data, check quality, and prepare model inputs."""
    df = load_data(file_path)

    check_data_quality(df)

    X, y = prepare_features_and_target(df)
    preprocessor = build_preprocessor(X)

    return X, y, preprocessor


if __name__ == "__main__":
    # Update this path to the location of your CSV file.
    dataset_path = Path("dataset/customer_churn_1M.csv")

    if dataset_path.exists():
        X, y, preprocessor = prepare_dataset(dataset_path)

        print("\nData preparation completed.")
        print("Features shape:", X.shape)
        print("Target shape:", y.shape)
    else:
        print(
            f"Dataset not found at: {dataset_path}\n"
            "Pass the correct CSV path when calling prepare_dataset()."
        )