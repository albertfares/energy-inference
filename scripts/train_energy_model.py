import sys
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "training_data"
MODEL_DIR = PROJECT_ROOT / "results" / "models"
MODEL_PATH = MODEL_DIR / "energy_cpu_linear.joblib"


def load_training_data() -> pd.DataFrame:
    """
    Load and concatenate all CSVs from data/training_data.

    This script assumes you manually curate which CSVs should be used for
    training by copying/moving them into data/training_data/.
    """
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Training data directory not found: {DATA_DIR}")

    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

    print("Using training CSVs:")
    for csv_path in csv_files:
        print(f"  - {csv_path}")

    frames = [pd.read_csv(p) for p in csv_files]
    df = pd.concat(frames, ignore_index=True)
    return df


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build a simple feature matrix X and target y for energy prediction.

    Predict CPU energy per run (`energy_cpu_J`) from static
    model / hyperparameter-derived features (no measured latency):
    FLOPs, batch size, input resolution, and precision.
    """
    required_columns = [
        "energy_cpu_J",
        "flops_total",
        "batch",
        "resolution",
        "precision",
    ]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for training: {missing}")

    # Basic coercion / cleaning.
    df = df.copy()
    for col in ("energy_cpu_J", "flops_total", "batch", "resolution"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["precision"] = df["precision"].astype(str).str.lower().str.strip()
    df = df[df["precision"] != ""]
    df = df.dropna(subset=required_columns)

    numeric_feature_cols = ["flops_total", "batch", "resolution"]
    X_num = df[numeric_feature_cols]
    X_precision = pd.get_dummies(df["precision"], prefix="precision")
    X = pd.concat([X_num, X_precision], axis=1)
    y = df["energy_cpu_J"]
    return X, y


def train_simple_regressor(X: pd.DataFrame, y: pd.Series) -> None:
    """
    Train a simple linear regression model and print basic metrics.

    This is intentionally minimal: the goal is to get a working baseline
    predictor that you can iterate on later.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("Trained LinearRegression on CPU energy (no latency feature).")
    print(f"Test R^2:  {r2:.4f}")
    print(f"Test MAE:  {mae:.6f} J")
    print("Coefficients (in feature_names order):")
    print(model.coef_)
    print(f"Intercept: {model.intercept_:.6f}")

    # Persist model + feature schema for later inference.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "feature_names": list(X.columns),
        "target": "energy_cpu_J",
    }
    dump(payload, MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")


def main() -> None:
    try:
        df = load_training_data()
        X, y = build_feature_matrix(df)
        train_simple_regressor(X, y)
    except Exception as exc:  # noqa: BLE001
        print(f"Error while training energy model: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

