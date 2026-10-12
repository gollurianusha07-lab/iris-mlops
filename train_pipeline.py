
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import to_categorical


# ==========================================
# 1. PROJECT FILE PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "iris.csv"
MODEL_PATH = BASE_DIR / "model.keras"
SCALER_PATH = BASE_DIR / "scaler.pkl"
ENCODER_PATH = BASE_DIR / "encoder.pkl"
LOG_PATH = BASE_DIR / "training_log.txt"


# ==========================================
# 2. LOGGING CONFIGURATION
# ==========================================

logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# ==========================================
# 3. TRAINING PIPELINE
# ==========================================

def train_pipeline():

    print("\n===== MLOps Training Pipeline =====")

    # Step 1: Check whether the dataset exists
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    # Step 2: Load the CSV dataset
    df = pd.read_csv(DATA_PATH)

    # Remove accidental CSV index columns
    df = df.loc[
        :,
        ~df.columns.astype(str).str.startswith("Unnamed:")
    ]

    if df.empty:
        raise ValueError("Dataset is empty.")

    # Step 3: Find the target column
    candidates = [
        "species",
        "target",
        "class",
        "variety",
    ]

    target_col = next(
        (
            col for col in df.columns
            if col.strip().lower() in candidates
        ),
        None,
    )

    if target_col is None:
        raise ValueError(
            "Target column not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Step 4: Prepare features and labels
    X_df = df.drop(columns=[target_col])

    # Remove ID columns because they are not flower measurements
    id_columns = [
        col for col in X_df.columns
        if col.strip().lower() in ["id", "index"]
    ]

    X_df = X_df.drop(columns=id_columns)

    # Convert feature columns to numbers
    X_df = X_df.apply(
        pd.to_numeric,
        errors="raise"
    )

    y_raw = df[target_col]

    # Validate missing values
    if X_df.empty:
        raise ValueError("No feature columns found.")

    if X_df.isnull().any().any():
        raise ValueError("Features contain missing values.")

    if y_raw.isnull().any():
        raise ValueError("Target labels contain missing values.")

    # Convert features to a NumPy array
    X = X_df.to_numpy(dtype=float)

    if not np.isfinite(X).all():
        raise ValueError(
            "Features contain infinite or invalid values."
        )

    # Step 5: Encode species labels
    encoder = LabelEncoder()

    y_labels = encoder.fit_transform(
        y_raw.astype(str).str.strip()
    )

    if len(encoder.classes_) < 2:
        raise ValueError(
            "At least two different species are required."
        )

    class_counts = pd.Series(y_labels).value_counts()

    if class_counts.min() < 2:
        raise ValueError(
            "Each species must have at least two records."
        )

    y = to_categorical(
        y_labels,
        num_classes=len(encoder.classes_)
    )

    print("Dataset rows:", len(df))
    print("Feature columns:", list(X_df.columns))
    print("Number of features:", X.shape[1])
    print("Species:", list(encoder.classes_))

    # Step 6: Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y_labels,
    )

    # Step 7: Scale the features
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Step 8: Build the neural network
    tf.keras.utils.set_random_seed(42)

    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(16, activation="relu"),
        Dense(8, activation="relu"),
        Dense(
            len(encoder.classes_),
            activation="softmax"
        ),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Step 9: Train the model
    print("\nTraining model...")

    model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=8,
        verbose=1,
    )

    # Step 10: Evaluate the model
    loss, _ = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    predictions = model.predict(
        X_test,
        verbose=0
    ).argmax(axis=1)

    actual = y_test.argmax(axis=1)

    accuracy = accuracy_score(
        actual,
        predictions
    )

    print("\nTest loss:", round(float(loss), 4))
    print("Test accuracy:", round(float(accuracy), 4))

    # Step 11: Save the trained model and preprocessing objects
    model.save(MODEL_PATH)

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoder, ENCODER_PATH)

    # Step 12: Record results in the training log
    logging.info(
        "Training completed | rows=%d | accuracy=%.4f | loss=%.4f",
        len(df),
        accuracy,
        loss,
    )

    print("\nModel saved:", MODEL_PATH.name)
    print("Scaler saved:", SCALER_PATH.name)
    print("Encoder saved:", ENCODER_PATH.name)
    print("Training log updated:", LOG_PATH.name)

    print("\n===== PIPELINE COMPLETED =====")


# ==========================================
# 4. RUN THE PIPELINE
# ==========================================

if __name__ == "__main__":

    try:
        train_pipeline()

    except Exception:
        logging.exception("Training pipeline failed")
        raise
