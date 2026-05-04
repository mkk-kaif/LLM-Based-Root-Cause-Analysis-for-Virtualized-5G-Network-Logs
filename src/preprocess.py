# =============================================================================
# preprocess.py
#
# Purpose:
#   This script ingests raw ETL log data, cleans and normalizes the log messages,
#   and extracts structured features for downstream tasks such as embedding,
#   similarity search, and model inference. The processed features are saved to
#   a CSV file, serving as a simple "ML Feature Store" for the RCA assistant.
# =============================================================================

import pandas as pd
from pathlib import Path


def load_logs(file_path: str) -> pd.DataFrame:
    # Reads logs from a CSV or JSON file.
    """
    Loads structured logs from a CSV or JSON file.
    Maps columns for 5G data if it's a CSV.
    """
    if file_path.endswith(".json"):
        df = pd.read_json(file_path)
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
        # Map 5G columns to standard schema
        mapping = {
            "Info": "message",
            "Protocol": "error_type",
            "Time": "timestamp"
        }
        df = df.rename(columns=mapping)
    else:
        raise ValueError("Expected a .json or .csv log file")
    return df


def clean_logs(df: pd.DataFrame) -> pd.DataFrame:
    # Cleans and normalizes the 'message' field for downstream processing.
    # - Drops rows with missing messages.
    # - Removes special characters and lowercases the message text.
    """
    Cleans and normalizes message field for downstream processing.
    """
    df = df.dropna(subset=["message"])
    # Note: timestamp in 5G CSV might not be a full date, but we'll try to parse it
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
    df["cleaned_message"] = df["message"].str.replace(r"[^a-zA-Z0-9 ]", "", regex=True).str.lower()
    return df


def create_feature_store(df: pd.DataFrame, out_path: str = "outputs/feature_store.csv"):
    # Extracts key elements from logs and saves them to a CSV file.
    """
    Extracts features from log messages and saves them to CSV.
    Includes Source and Destination for network context.
    """
    # If error_type wasn't mapped (e.g. JSON), derive it
    if "error_type" not in df.columns:
        df["error_type"] = df["cleaned_message"].apply(lambda x: x.split(" ")[0])

    cols_to_keep = ["timestamp", "error_type", "cleaned_message"]
    if "Source" in df.columns:
        cols_to_keep.append("Source")
    if "Destination" in df.columns:
        cols_to_keep.append("Destination")
    if "error_code" in df.columns:
        cols_to_keep.append("error_code")

    features = df[cols_to_keep]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_path, index=False)
    print(f"Feature store saved to: {out_path}")


if __name__ == "__main__":
    # Main execution flow:
    # 1. Load raw logs from the 5G CSV file.
    # 2. Clean and normalize the logs.
    # 3. Extract features and save them to the feature store CSV.
    input_log_path = "data/5g_logs.csv"
    logs_df = load_logs(input_log_path)
    cleaned_df = clean_logs(logs_df)
    create_feature_store(cleaned_df)
