"""
data.py -- Carga CSV, aplica filtro de existencia y split S1.
Devuelve DataFrames crudos para que features.py los procese.
"""

import pandas as pd
from config import (
    CSV_PATH, TRAIN_SUBJECTS, VAL_SUBJECTS, TEST_SUBJECTS,
    CONTACT_SUM_THRESHOLD,
)


def load_and_split(csv_path=CSV_PATH, contact_filter=True):
    """
    Carga el CSV y devuelve (train, val, test) como DataFrames.

    Si contact_filter=True aplica el filtro de existencia (contact_sum > 0).
    Si contact_filter=False devuelve todos los frames (uso GCN).
    """
    df = pd.read_csv(csv_path)

    if contact_filter:
        df = df[df["contact_sum"] > CONTACT_SUM_THRESHOLD]

    train = df[df["subject_id"].isin(TRAIN_SUBJECTS)].reset_index(drop=True)
    val   = df[df["subject_id"].isin(VAL_SUBJECTS)].reset_index(drop=True)
    test  = df[df["subject_id"].isin(TEST_SUBJECTS)].reset_index(drop=True)

    print(f"Train: {len(train):,} frames ({train.subject_id.nunique()} sujetos)")
    print(f"Val:   {len(val):,} frames ({val.subject_id.nunique()} sujetos)")
    print(f"Test:  {len(test):,} frames ({test.subject_id.nunique()} sujetos)")

    if contact_filter:
        n_removed = len(pd.read_csv(csv_path)) - (len(train) + len(val) + len(test))
        print(f"Filtro de existencia: {n_removed:,} frames descartados (contact_sum = 0)")

    return train, val, test


def get_xyz(df):
    """Extrae los 63 valores XYZ como ndarray (N, 63)."""
    return df.iloc[:, 5:].values.astype("float32")


def get_labels(df):
    """Extrae grasp_type como ndarray (N,)."""
    return df["grasp_type"].values


def get_sequence_ids(df):
    """Extrae sequence_id como Series."""
    return df["sequence_id"]


if __name__ == "__main__":
    train, val, test = load_and_split()
