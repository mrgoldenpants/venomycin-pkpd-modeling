"""Cleaning and validation functions for PK tabular data."""
"""
data_processing.py prepares clinical PK data for analysis. It converts required
columns to numeric values, flags suspicious observations such as negative
concentrations or duplicate times, and validates that a dataset is analysis-ready.
"""
import numpy as np
import pandas as pd


DEFAULT_NUMERIC_COLUMNS = (
    "dose",
    "A1",
    "A2",
    "C",
    "Time",
    "Volume",
    "CL",
    "Q",
)


def clean_data(data, numeric_columns=DEFAULT_NUMERIC_COLUMNS):
    """Return a cleaned copy with selected columns converted to numeric values."""
    cleaned = data.copy()
    missing = [column for column in numeric_columns if column not in cleaned.columns]
    if missing:
        raise ValueError(f"Missing required numeric columns: {missing}")
    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="raise")
    return cleaned


def flag_suspicious_rows(data):
    """Add flags for negative concentrations, duplicate times, and pre-dose rows."""
    flagged = data.copy()
    subject_column = next(
        (name for name in ("Subjects", "Subject") if name in flagged.columns),
        None,
    )
    flagged["Negative concentration"] = flagged["C"] < 0
    duplicate_columns = ["Time"] if subject_column is None else [subject_column, "Time"]
    flagged["Duplicate time"] = flagged.duplicated(
        subset=duplicate_columns, keep=False
    )

    if subject_column is None:
        dose_times = flagged.loc[flagged["dose"] > 0, "Time"]
        first_dose = dose_times.min() if not dose_times.empty else np.nan
        flagged["Before first dose"] = (
            flagged["Time"] < first_dose if np.isfinite(first_dose) else True
        )
    else:
        first_dose_by_subject = (
            flagged.loc[flagged["dose"] > 0]
            .groupby(subject_column)["Time"]
            .min()
        )
        first_dose = flagged[subject_column].map(first_dose_by_subject)
        flagged["Before first dose"] = first_dose.isna() | (
            flagged["Time"] < first_dose
        )
    return flagged


def validate_pk_data(data):
    """Return True when no suspicious rows or missing required values exist."""
    required = ["dose", "C", "Time"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if data[required].isna().any().any():
        return False
    flagged = flag_suspicious_rows(data)
    flag_columns = [
        "Negative concentration",
        "Duplicate time",
        "Before first dose",
    ]
    return not bool(flagged[flag_columns].any(axis=None))
