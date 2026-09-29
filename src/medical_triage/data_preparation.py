import pandas as pd
from sklearn.model_selection import train_test_split

TEXT_COL = "medical_abstract"
TARGET_COL = "condition_label"


def clean_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize whitespace and remove ambiguous and duplicate texts."""
    required_columns = {TEXT_COL, TARGET_COL}
    missing_columns = required_columns.difference(raw_df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if raw_df.empty:
        raise ValueError("The dataset is empty.")

    if raw_df[[TEXT_COL, TARGET_COL]].isna().any().any():
        raise ValueError("Text and target columns must not contain missing values.")

    if not raw_df[TEXT_COL].map(lambda value: isinstance(value, str)).all():
        raise ValueError("All texts must be strings.")

    prepared_df = raw_df.copy()

    prepared_df[TEXT_COL] = (
        prepared_df[TEXT_COL]
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    if prepared_df[TEXT_COL].eq("").any():
        raise ValueError("Texts must not be empty after normalization.")

    labels_per_text = (
        prepared_df.groupby(TEXT_COL)[TARGET_COL].nunique()
    )

    ambiguous_texts = labels_per_text[
        labels_per_text > 1
    ].index

    clean_df = (
        prepared_df[
            ~prepared_df[TEXT_COL].isin(ambiguous_texts)
        ]
        .drop_duplicates(subset=[TEXT_COL])
        .reset_index(drop=True)
    )

    if clean_df.empty:
        raise ValueError("No records remain after cleaning.")

    return clean_df

def split_dataset(
    clean_df: pd.DataFrame,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """Split unique texts into stratified 70/15/15 datasets."""
    if clean_df[TEXT_COL].duplicated().any():
        raise ValueError("Texts must be unique before splitting.")

    train_df, temporary_df = train_test_split(
        clean_df,
        test_size=0.30,
        stratify=clean_df[TARGET_COL],
        random_state=random_state,
    )

    validation_df, test_df = train_test_split(
        temporary_df,
        test_size=0.50,
        stratify=temporary_df[TARGET_COL],
        random_state=random_state,
    )

    return {
        "train": train_df.reset_index(drop=True),
        "validation": validation_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }