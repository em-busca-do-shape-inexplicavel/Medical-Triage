import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from medical_triage.data_preparation import TEXT_COL, TARGET_COL


def build_model(random_state: int = 42) -> Pipeline:
    """Build the configuration selected during model experiments."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=10.0,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def train_model(training_df: pd.DataFrame) -> Pipeline:
    """Fit the selected pipeline using the supplied training data."""
    required_columns = {TEXT_COL, TARGET_COL}

    if not required_columns.issubset(training_df.columns):
        raise ValueError("Training data is missing required columns.")

    if training_df.empty:
        raise ValueError("Training data is empty.")

    if training_df[[TEXT_COL, TARGET_COL]].isna().any().any():
        raise ValueError("Training data contains missing values.")

    if not training_df[TEXT_COL].map(
        lambda value: isinstance(value, str)
    ).all():
        raise ValueError("All training texts must be strings.")

    if training_df[TEXT_COL].str.strip().eq("").any():
        raise ValueError("Training data contains blank texts.")

    if training_df[TEXT_COL].duplicated().any():
        raise ValueError("Training texts must be unique.")

    if training_df[TARGET_COL].nunique() < 2:
        raise ValueError("Training requires at least two classes.")

    model = build_model()
    model.fit(
        training_df[TEXT_COL],
        training_df[TARGET_COL],
    )

    return model