import argparse
from pathlib import Path

import joblib
import pandas as pd

from medical_triage.data_preparation import TEXT_COL, TARGET_COL
from medical_triage.training import train_model


def main():
    project_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Train and export the selected text classifier."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root / "data" / "processed",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination for the new model artifact.",
    )
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(
            f"Output already exists: {args.output}. "
            "Choose a new path to preserve the existing model."
        )

    train_df = pd.read_csv(args.data_dir / "train.csv")
    validation_df = pd.read_csv(args.data_dir / "validation.csv")
    labels_df = pd.read_csv(args.data_dir / "labels.csv")

    if labels_df[TARGET_COL].duplicated().any():
        raise ValueError("The label mapping contains duplicate labels.")

    if labels_df[[TARGET_COL, "condition_name"]].isna().any().any():
        raise ValueError("The label mapping contains missing values.")

    final_train_df = pd.concat(
        [train_df, validation_df],
        ignore_index=True,
    )

    label_mapping = labels_df.set_index(TARGET_COL)[
        "condition_name"
    ].to_dict()

    if set(final_train_df[TARGET_COL]) != set(label_mapping):
        raise ValueError(
            "Training classes do not match the label mapping."
        )

    model = train_model(final_train_df)

    artifact = {
        "pipeline": model,
        "label_mapping": label_mapping,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output)

    loaded_artifact = joblib.load(args.output)

    sample_texts = final_train_df[TEXT_COL].head(20)
    original_predictions = model.predict(sample_texts)
    reloaded_predictions = loaded_artifact["pipeline"].predict(
        sample_texts
    )

    if not (original_predictions == reloaded_predictions).all():
        raise RuntimeError("Predictions changed after reloading the model.")

    print(f"Training samples: {len(final_train_df)}")
    print(f"Vocabulary size: {len(model.named_steps['tfidf'].vocabulary_)}")
    print(f"Model saved: {args.output}")
    print("Reloaded predictions match the original predictions.")


if __name__ == "__main__":
    main()