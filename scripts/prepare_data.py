import argparse
from pathlib import Path

import pandas as pd

from medical_triage.data_preparation import clean_dataset, split_dataset


def main():
    project_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Prepare reproducible medical text datasets."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=project_root / "data" / "raw",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "processed",
    )
    args = parser.parse_args()

    train_raw = pd.read_csv(args.raw_dir / "medical_tc_train.csv")
    test_raw = pd.read_csv(args.raw_dir / "medical_tc_test.csv")
    labels = pd.read_csv(args.raw_dir / "medical_tc_labels.csv")

    raw_df = pd.concat(
        [train_raw, test_raw],
        ignore_index=True,
    )

    clean_df = clean_dataset(raw_df)
    datasets = split_dataset(clean_df)

    if not set(clean_df["condition_label"]).issubset(
        set(labels["condition_label"])
    ):
        raise ValueError("Dataset contains labels absent from the label mapping.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name, dataset in datasets.items():
        output_path = args.output_dir / f"{name}.csv"
        dataset.to_csv(output_path, index=False)
        print(f"{name}: {len(dataset)} rows -> {output_path}")

    labels.to_csv(args.output_dir / "labels.csv", index=False)


if __name__ == "__main__":
    main()