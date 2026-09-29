import pandas as pd

from medical_triage.data_preparation import clean_dataset, split_dataset


def test_clean_dataset_normalizes_and_removes_conflicting_texts():
    raw_df = pd.DataFrame(
        {
            "condition_label": [1, 1, 2, 3],
            "medical_abstract": [
                "  unique   abstract ",
                "unique abstract",
                "shared abstract",
                "shared   abstract",
            ],
        }
    )

    original_df = raw_df.copy(deep=True)

    result = clean_dataset(raw_df)

    expected = pd.DataFrame(
        {
            "condition_label": [1],
            "medical_abstract": ["unique abstract"],
        }
    )

    pd.testing.assert_frame_equal(result, expected)
    pd.testing.assert_frame_equal(raw_df, original_df)

def test_split_dataset_is_reproducible_and_has_no_overlap():
    clean_df = pd.DataFrame(
        [
            {
                "condition_label": label,
                "medical_abstract": f"class {label} document {index}",
            }
            for label in range(1, 6)
            for index in range(20)
        ]
    )

    datasets = split_dataset(clean_df)
    repeated_datasets = split_dataset(clean_df)

    assert len(datasets["train"]) == 70
    assert len(datasets["validation"]) == 15
    assert len(datasets["test"]) == 15

    text_sets = {
        name: set(dataset["medical_abstract"])
        for name, dataset in datasets.items()
    }

    assert text_sets["train"].isdisjoint(text_sets["validation"])
    assert text_sets["train"].isdisjoint(text_sets["test"])
    assert text_sets["validation"].isdisjoint(text_sets["test"])

    assert set.union(*text_sets.values()) == set(
        clean_df["medical_abstract"]
    )

    for name, dataset in datasets.items():
        assert set(dataset["condition_label"]) == {1, 2, 3, 4, 5}

        expected_per_class = 14 if name == "train" else 3
        assert (
            dataset["condition_label"].value_counts()
            == expected_per_class
        ).all()

        pd.testing.assert_frame_equal(
            dataset,
            repeated_datasets[name],
        )