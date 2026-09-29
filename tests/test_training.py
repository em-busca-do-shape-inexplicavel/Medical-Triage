import pandas as pd

from medical_triage.training import train_model


def test_train_model_predicts_from_raw_text():
    training_df = pd.DataFrame(
        {
            "medical_abstract": [
                "cardiac artery treatment study",
                "cardiac artery clinical research",
                "digestive bowel treatment study",
                "digestive bowel clinical research",
            ],
            "condition_label": [1, 1, 2, 2],
        }
    )

    model = train_model(training_df)

    predictions = model.predict(
        [
            "cardiac artery treatment",
            "digestive bowel treatment",
        ]
    )

    assert predictions.tolist() == [1, 2]