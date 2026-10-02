import argparse
from pathlib import Path

import joblib
import pandas as pd

from medical_triage.training import train_model


def main():
    parser = argparse.ArgumentParser(
        description="Create a synthetic model for CI smoke tests."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing artifact: {args.output}"
        )

    categories = {
        1: ("neoplasms", "tumor cancer oncology"),
        2: ("digestive system diseases", "digestive bowel intestinal"),
        3: ("nervous system diseases", "neural brain neurological"),
        4: ("cardiovascular diseases", "cardiac artery cardiovascular"),
        5: ("general pathological conditions", "general systemic pathology"),
    }

    rows = [
        {
            "condition_label": label,
            "medical_abstract": f"{terms} {suffix}",
        }
        for label, (_, terms) in categories.items()
        for suffix in ["clinical study", "treatment research", "patient outcomes"]
    ]

    model = train_model(pd.DataFrame(rows))

    artifact = {
        "pipeline": model,
        "label_mapping": {
            label: name
            for label, (name, _) in categories.items()
        },
        "purpose": "synthetic-ci-only",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output)

    print(f"Synthetic CI model saved to: {args.output}")


if __name__ == "__main__":
    main()