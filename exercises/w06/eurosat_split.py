"""W6, task Z0: download EuroSAT and write its fixed split as CSV files. Prepared by the lecturer; do not change.

Run once from the project root (224 MB to data/, outside Git):
    uv run python exercises/w06/eurosat_split.py

Writes data/eurosat/splits/train.csv, val.csv, test.csv (columns path,label) and classes.json: 27 000 images,
60/20/20, stratified by class, seed 42 (16 200 / 5 400 / 5 400), the same split as in the lecture.
"""

import csv
import json
from pathlib import Path

from sklearn.model_selection import train_test_split
from torchvision.datasets import EuroSAT

ROOT = Path("data/eurosat")
SEED = 42


def main():
    ds = EuroSAT(root=str(ROOT), download=True)
    paths = [str(Path(p).resolve()) for p, _ in ds.samples]
    labels = [label for _, label in ds.samples]
    trval, test = train_test_split(range(len(labels)), test_size=0.2, stratify=labels, random_state=SEED)
    train, val = train_test_split(trval, test_size=0.25, stratify=[labels[i] for i in trval], random_state=SEED)
    out = ROOT / "splits"
    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("val", val), ("test", test)):
        with open(out / f"{name}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "label"])
            writer.writerows((paths[i], labels[i]) for i in part)
    (out / "classes.json").write_text(json.dumps(ds.classes))
    print(f"{len(labels)} images, {len(ds.classes)} classes; train {len(train)}, val {len(val)}, test {len(test)} -> {out}/")


if __name__ == "__main__":
    main()
