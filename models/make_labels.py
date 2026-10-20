import json
from pathlib import Path

folder = Path(__file__).parent
source = folder / "synset.txt"
destination = folder / "imagenet_labels.json"

with source.open(encoding="utf-8") as file:
    labels = [
        line.strip().split(" ", 1)[1]
        for line in file
        if line.strip()
    ]

with destination.open("w", encoding="utf-8") as file:
    json.dump(labels, file, ensure_ascii=False, indent=2)

print(f"Created {destination.name} with {len(labels)} labels.")