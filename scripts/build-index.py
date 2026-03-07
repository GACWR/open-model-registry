#!/usr/bin/env python3
"""Aggregate all models/*/model.json into registry/index.json.

Run this script from the repository root:
    python scripts/build-index.py

Or let the GitHub Actions workflow run it on every push.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parent.parent
    models_dir = repo_root / "models"
    output_file = repo_root / "registry" / "index.json"

    if not models_dir.exists():
        print(f"Error: {models_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    models = []
    errors = []

    for model_dir in sorted(models_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        manifest = model_dir / "model.json"
        if not manifest.exists():
            errors.append(f"Missing model.json in {model_dir.name}")
            continue

        try:
            with open(manifest) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON in {model_dir.name}/model.json: {e}")
            continue

        # Validate required fields
        required = ["name", "version", "description", "author", "framework", "category", "files"]
        missing = [f for f in required if f not in data]
        if missing:
            errors.append(f"{model_dir.name}: missing required fields: {missing}")
            continue

        # Ensure name matches directory
        if data["name"] != model_dir.name:
            errors.append(f"{model_dir.name}: name '{data['name']}' doesn't match directory")
            continue

        # Verify listed files exist
        for fname in data.get("files", []):
            if not (model_dir / fname).exists():
                errors.append(f"{model_dir.name}: listed file '{fname}' not found")

        # Add registry metadata
        data["_registry"] = {
            "path": f"models/{model_dir.name}",
            "raw_url_prefix": f"https://raw.githubusercontent.com/GACWR/open-model-registry/main/models/{model_dir.name}",
        }

        models.append(data)

    if errors:
        print("Warnings:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)

    index = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_models": len(models),
        "models": models,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(index, f, indent=2)

    print(f"Generated {output_file} with {len(models)} models")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
