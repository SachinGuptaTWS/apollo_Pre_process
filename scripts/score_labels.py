from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml

POSITIVE = "ocr_required"


def _load_high_titles(config_path: Path) -> set[str]:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    titles = {rule["title"].casefold() for rule in data["forms"] if rule["tier"] == "high"}
    titles.update(rule["id"].replace("_", " ").casefold() for rule in data["forms"] if rule["tier"] == "high")
    return titles


def _is_high(true_form: str, high_titles: set[str]) -> bool:
    folded = true_form.casefold().strip()
    if not folded:
        return False
    return folded in high_titles or any(title in folded for title in high_titles)


def _scores(rows: list[dict[str, str]], high_titles: set[str]) -> None:
    labeled = [row for row in rows if row["true_status"].strip()]
    if not labeled:
        print("No true_status values filled in.")
        return
    tp = fp = fn = 0
    high_tp = high_fn = 0
    for row in labeled:
        predicted = row["predicted_status"].strip()
        truth = row["true_status"].strip()
        if predicted == POSITIVE and truth == POSITIVE:
            tp += 1
        elif predicted == POSITIVE and truth != POSITIVE:
            fp += 1
        elif predicted != POSITIVE and truth == POSITIVE:
            fn += 1
        if _is_high(row["true_form"], high_titles) and truth == POSITIVE:
            if predicted == POSITIVE:
                high_tp += 1
            else:
                high_fn += 1
                print(f"HIGH miss {row['page_id']} predicted={predicted} form={row['true_form']}")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    high_recall = high_tp / (high_tp + high_fn) if high_tp + high_fn else 0.0
    print(f"labeled={len(labeled)} precision={precision:.3f} recall={recall:.3f}")
    print(f"high_tier_recall={high_recall:.3f} high_tp={high_tp} high_fn={high_fn}")
    if high_tp + high_fn and high_recall < 0.98:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a filled hand-label CSV.")
    parser.add_argument("--csv", type=Path, default=Path("labels.csv"))
    parser.add_argument("--config", type=Path, default=Path("config/apollo_chennai.yaml"))
    args = parser.parse_args()
    with args.csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    _scores(rows, _load_high_titles(args.config))


if __name__ == "__main__":
    main()
