from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a hand-label CSV from manifest.json files.")
    parser.add_argument("--out-dir", type=Path, default=Path("out"))
    parser.add_argument("--csv", type=Path, default=Path("labels.csv"))
    args = parser.parse_args()
    rows: list[dict[str, str]] = []
    manifests = sorted(args.out_dir.glob("*/manifest.json"))
    for manifest_path in manifests:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for page in payload["pages"]:
            rows.append(
                {
                    "page_id": page["page_id"],
                    "predicted_form": page.get("form_title") or page.get("form_code") or "",
                    "predicted_status": page.get("status") or "",
                    "true_form": "",
                    "true_status": "",
                }
            )
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["page_id", "predicted_form", "predicted_status", "true_form", "true_status"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
