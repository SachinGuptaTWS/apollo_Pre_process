# apollo-preprocess

Local preprocessing for case-sheet photo PDFs before they are sent to paid OCR. It drops unusable, repeated and low-value photos and turns the rest upright. Everything runs on the local machine.

## Requirements

- Python 3.11+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) 5 with the `eng` language data. On Windows it is picked up from `C:\Program Files\Tesseract-OCR`; elsewhere set `tesseract_cmd` in the config or put it on `PATH`.
- Optional: the zbar library for sticker barcodes (`pyzbar`). Without it, stickers are read with OCR only.

```bash
pip install -e ".[dev]"
```

## Usage

Command line, for one PDF or a folder of PDFs:

```bash
preprocess --input path/to/pdfs --out out/
```

Web page for uploading a single PDF, at http://127.0.0.1:8765:

```bash
python -m apollo_preprocess.web
```

## Pipeline

1. **Ingest:** each "Patient Details" cover page starts a session, and the embedded photo on each following page is read without re-encoding.
2. **Quality:** blank, blurred, dark and glare-heavy photos are rejected.
3. **Crop:** finds the sheet outline and flags a sheet partly covered by another.
4. **Orient:** reads the page at 0°, 90°, 180° and 270° with Tesseract and keeps the turn with the most confidently read printed words.
5. **Identity:** reads the form title, form code and patient sticker. A sticker with another patient's IP number is quarantined.
6. **Dedupe:** matches photos of the same patient using SIFT features and a RANSAC homography, then compares the handwriting after alignment.
   - Same ink → duplicate (the latest photo is kept)
   - Extra ink in the later photo → running chart (the later photo is kept)
   - Ink missing from the later photo → sent for review
7. **Value:** forms with a `low` tier are not sent to OCR.

## Output

Each IP number gets its own folder under `out/`:

- `manifest.json`: status, reasons and related page for every photo
- `coverage.json`: forms captured in each session
- `report.html`
- `pages/`: upright PNGs
- `thumbs/`: thumbnails

Statuses: `ocr_required`, `duplicate_of`, `superseded_by`, `review_possible_erasure`, `skipped_low_value`, `rejected_quality`, `quarantine_wrong_patient`, `partial_occluded`, `unknown_form_review`.

## Configuration

`config/apollo_chennai.yaml` holds everything site-specific:

- cover-page patterns
- form titles and value tiers
- sticker patterns
- the thresholds for every step

For another hospital, add a new config file and pass it with `--config`.

## Tests

```bash
pytest
```

Tests that need real sample PDFs are skipped unless `tests/local_samples.json` exists. It is git-ignored and maps each PDF file name (placed in the project root) to its expected sessions and blank pages, for example:

```json
{"SAMPLE.pdf": {"sessions": {"1": [2, 3, 4]}, "blank_pages": [4]}}
```

Patient PDFs and pipeline output must never be committed. `*.pdf` and `out/` are ignored.

## Measuring accuracy

```bash
python scripts/make_labels.py --out-dir out --csv labels.csv
python scripts/score_labels.py --csv labels.csv
```

The first command writes a CSV to fill in by hand. The second reports precision and recall for the pages sent to OCR.
