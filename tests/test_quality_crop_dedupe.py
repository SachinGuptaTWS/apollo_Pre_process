import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from apollo_preprocess.crop import split_documents
from apollo_preprocess.dedupe import dedupe
from apollo_preprocess.ingest import WorkingPage, ingest_pdf
from apollo_preprocess.models import load_config
from apollo_preprocess.quality import frame_reject_reason, measure, page_reject_reason

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config" / "apollo_chennai.yaml")
SAMPLE_FILE = ROOT / "tests" / "local_samples.json"
SAMPLES = json.loads(SAMPLE_FILE.read_text(encoding="utf-8")) if SAMPLE_FILE.exists() else {}


def _page(image: np.ndarray, page_id: str, scan_time: str, family: str | None = "billing_card") -> WorkingPage:
    return WorkingPage(
        page_id=page_id,
        source_pdf="unit.pdf",
        source_page=int(page_id[-1]),
        session_id="s",
        scan_time=scan_time,
        user="n",
        ip_number="CSNIP000001",
        image=image,
        sha256=page_id,
        form_family=family,
        form_title="BILLING CARD",
        value_tier="high",
        sticker_ip="CSNIP000001",
        quality=measure(image, CFG, 1.0),
    )


_ENTRIES = [
    "Dr Abdul Ghafur 26/09 visit",
    "Dr Ashwin 26/9 28/9 review",
    "IV line Rs 500 charged 1/10",
    "Sampling Rs 750 lab 2/10",
    "Physio session 3/10 done",
    "Diet consult 4/10 noted",
]


def _sheet(new_line: bool = False, erase_line: bool = False) -> np.ndarray:
    image = np.full((800, 600, 3), 236, np.uint8)
    cv2.putText(image, "APOLLO HOSPITALS", (150, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (20, 20, 20), 2)
    cv2.putText(image, "BILLING CARD  ASH-QRF-NS-01", (110, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    for y in range(130, 760, 60):
        cv2.line(image, (40, y), (560, y), (60, 60, 60), 2)
    cv2.line(image, (200, 130), (200, 730), (60, 60, 60), 2)
    for row, label in enumerate(("Consultant", "Visits", "Procedure", "Lab", "Pharmacy", "Others", "Diet", "Nursing", "Total", "Sign")):
        cv2.putText(image, label, (48, 168 + row * 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    entries = list(_ENTRIES[:4])
    if new_line:
        entries.append(_ENTRIES[4])
    for index, text in enumerate(entries):
        if erase_line and index == 2:
            continue
        cv2.putText(image, text, (212, 172 + index * 60), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.8, (120, 40, 20), 2)
    return image


def _rephotograph(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    src = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    dst = np.float32([[18, 10], [width - 6, 24], [width - 20, height - 8], [4, height - 20]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, matrix, (width, height), borderValue=(90, 90, 90))
    turned = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return cv2.convertScaleAbs(turned, alpha=0.9, beta=-8)


def test_grey_frames_in_sample_pdfs_are_blank():
    if not SAMPLES:
        pytest.skip("sample PDFs are not in the repository")
    for name, expected in SAMPLES.items():
        pages, _sessions = ingest_pdf(ROOT / name, CFG)
        blank = {page.source_page for page in pages if frame_reject_reason(page.image, CFG) == "blank"}
        assert blank == set(expected["blank_pages"])


def test_readable_page_is_not_blank_or_blur():
    image = _sheet()
    assert frame_reject_reason(image, CFG) is None
    metrics = measure(image, CFG, 1.0)
    assert page_reject_reason(image, metrics, CFG) is None


def test_heavy_blur_is_rejected():
    image = cv2.GaussianBlur(_sheet(), (51, 51), 0)
    metrics = measure(image, CFG, 1.0)
    assert page_reject_reason(image, metrics, CFG) == "blur"


def test_two_overlapping_rectangles_keep_the_smaller_on_top():
    image = np.full((900, 700, 3), 90, np.uint8)
    cv2.rectangle(image, (40, 30), (660, 860), (250, 230, 120), thickness=-1)
    cv2.rectangle(image, (40, 30), (660, 860), (20, 20, 20), thickness=4)
    cv2.rectangle(image, (120, 180), (560, 700), (250, 250, 250), thickness=-1)
    cv2.rectangle(image, (120, 180), (560, 700), (10, 10, 10), thickness=4)
    parts = split_documents(image, CFG)
    assert len(parts) == 2
    top, _top_area = parts[0]
    assert top.shape[0] < 860
    assert top.shape[1] < 660


def test_exact_and_near_duplicates_keep_the_sharper_copy():
    sharp = _sheet()
    soft = cv2.GaussianBlur(sharp, (3, 3), 0)
    pages = [
        _page(sharp, "p1", "2026-09-01T10:00:00"),
        _page(soft, "p2", "2026-09-02T10:00:00"),
    ]
    pages[0].sha256 = "same"
    pages[1].sha256 = "same"
    dedupe(pages, CFG)
    assert pages[0].status is None
    assert pages[1].status == "duplicate_of"
    assert pages[1].related_page_id == "p1"


def test_rephotographed_sheet_is_a_duplicate_without_identity():
    early = _page(_sheet(), "p1", "2026-09-01T10:00:00", family=None)
    late = _page(_rephotograph(_sheet()), "p2", "2026-09-05T10:00:00", family=None)
    dedupe([early, late], CFG)
    statuses = {early.status, late.status}
    assert statuses == {None, "duplicate_of"}


def test_different_sheets_of_one_form_are_kept():
    first = _sheet()
    second = np.full_like(first, 236)
    second[:110] = first[:110]
    cv2.putText(second, "Night shift handover notes", (60, 300), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.0, (120, 40, 20), 2)
    cv2.putText(second, "Patient stable, review at 6", (60, 420), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.0, (120, 40, 20), 2)
    pages = [
        _page(first, "p1", "2026-09-01T10:00:00"),
        _page(second, "p2", "2026-09-05T10:00:00"),
    ]
    dedupe(pages, CFG)
    assert pages[0].status is None
    assert pages[1].status is None


def test_running_chart_keeps_the_later_photo():
    early = _page(_sheet(), "p1", "2026-09-01T10:00:00")
    late = _page(_rephotograph(_sheet(new_line=True)), "p2", "2026-09-05T10:00:00")
    dedupe([early, late], CFG)
    assert early.status == "superseded_by"
    assert early.related_page_id == "p2"
    assert late.status is None


def test_removed_ink_is_an_erasure_review():
    early = _page(_sheet(), "p1", "2026-09-01T10:00:00")
    late = _page(_sheet(erase_line=True), "p2", "2026-09-05T10:00:00")
    dedupe([early, late], CFG)
    assert early.status == "review_possible_erasure"
    assert late.status == "review_possible_erasure"
