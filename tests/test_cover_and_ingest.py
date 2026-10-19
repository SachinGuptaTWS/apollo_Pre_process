import json
from pathlib import Path

import pytest

from apollo_preprocess.ingest import ingest_pdf, parse_cover_text
from apollo_preprocess.models import load_config

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config" / "apollo_chennai.yaml")
SAMPLE_FILE = ROOT / "tests" / "local_samples.json"
SAMPLES = json.loads(SAMPLE_FILE.read_text(encoding="utf-8")) if SAMPLE_FILE.exists() else {}

COVERS = [
    (
        "Patient Details Hospital Name: ACI IP NUMBER: CSNIP100001 UHID: ASH1.0000000001 "
        "PATIENT NAME: Mr. TEST PATIENT ONE ADMISSION DATE:28-09-2026 SEX : Male "
        "DATE: 29-09-2026 18:16:31 USER NAME : Staff A SCAN TIME : 18:16:31",
        "CSNIP100001",
        "2026-09-29T18:16:31",
        "Staff A",
        "ASH1.0000000001",
    ),
    (
        "Patient Details Hospital Name: ACI IP NUMBER: CSNIP100002 UHID: ASH1.0000000002 "
        "PATIENT NAME: Mrs. TEST PATIENT TWO ADMISSION DATE: 24-09-2026 SEX: Female "
        "DATE: 24-09-2026 19:10:06 USER NAME: Staff A SCAN TIME: 19:10:06",
        "CSNIP100002",
        "2026-09-24T19:10:06",
        "Staff A",
        "ASH1.0000000002",
    ),
    (
        "Patient Details Hospital Name: ACI IP NUMBER: CSNIP100002 UHID: ASH1.0000000002 "
        "PATIENT NAME: Mrs. TEST PATIENT TWO ADMISSION DATE:24-09-2026 SEX : Female "
        "DATE: 29-09-2026 13:56:31 USER NAME : Staff B SCAN TIME : 13:56:31",
        "CSNIP100002",
        "2026-09-29T13:56:31",
        "Staff B",
        "ASH1.0000000002",
    ),
]


def test_cover_strings_from_the_sample_pdfs():
    for text, ip_number, scan_time, user, uhid in COVERS:
        session = parse_cover_text(text, CFG, "sample.pdf", 1)
        assert session is not None
        assert session.ip_number == ip_number
        assert session.scan_time == scan_time
        assert session.user == user
        assert session.uhid == uhid
        assert session.hospital_name == "ACI"


def test_tcpdf_footer_is_not_a_cover():
    assert parse_cover_text("Powered by TCPDF (www.tcpdf.org)", CFG, "sample.pdf", 18) is None


def test_every_photo_is_attached_to_the_cover_session():
    if not SAMPLES:
        pytest.skip("sample PDFs are not in the repository")
    for name, expected in SAMPLES.items():
        pages, sessions = ingest_pdf(ROOT / name, CFG)
        covers = {session.session_id: session.cover_page for session in sessions}
        got: dict[str, list[int]] = {}
        for page in pages:
            assert page.ip_number == Path(name).stem
            got.setdefault(str(covers[page.session_id]), []).append(page.source_page)
        assert got == expected["sessions"]
