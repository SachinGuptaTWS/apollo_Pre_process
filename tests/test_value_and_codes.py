from pathlib import Path

from apollo_preprocess.identity import extract_form_code, extract_ips
from apollo_preprocess.models import load_config
from apollo_preprocess.value import choose_tier, match_forms

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config" / "apollo_chennai.yaml")


def test_noisy_drug_chart_title_is_high():
    tier, title, family, disagree = choose_tier("DRUG & DIET CH4RT  AH QF NS 008", None, CFG)
    assert tier == "high"
    assert family == "drug_diet"
    assert title == "DRUG & DIET CHART"
    assert disagree is False


def test_high_title_wins_over_low_title():
    tier, _title, family, _disagree = choose_tier("ADMISSION FORM lying on OPERATION NOTES", None, CFG)
    assert tier == "high"
    assert family == "operation_notes"


def test_low_admission_form():
    tier, _title, family, disagree = choose_tier("ADMISSION FORM", "AH-QF-NS-001", CFG)
    assert tier == "low"
    assert family == "admission_form"
    assert disagree is False


def test_unknown_when_nothing_matches():
    tier, title, family, disagree = choose_tier("grocery list", None, CFG)
    assert tier is None
    assert title is None
    assert family is None
    assert disagree is False


def test_shift_words_mark_handover():
    hits = match_forms("MORNING round EVENING round NIGHT round", CFG)
    assert any(hit.family == "nursing_handover" for hit in hits)


def test_form_code_patterns():
    assert extract_form_code("AH-QF-NS-008", CFG) == "AH-QF-NS-008"
    assert extract_form_code("AH QF NS 008", CFG) == "AH-QF-NS-008"
    assert extract_form_code("ASH-QF-OT-12a", CFG) == "AH-QF-OT-12A"
    assert extract_form_code("Ref AH-QF-FO-08 page", CFG) == "AH-QF-FO-08"
    assert extract_form_code("AHQFNS00B", CFG) == "AH-QF-NS-008"


def test_ip_ocr_repairs_letter_o():
    assert extract_ips("IPNO: CSNIP1O0002") == ["CSNIP100002"]
