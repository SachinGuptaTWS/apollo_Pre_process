from __future__ import annotations

from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class PageStatus(str, Enum):
    ocr_required = "ocr_required"
    skipped_low_value = "skipped_low_value"
    rejected_quality = "rejected_quality"
    duplicate_of = "duplicate_of"
    superseded_by = "superseded_by"
    partial_occluded = "partial_occluded"
    quarantine_wrong_patient = "quarantine_wrong_patient"
    unknown_form_review = "unknown_form_review"
    review_possible_erasure = "review_possible_erasure"


class QualityMetrics(BaseModel):
    blur: float | None = None
    brightness: float | None = None
    glare: float | None = None
    page_area: float | None = None


class Sticker(BaseModel):
    ip: str | None = None
    uhid: str | None = None
    bed: str | None = None


class PageRecord(BaseModel):
    page_id: str
    source_pdf: str
    source_page: int
    session_id: str
    scan_time: str
    user: str
    status: PageStatus
    form_code: str | None = None
    form_title: str | None = None
    value_tier: str | None = None
    rotation_applied: int = 0
    quality: QualityMetrics = Field(default_factory=QualityMetrics)
    sticker: Sticker = Field(default_factory=Sticker)
    related_page_id: str | None = None
    reasons: list[str] = Field(default_factory=list)


class Manifest(BaseModel):
    ip_number: str
    pages: list[PageRecord]


class CoverageCapture(BaseModel):
    session_id: str
    scan_time: str
    page_ids: list[str]
    status: str


class CoverageForm(BaseModel):
    form_code: str | None = None
    form_title: str | None = None
    value_tier: str | None = None
    sessions: list[CoverageCapture]
    service_dates: list[str] = Field(default_factory=list)


class Coverage(BaseModel):
    ip_number: str
    uhid: str | None = None
    patient_name: str | None = None
    admission_date: str | None = None
    forms: list[CoverageForm]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CoverConfig(_Strict):
    marker: str
    hospital: str
    ip: str
    uhid: str
    patient_name: str
    admission_date: str
    scan_date: str
    scan_time: str
    user: str


class QualityConfig(_Strict):
    blank_std_max: float
    blank_edge_max: float
    blank_canny_low: int
    blank_canny_high: int
    blur_eval_width: int
    blur_laplacian_min: float
    dark_mean_min: float
    glare_value: int
    glare_fraction_max: float
    min_page_side_px: int


class CropConfig(_Strict):
    canny_low: int
    canny_high: int
    blur_ksize: int
    dilate_px: int
    dilate_iters: int
    approx_epsilon_frac: float
    min_quad_area_frac: float
    max_quad_area_frac: float
    min_aspect: float
    max_aspect: float
    min_solidity: float
    min_rect_fill: float
    overlap_iom_min: float
    min_second_doc_area_frac: float
    quad_dedupe_iou: float


class OrientConfig(_Strict):
    ocr_upscale: float
    ocr_max_side_px: int
    word_min_conf: float
    word_min_letters: int
    min_score: float


class IdentityConfig(_Strict):
    header_frac: float
    sticker_height_frac: float
    sticker_width_frac: float
    upscale: int
    oem: int
    psm: int
    psm_fallback: int
    lang: str
    timeout_sec: int


class DedupeConfig(_Strict):
    compare_long_side: int
    sift_features: int
    match_ratio: float
    ransac_px: float
    min_inliers: int
    min_overlap_frac: float
    ink_contain_min: float
    new_ink_blob_px: int
    dilation_px: int
    border_px: int


class FormRule(_Strict):
    id: str
    title: str
    tier: str
    phrases: list[str] = Field(default_factory=list)
    require_all: list[str] = Field(default_factory=list)


class HospitalConfig(_Strict):
    hospital_id: str
    hospital_names: list[str]
    tesseract_cmd: str | None = None
    cover: CoverConfig
    form_code_patterns: list[str]
    form_code_depts: list[str]
    sticker_ip_pattern: str
    sticker_uhid_pattern: str
    sticker_bed_pattern: str
    code_tiers: dict[str, str] = Field(default_factory=dict)
    quality: QualityConfig
    crop: CropConfig
    orient: OrientConfig
    identity: IdentityConfig
    dedupe: DedupeConfig
    forms: list[FormRule]


def load_config(path: Path) -> HospitalConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return HospitalConfig.model_validate(data)
