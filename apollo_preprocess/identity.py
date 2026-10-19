from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from PIL import Image

from apollo_preprocess.images import resize_width
from apollo_preprocess.models import HospitalConfig
from apollo_preprocess.value import choose_tier

logger = logging.getLogger("apollo_preprocess")

_BARCODE_TRIED = False
_BARCODE_OK = False
_DIGIT_FIX = str.maketrans({"O": "0", "I": "1", "L": "1", "S": "5", "B": "8"})


@dataclass
class IdentityResult:
    form_code: str | None = None
    form_title: str | None = None
    form_family: str | None = None
    value_tier: str | None = None
    sticker_ip: str | None = None
    sticker_uhid: str | None = None
    sticker_bed: str | None = None
    quarantine: bool = False
    disagreement: bool = False
    reasons: list[str] = field(default_factory=list)
    header_text: str = ""


def ensure_tesseract(cfg: HospitalConfig) -> str:
    import pytesseract

    if cfg.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = cfg.tesseract_cmd
    else:
        windows = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        from pathlib import Path

        if Path(windows).exists():
            pytesseract.pytesseract.tesseract_cmd = windows
    version = str(pytesseract.get_tesseract_version())
    return version


def _upscale(image: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return image
    height, width = image.shape[:2]
    return resize_width(image, width * factor)


def ocr_image(image: np.ndarray, cfg: HospitalConfig, psm: int | None = None) -> str:
    import pytesseract

    psm = cfg.identity.psm if psm is None else psm
    config = f"--oem {cfg.identity.oem} --psm {psm}"
    try:
        text = pytesseract.image_to_string(
            Image.fromarray(_upscale(image, cfg.identity.upscale)),
            lang=cfg.identity.lang,
            config=config,
            timeout=cfg.identity.timeout_sec,
        )
    except Exception as exc:
        logger.info("ocr_failed error=%s", exc)
        return ""
    return text or ""


def read_header(image: np.ndarray, cfg: HospitalConfig) -> str:
    height = image.shape[0]
    strip = image[: max(1, int(height * cfg.identity.header_frac)), :]
    text = ocr_image(strip, cfg, cfg.identity.psm)
    from apollo_preprocess.value import match_forms

    if not match_forms(text, cfg):
        extra = ocr_image(strip, cfg, cfg.identity.psm_fallback)
        text = f"{text}\n{extra}"
    return text


def _sticker_crop(image: np.ndarray, cfg: HospitalConfig) -> np.ndarray:
    height, width = image.shape[:2]
    crop_h = max(1, int(height * cfg.identity.sticker_height_frac))
    crop_w = max(1, int(width * cfg.identity.sticker_width_frac))
    return image[:crop_h, width - crop_w :]


def _barcodes(image: np.ndarray) -> list[str]:
    global _BARCODE_OK, _BARCODE_TRIED
    if not _BARCODE_TRIED:
        _BARCODE_TRIED = True
        try:
            from pyzbar.pyzbar import decode as _decode

            _BARCODE_OK = True
        except Exception as exc:
            logger.info("barcode_unavailable error=%s", exc)
            _BARCODE_OK = False
    if not _BARCODE_OK:
        return []
    from pyzbar.pyzbar import decode

    try:
        rows = decode(Image.fromarray(image))
    except Exception as exc:
        logger.info("barcode_failed error=%s", exc)
        return []
    values: list[str] = []
    for row in rows:
        try:
            values.append(row.data.decode("utf-8", errors="ignore"))
        except Exception:
            continue
    return values


def _unique(items: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for item in items:
        seen.setdefault(item, None)
    return list(seen)


def _ip_digit_distance(left: str, right: str) -> int:
    left_digits = left[-6:]
    right_digits = right[-6:]
    return sum(a != b for a, b in zip(left_digits, right_digits)) + abs(len(left_digits) - len(right_digits))


def extract_ips(text: str) -> list[str]:
    found = re.findall(r"CSNIP\d{6}", text.upper())
    compact = re.sub(r"[^A-Z0-9]", "", text.upper())
    for match in re.finditer(r"CSNIP([0-9OILSB]{6})", compact):
        digits = match.group(1).translate(_DIGIT_FIX)
        if digits.isdigit():
            found.append(f"CSNIP{digits}")
    return _unique(found)


def _clean_num(num: str) -> str:
    match = re.fullmatch(r"([0-9OILSB]+)([A-Z]?)", num.upper())
    if not match:
        return num.upper()
    return match.group(1).translate(_DIGIT_FIX) + match.group(2)


@lru_cache(maxsize=32)
def _compile(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, flags=re.IGNORECASE)


def extract_form_code(text: str, cfg: HospitalConfig) -> str | None:
    folded = text.upper().replace("–", "-").replace("—", "-")
    compact = re.sub(r"[^A-Z0-9]", "", folded)
    for candidate in (folded, compact):
        for pattern in cfg.form_code_patterns:
            match = _compile(pattern).search(candidate)
            if not match:
                continue
            dept = match.group("dept").upper()
            num = _clean_num(match.group("num"))
            if dept not in cfg.form_code_depts:
                continue
            return f"AH-QF-{dept}-{num}"
    depts = "|".join(cfg.form_code_depts)
    match = re.search(rf"A[S]?H(?:QF|QRF|RF|F)({depts})(\d{{2,4}}[A-Z]?)", compact)
    if match:
        return f"AH-QF-{match.group(1)}-{_clean_num(match.group(2))}"
    return None


def extract_uhid(text: str) -> str | None:
    match = re.search(r"ASH\s*([0-9OIL])\s*[.\s]\s*([0-9OIL]{5,})", text.upper())
    if not match:
        return None
    prefix = match.group(1).translate(_DIGIT_FIX)
    rest = match.group(2).translate(_DIGIT_FIX)
    return f"ASH{prefix}.{rest}"


def extract_bed(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text.upper())
    if not match:
        return None
    return match.group(1)


def identify(
    image: np.ndarray,
    header_text: str,
    session_ip: str,
    cfg: HospitalConfig,
) -> IdentityResult:
    sticker_img = _sticker_crop(image, cfg)
    sticker_text = ocr_image(sticker_img, cfg, cfg.identity.psm)
    barcodes = _barcodes(sticker_img) + _barcodes(image[: max(1, int(image.shape[0] * cfg.identity.sticker_height_frac)), :])
    combined = "\n".join([header_text, sticker_text, *barcodes])
    form_code = extract_form_code(combined, cfg)
    tier, title, family, disagree = choose_tier(combined, form_code, cfg)
    sticker_ips = extract_ips("\n".join([sticker_text, *barcodes]))
    header_ips = extract_ips(header_text)
    result = IdentityResult(
        form_code=form_code,
        form_title=title,
        form_family=family,
        value_tier=tier,
        sticker_uhid=extract_uhid(combined),
        sticker_bed=extract_bed(combined, cfg.sticker_bed_pattern),
        disagreement=disagree,
        header_text=header_text,
    )
    ips = _unique(sticker_ips + header_ips)
    if session_ip in ips:
        result.sticker_ip = session_ip
    elif ips:
        closest = min(ips, key=lambda ip: _ip_digit_distance(ip, session_ip))
        if _ip_digit_distance(closest, session_ip) <= 1:
            result.sticker_ip = session_ip
            result.reasons.append("sticker_ip_fuzzy")
        else:
            result.sticker_ip = closest
            result.quarantine = True
            result.reasons.append("sticker_ip_mismatch")
    else:
        result.reasons.append("sticker_unreadable")
    if disagree:
        result.reasons.append("form_signal_disagreement")
        result.value_tier = None
    return result
