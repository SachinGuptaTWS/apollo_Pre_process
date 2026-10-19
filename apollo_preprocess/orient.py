from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytesseract
from PIL import Image

from apollo_preprocess.images import rotate_rgb, to_gray
from apollo_preprocess.models import HospitalConfig

logger = logging.getLogger("apollo_preprocess")
_ANGLES = (0, 90, 180, 270)


def _ocr_gray(image: np.ndarray, cfg: HospitalConfig) -> np.ndarray:
    gray = to_gray(image)
    scale = cfg.orient.ocr_upscale
    longest = max(gray.shape[:2]) * scale
    if longest > cfg.orient.ocr_max_side_px:
        scale *= cfg.orient.ocr_max_side_px / longest
    if abs(scale - 1.0) < 1e-3:
        return gray
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interp)


def _printed_words(gray: np.ndarray, cfg: HospitalConfig) -> tuple[float, str]:
    data = pytesseract.image_to_data(
        Image.fromarray(gray),
        lang=cfg.identity.lang,
        config="--oem 1 --psm 6",
        output_type=pytesseract.Output.DICT,
        timeout=cfg.identity.timeout_sec,
    )
    score = 0.0
    header: list[str] = []
    header_limit = gray.shape[0] * 0.5
    for token, conf, top in zip(data["text"], data["conf"], data["top"]):
        try:
            conf_value = float(conf)
        except (TypeError, ValueError):
            continue
        token = (token or "").strip()
        letters = sum(char.isalpha() for char in token)
        if conf_value < cfg.orient.word_min_conf or letters < cfg.orient.word_min_letters:
            continue
        if not token[0].isalpha() or letters < len(token) - 2:
            continue
        score += conf_value - 70.0
        if float(top) < header_limit:
            header.append(token)
    return score, " ".join(header)


def upright(image: np.ndarray, cfg: HospitalConfig) -> tuple[np.ndarray, int, str]:
    gray = _ocr_gray(image, cfg)

    def _read(angle: int) -> tuple[float, str]:
        try:
            return _printed_words(rotate_rgb(gray, angle), cfg)
        except Exception as exc:
            logger.info("orient_ocr_failed angle=%s error=%s", angle, exc)
            return 0.0, ""

    with ThreadPoolExecutor(max_workers=len(_ANGLES)) as pool:
        results = dict(zip(_ANGLES, pool.map(_read, _ANGLES)))
    best = max(_ANGLES, key=lambda angle: results[angle][0])
    logger.info(
        "orient scores=%s chosen=%s",
        {angle: round(results[angle][0]) for angle in _ANGLES},
        best,
    )
    if results[best][0] >= cfg.orient.min_score:
        return rotate_rgb(image, best), best, results[best][1]
    return image, 0, results[0][1]
