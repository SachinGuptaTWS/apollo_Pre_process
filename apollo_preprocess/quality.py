from __future__ import annotations

import cv2
import numpy as np

from apollo_preprocess.images import resize_width, to_gray
from apollo_preprocess.models import HospitalConfig, QualityMetrics


def _round(value: float, places: int) -> float:
    return round(float(value), places)


def measure(image: np.ndarray, cfg: HospitalConfig, page_area: float) -> QualityMetrics:
    gray = to_gray(image)
    brightness = float(gray.mean())
    glare = float((gray >= cfg.quality.glare_value).mean())
    sample = resize_width(gray, cfg.quality.blur_eval_width)
    blur = float(cv2.Laplacian(sample, cv2.CV_64F).var())
    return QualityMetrics(
        blur=_round(blur, 2),
        brightness=_round(brightness, 2),
        glare=_round(glare, 4),
        page_area=_round(page_area, 4),
    )


def frame_reject_reason(image: np.ndarray, cfg: HospitalConfig) -> str | None:
    gray = to_gray(image)
    std = float(gray.std())
    edges = cv2.Canny(gray, cfg.quality.blank_canny_low, cfg.quality.blank_canny_high)
    edge = float((edges > 0).mean())
    if std < cfg.quality.blank_std_max and edge < cfg.quality.blank_edge_max:
        return "blank"
    if float(gray.mean()) < cfg.quality.dark_mean_min:
        return "dark"
    return None


def page_reject_reason(image: np.ndarray, metrics: QualityMetrics, cfg: HospitalConfig) -> str | None:
    height, width = image.shape[:2]
    if min(width, height) < cfg.quality.min_page_side_px:
        return "too_small"
    if metrics.blur is not None and metrics.blur < cfg.quality.blur_laplacian_min:
        return "blur"
    if metrics.glare is not None and metrics.glare > cfg.quality.glare_fraction_max:
        return "glare"
    return None
