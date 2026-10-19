from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from apollo_preprocess.images import to_gray
from apollo_preprocess.models import HospitalConfig


@dataclass
class Quad:
    pts: np.ndarray
    area_frac: float


def order_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    total = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(total)]
    ordered[2] = pts[np.argmax(total)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered


def _aspect(pts: np.ndarray) -> float:
    width_top = float(np.linalg.norm(pts[1] - pts[0]))
    width_bot = float(np.linalg.norm(pts[2] - pts[3]))
    height_left = float(np.linalg.norm(pts[3] - pts[0]))
    height_right = float(np.linalg.norm(pts[2] - pts[1]))
    width = max(width_top, width_bot)
    height = max(height_left, height_right)
    if height < 1 or width < 1:
        return 0.0
    ratio = width / height
    return ratio if ratio >= 1 else 1 / ratio


def _masks(shape: tuple[int, ...], a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    height, width = shape[:2]
    scale = 240 / max(width, 1)
    ws = max(1, int(round(width * scale)))
    hs = max(1, int(round(height * scale)))
    ma = np.zeros((hs, ws), np.uint8)
    mb = np.zeros((hs, ws), np.uint8)
    cv2.fillConvexPoly(ma, np.round(a * scale).astype(np.int32), 1)
    cv2.fillConvexPoly(mb, np.round(b * scale).astype(np.int32), 1)
    inter = int(np.logical_and(ma, mb).sum())
    union = int(np.logical_or(ma, mb).sum())
    smaller = max(1, min(int(ma.sum()), int(mb.sum())))
    return inter / max(1, union), inter / smaller


def warp(image: np.ndarray, pts: np.ndarray) -> np.ndarray | None:
    src = order_points(pts)
    width = int(round(max(np.linalg.norm(src[1] - src[0]), np.linalg.norm(src[2] - src[3]))))
    height = int(round(max(np.linalg.norm(src[3] - src[0]), np.linalg.norm(src[2] - src[1]))))
    if width < 40 or height < 40:
        return None
    if width > image.shape[1] * 1.2 or height > image.shape[0] * 1.2:
        return None
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def find_quads(image: np.ndarray, cfg: HospitalConfig) -> list[Quad]:
    gray = to_gray(image)
    ksize = cfg.crop.blur_ksize if cfg.crop.blur_ksize % 2 else cfg.crop.blur_ksize + 1
    blur = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    edges = cv2.Canny(blur, cfg.crop.canny_low, cfg.crop.canny_high)
    kernel = np.ones((cfg.crop.dilate_px, cfg.crop.dilate_px), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=cfg.crop.dilate_iters)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    frame = float(gray.shape[0] * gray.shape[1])
    found: list[Quad] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        frac = area / frame
        if frac < cfg.crop.min_quad_area_frac:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, cfg.crop.approx_epsilon_frac * peri, True)
        pts: np.ndarray | None = None
        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = approx.reshape(4, 2).astype(np.float32)
        else:
            hull = cv2.convexHull(contour)
            hull_area = float(cv2.contourArea(hull))
            if hull_area <= 0 or area / hull_area < cfg.crop.min_solidity:
                continue
            box = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)
            rect_area = float(cv2.contourArea(box))
            if rect_area <= 0 or area / rect_area < cfg.crop.min_rect_fill:
                continue
            pts = box
        pts = order_points(pts)
        aspect = _aspect(pts)
        if aspect < cfg.crop.min_aspect or aspect > cfg.crop.max_aspect:
            continue
        found.append(Quad(pts, frac))
    found.sort(key=lambda quad: quad.area_frac, reverse=True)
    kept: list[Quad] = []
    for quad in found:
        if any(_masks(image.shape, quad.pts, prior.pts)[0] > cfg.crop.quad_dedupe_iou for prior in kept):
            continue
        kept.append(quad)
    return kept


def split_documents(image: np.ndarray, cfg: HospitalConfig) -> list[tuple[np.ndarray, float]]:
    quads = [quad for quad in find_quads(image, cfg) if quad.area_frac <= cfg.crop.max_quad_area_frac]
    if not quads:
        return [(image, 1.0)]
    quads.sort(key=lambda quad: quad.area_frac, reverse=True)
    primary = quads[0]
    partner: Quad | None = None
    for quad in quads[1:]:
        _iou, iom = _masks(image.shape, primary.pts, quad.pts)
        if iom >= cfg.crop.overlap_iom_min and quad.area_frac >= cfg.crop.min_second_doc_area_frac:
            partner = quad
            break
    if partner is None:
        warped = warp(image, primary.pts)
        short_side = 0 if warped is None else min(warped.shape[:2])
        if warped is None or short_side < cfg.quality.min_page_side_px or primary.area_frac < 0.55:
            return [(image, 1.0)]
        return [(warped, primary.area_frac)]
    larger, smaller = (primary, partner) if primary.area_frac >= partner.area_frac else (partner, primary)
    top_img = warp(image, smaller.pts)
    under_img = warp(image, larger.pts)
    if top_img is None:
        return [(image, 1.0)]
    pages = [(top_img, smaller.area_frac)]
    if under_img is not None:
        pages.append((under_img, larger.area_frac))
    return pages
