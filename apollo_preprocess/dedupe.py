from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from apollo_preprocess.images import to_gray
from apollo_preprocess.ingest import WorkingPage
from apollo_preprocess.models import HospitalConfig

logger = logging.getLogger("apollo_preprocess")


@dataclass
class _Features:
    gray: np.ndarray
    points: np.ndarray
    descriptors: np.ndarray | None


def _features(image: np.ndarray, cfg: HospitalConfig) -> _Features:
    gray = to_gray(image)
    scale = cfg.dedupe.compare_long_side / max(gray.shape[:2])
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    sift = cv2.SIFT_create(nfeatures=cfg.dedupe.sift_features)
    keypoints, descriptors = sift.detectAndCompute(enhanced, None)
    points = np.float32([keypoint.pt for keypoint in keypoints]).reshape(-1, 2)
    return _Features(gray, points, descriptors)


def _ink(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)


@dataclass
class _Comparison:
    early_kept: float
    late_known: float
    early_blob: int
    late_blob: int
    inliers: int
    overlap: float


def _largest_blob(unmatched: np.ndarray) -> int:
    if not unmatched.any():
        return 0
    grouped = cv2.dilate(unmatched.astype(np.uint8), np.ones((9, 15), np.uint8))
    count, labels = cv2.connectedComponents(grouped, connectivity=8)
    sizes = np.bincount(labels[unmatched], minlength=count)
    sizes[0] = 0
    return int(sizes.max())


def _containment(early: _Features, late: _Features, cfg: HospitalConfig) -> _Comparison | None:
    if early.descriptors is None or late.descriptors is None:
        return None
    if len(early.descriptors) < 8 or len(late.descriptors) < 8:
        return None
    cv2.setRNGSeed(0)
    matcher = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 4}, {"checks": 48})
    pairs = matcher.knnMatch(late.descriptors, early.descriptors, k=2)
    good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < cfg.dedupe.match_ratio * pair[1].distance]
    if len(good) < cfg.dedupe.min_inliers:
        return None
    source = late.points[[match.queryIdx for match in good]]
    target = early.points[[match.trainIdx for match in good]]
    homography, mask = cv2.findHomography(source, target, cv2.RANSAC, cfg.dedupe.ransac_px)
    if homography is None or mask is None:
        return None
    inliers = int(mask.sum())
    if inliers < cfg.dedupe.min_inliers:
        return None
    height, width = early.gray.shape[:2]
    warped = cv2.warpPerspective(late.gray, homography, (width, height), borderValue=255)
    valid = cv2.warpPerspective(np.full(late.gray.shape[:2], 255, np.uint8), homography, (width, height))
    border = max(1, cfg.dedupe.border_px)
    valid = cv2.erode(valid, np.ones((border * 2 + 1, border * 2 + 1), np.uint8))
    hull = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(target[mask.ravel() > 0]).astype(np.int32), 255)
    region = (valid > 0) & (hull > 0)
    overlap = float(region.mean())
    if overlap < cfg.dedupe.min_overlap_frac:
        return None
    size = cfg.dedupe.dilation_px * 2 + 1
    kernel = np.ones((size, size), np.uint8)
    ink_early = (_ink(early.gray) > 0) & region
    ink_late = (_ink(warped) > 0) & region
    near_early = cv2.dilate(ink_early.astype(np.uint8), kernel) > 0
    near_late = cv2.dilate(ink_late.astype(np.uint8), kernel) > 0
    early_total = int(ink_early.sum())
    late_total = int(ink_late.sum())
    return _Comparison(
        early_kept=float((ink_early & near_late).sum()) / early_total if early_total else 1.0,
        late_known=float((ink_late & near_early).sum()) / late_total if late_total else 1.0,
        early_blob=_largest_blob(ink_early & ~near_late),
        late_blob=_largest_blob(ink_late & ~near_early),
        inliers=inliers,
        overlap=overlap,
    )


def _relation(early: _Features, late: _Features, cfg: HospitalConfig) -> tuple[str, str]:
    found = _containment(early, late, cfg)
    if found is None:
        return "different", "no_alignment"
    detail = (
        f"inliers={found.inliers} overlap={found.overlap:.2f} early_kept={found.early_kept:.3f} "
        f"late_known={found.late_known:.3f} early_blob={found.early_blob} late_blob={found.late_blob}"
    )
    floor = cfg.dedupe.ink_contain_min
    blob = cfg.dedupe.new_ink_blob_px
    lost = found.early_kept < floor or found.early_blob >= blob
    added = found.late_known < floor or found.late_blob >= blob
    if not lost and not added:
        return "duplicate", detail
    if added and not lost:
        return "later_superset", detail
    if lost and not added:
        return "erasure", detail
    return "different", detail


def _mark_duplicate(page: WorkingPage, keeper: WorkingPage, reason: str) -> None:
    page.status = "duplicate_of"
    page.terminal = True
    page.related_page_id = keeper.page_id
    page.reasons.append(reason)


def _keeper_rank(page: WorkingPage) -> tuple:
    return (page.quality.blur or 0.0, page.scan_time, page.source_page, page.page_id)


def _latest_rank(page: WorkingPage) -> tuple:
    return (page.scan_time, page.quality.blur or 0.0, page.source_page, page.page_id)


def _time_order(page: WorkingPage) -> tuple:
    return (page.scan_time, page.source_pdf, page.source_page, page.page_id)


def _exact(pages: list[WorkingPage]) -> None:
    groups: dict[str, list[WorkingPage]] = {}
    for page in pages:
        if page.terminal or page.image is None:
            continue
        groups.setdefault(page.sha256, []).append(page)
    for group in groups.values():
        if len(group) < 2:
            continue
        keeper = max(group, key=_keeper_rank)
        for page in group:
            if page is keeper:
                continue
            _mark_duplicate(page, keeper, "exact_hash")


def _comparable(first: WorkingPage, second: WorkingPage) -> bool:
    if first.sticker_ip and second.sticker_ip and first.sticker_ip != second.sticker_ip:
        return False
    return True


def _relation_between(first: WorkingPage, second: WorkingPage, relations: dict[tuple[str, str], str]) -> str:
    return relations.get((first.page_id, second.page_id)) or relations.get((second.page_id, first.page_id)) or "different"


def _collapse_duplicates(cluster: list[WorkingPage], relations: dict[tuple[str, str], str]) -> list[WorkingPage]:
    groups: list[list[WorkingPage]] = []
    for page in sorted(cluster, key=_time_order):
        for group in groups:
            if all(_relation_between(page, other, relations) == "duplicate" for other in group):
                group.append(page)
                break
        else:
            groups.append([page])
    survivors: list[WorkingPage] = []
    for group in groups:
        keeper = max(group, key=_latest_rank)
        for page in group:
            if page is not keeper:
                _mark_duplicate(page, keeper, "near_duplicate")
        survivors.append(keeper)
    return survivors


def _resolve_cluster(cluster: list[WorkingPage], relations: dict[tuple[str, str], str]) -> None:
    ordered = sorted(_collapse_duplicates(cluster, relations), key=_time_order)
    if len(ordered) < 2:
        return
    latest = ordered[-1]
    duplicates: list[WorkingPage] = []
    erasure = False
    for page in ordered[:-1]:
        relation = _relation_between(page, latest, relations)
        if relation == "erasure":
            erasure = True
            break
        if relation == "later_superset":
            page.status = "superseded_by"
            page.terminal = True
            page.related_page_id = latest.page_id
            page.reasons.append("running_chart")
        elif relation == "duplicate":
            duplicates.append(page)
    if erasure:
        for page in ordered:
            page.status = "review_possible_erasure"
            page.terminal = True
            page.related_page_id = latest.page_id if page is not latest else ordered[-2].page_id
            if "ink_removed" not in page.reasons:
                page.reasons.append("ink_removed")
        return
    candidates = [latest, *duplicates]
    keeper = max(candidates, key=_latest_rank)
    for page in candidates:
        if page is keeper or page.terminal:
            continue
        _mark_duplicate(page, keeper, "near_duplicate")
    for page in ordered:
        if page.status == "superseded_by":
            page.related_page_id = keeper.page_id


def dedupe(pages: list[WorkingPage], cfg: HospitalConfig) -> None:
    _exact(pages)
    active = [page for page in pages if not page.terminal and page.image is not None]
    features = {page.page_id: _features(page.image, cfg) for page in active if page.image is not None}
    groups: dict[str, list[WorkingPage]] = {}
    for page in active:
        groups.setdefault(page.ip_number, []).append(page)
    for group in groups.values():
        group.sort(key=_time_order)
        parent = {page.page_id: page.page_id for page in group}

        def _root(page_id: str) -> str:
            while parent[page_id] != page_id:
                parent[page_id] = parent[parent[page_id]]
                page_id = parent[page_id]
            return page_id

        relations: dict[tuple[str, str], str] = {}
        for index, earlier in enumerate(group):
            for later in group[index + 1 :]:
                if not _comparable(earlier, later):
                    continue
                relation, detail = _relation(features[earlier.page_id], features[later.page_id], cfg)
                relations[(earlier.page_id, later.page_id)] = relation
                if relation != "different" or detail != "no_alignment":
                    logger.info(
                        "dedupe page=%s other=%s relation=%s %s",
                        earlier.page_id,
                        later.page_id,
                        relation,
                        detail,
                    )
                if relation != "different":
                    parent[_root(later.page_id)] = _root(earlier.page_id)
        clusters: dict[str, list[WorkingPage]] = {}
        for page in group:
            clusters.setdefault(_root(page.page_id), []).append(page)
        for cluster in clusters.values():
            _resolve_cluster(cluster, relations)
