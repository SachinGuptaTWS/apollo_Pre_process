from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageOps

from apollo_preprocess.models import HospitalConfig, QualityMetrics


@dataclass
class SessionInfo:
    session_id: str
    ip_number: str
    uhid: str | None
    patient_name: str | None
    admission_date: str | None
    scan_time: str
    user: str
    hospital_name: str | None
    source_pdf: str
    cover_page: int


@dataclass
class WorkingPage:
    page_id: str
    source_pdf: str
    source_page: int
    session_id: str
    scan_time: str
    user: str
    ip_number: str
    image: np.ndarray | None
    sha256: str
    uhid: str | None = None
    patient_name: str | None = None
    admission_date: str | None = None
    status: str | None = None
    form_code: str | None = None
    form_title: str | None = None
    form_family: str | None = None
    value_tier: str | None = None
    rotation_applied: int = 0
    quality: QualityMetrics = field(default_factory=QualityMetrics)
    sticker_ip: str | None = None
    sticker_uhid: str | None = None
    sticker_bed: str | None = None
    related_page_id: str | None = None
    reasons: list[str] = field(default_factory=list)
    terminal: bool = False
    header_text: str = ""


def collect_pdfs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(item for item in path.iterdir() if item.suffix.lower() == ".pdf" and item.is_file())


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _search(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip()


def _iso(date_text: str, time_text: str) -> str:
    day, month, year = date_text.split("-")
    return f"{year}-{month}-{day}T{time_text}"


def _slug(user: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", user.lower()).strip("-") or "unknown"


def parse_cover_text(text: str, cfg: HospitalConfig, source_pdf: str, cover_page: int) -> SessionInfo | None:
    flat = _collapse(text)
    if cfg.cover.marker not in flat:
        return None
    ip_number = _search(cfg.cover.ip, flat)
    if not ip_number:
        return None
    ip_number = ip_number.upper()
    scan_date = _search(cfg.cover.scan_date, flat)
    scan_clock = _search(cfg.cover.scan_time, flat)
    admission = _search(cfg.cover.admission_date, flat)
    if not scan_clock:
        clock_match = re.search(
            r"(?<!ADMISSION )DATE:\s*\d{2}-\d{2}-\d{4}\s+(\d{2}:\d{2}:\d{2})",
            flat,
            flags=re.IGNORECASE,
        )
        if clock_match:
            scan_clock = clock_match.group(1)
    if scan_date:
        scan_time = _iso(scan_date, scan_clock or "00:00:00")
    elif admission:
        scan_time = _iso(admission, scan_clock or "00:00:00")
    else:
        scan_time = "1970-01-01T00:00:00"
    user = _search(cfg.cover.user, flat) or ""
    stamp = scan_time.replace("-", "").replace(":", "").replace("T", "-")
    session_id = f"{ip_number}-{stamp}-{_slug(user)}"
    return SessionInfo(
        session_id=session_id,
        ip_number=ip_number,
        uhid=_search(cfg.cover.uhid, flat),
        patient_name=_search(cfg.cover.patient_name, flat),
        admission_date=admission,
        scan_time=scan_time,
        user=user,
        hospital_name=_search(cfg.cover.hospital, flat),
        source_pdf=source_pdf,
        cover_page=cover_page,
    )


def _decode(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as raw:
        turned = ImageOps.exif_transpose(raw)
        if turned is None:
            turned = raw
        rgb = turned.convert("RGB")
        return np.asarray(rgb)


def _fallback_session(path: Path) -> SessionInfo:
    match = re.search(r"CSNIP\d{6}", path.stem.upper())
    ip_number = match.group(0) if match else "UNKNOWN"
    return SessionInfo(
        session_id=f"{ip_number}-unknown-session",
        ip_number=ip_number,
        uhid=None,
        patient_name=None,
        admission_date=None,
        scan_time="1970-01-01T00:00:00",
        user="",
        hospital_name=None,
        source_pdf=path.name,
        cover_page=0,
    )


def ingest_pdf(path: Path, cfg: HospitalConfig) -> tuple[list[WorkingPage], list[SessionInfo]]:
    document = pymupdf.open(path)
    pages: list[WorkingPage] = []
    sessions: list[SessionInfo] = []
    current: SessionInfo | None = None
    try:
        for index, page in enumerate(document):
            text = page.get_text("text") or ""
            cover = parse_cover_text(text, cfg, path.name, index + 1)
            if cover is not None:
                current = cover
                sessions.append(cover)
                continue
            infos = page.get_image_info(xrefs=True)
            if not infos:
                continue
            infos = sorted(infos, key=lambda item: int(item.get("width", 0)) * int(item.get("height", 0)), reverse=True)
            xref = int(infos[0]["xref"])
            extracted = document.extract_image(xref)
            blob = extracted["image"]
            session = current or _fallback_session(path)
            if current is None:
                current = session
                sessions.append(session)
            reasons: list[str] = []
            if session.cover_page == 0:
                reasons.append("missing_session")
            page_id = f"{session.ip_number}-{path.stem}-p{index + 1:04d}"
            pages.append(
                WorkingPage(
                    page_id=page_id,
                    source_pdf=path.name,
                    source_page=index + 1,
                    session_id=session.session_id,
                    scan_time=session.scan_time,
                    user=session.user,
                    ip_number=session.ip_number,
                    image=_decode(blob),
                    sha256=hashlib.sha256(blob).hexdigest(),
                    uhid=session.uhid,
                    patient_name=session.patient_name,
                    admission_date=session.admission_date,
                    reasons=reasons,
                )
            )
    finally:
        document.close()
    return pages, sessions
