from __future__ import annotations

import html
import json
from pathlib import Path

import cv2
import numpy as np

from apollo_preprocess.ingest import WorkingPage
from apollo_preprocess.models import Manifest, PageRecord, PageStatus, QualityMetrics, Sticker


def to_record(page: WorkingPage) -> PageRecord:
    status = page.status or PageStatus.unknown_form_review.value
    return PageRecord(
        page_id=page.page_id,
        source_pdf=page.source_pdf,
        source_page=page.source_page,
        session_id=page.session_id,
        scan_time=page.scan_time,
        user=page.user,
        status=PageStatus(status),
        form_code=page.form_code,
        form_title=page.form_title,
        value_tier=page.value_tier,
        rotation_applied=page.rotation_applied,
        quality=page.quality or QualityMetrics(),
        sticker=Sticker(ip=page.sticker_ip, uhid=page.sticker_uhid, bed=page.sticker_bed),
        related_page_id=page.related_page_id,
        reasons=list(page.reasons),
    )


def manifest_for(ip_number: str, pages: list[WorkingPage]) -> Manifest:
    ordered = sorted(pages, key=lambda page: (page.source_pdf, page.source_page, page.page_id))
    return Manifest(ip_number=ip_number, pages=[to_record(page) for page in ordered])


def dump_model(model, path: Path) -> None:
    payload = model.model_dump(mode="json")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_png(path: Path, image: np.ndarray) -> None:
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])


def save_thumb(path: Path, image: np.ndarray) -> None:
    height, width = image.shape[:2]
    scale = 240 / max(height, width)
    if scale < 1:
        thumb = cv2.resize(
            image,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        thumb = image
    bgr = cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])


def _cell(value: object) -> str:
    if value is None or value == "":
        return ""
    return html.escape(str(value))


def write_report(path: Path, ip_number: str, pages: list[WorkingPage]) -> None:
    ordered = sorted(pages, key=lambda page: (page.source_pdf, page.source_page, page.page_id))
    rows: list[str] = []
    for page in ordered:
        thumb = f"thumbs/{page.page_id}.jpg" if page.image is not None else ""
        image_html = f'<img src="{html.escape(thumb)}" alt="">' if thumb else ""
        quality = page.quality
        blur = "" if quality.blur is None else quality.blur
        bright = "" if quality.brightness is None else quality.brightness
        glare = "" if quality.glare is None else quality.glare
        area = "" if quality.page_area is None else quality.page_area
        rows.append(
            "<tr>"
            f"<td>{image_html}</td>"
            f"<td>{_cell(page.page_id)}</td>"
            f"<td>{_cell(page.source_pdf)}</td>"
            f"<td>{page.source_page}</td>"
            f"<td class='status status-{_cell(page.status)}'>{_cell(page.status)}</td>"
            f"<td>{_cell(page.form_code)}</td>"
            f"<td>{_cell(page.form_title)}</td>"
            f"<td>{_cell(page.value_tier)}</td>"
            f"<td>{page.rotation_applied}</td>"
            f"<td>{_cell(page.session_id)}</td>"
            f"<td>{_cell(page.scan_time)}</td>"
            f"<td>{_cell(page.user)}</td>"
            f"<td>{blur}</td><td>{bright}</td><td>{glare}</td><td>{area}</td>"
            f"<td>{_cell(page.sticker_ip)}</td>"
            f"<td>{_cell(page.sticker_uhid)}</td>"
            f"<td>{_cell(page.sticker_bed)}</td>"
            f"<td>{_cell(page.related_page_id)}</td>"
            f"<td>{_cell(', '.join(page.reasons))}</td>"
            "</tr>"
        )
    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(ip_number)} preprocess report</title>
<style>
body {{ font-family: Georgia, serif; margin: 24px; color: #1c1917; background: #faf7f2; }}
h1 {{ font-size: 22px; margin-bottom: 4px; }}
p {{ color: #57534e; }}
table {{ border-collapse: collapse; width: 100%; background: white; }}
th, td {{ border: 1px solid #e7e5e4; padding: 6px 8px; vertical-align: top; font-size: 13px; }}
th {{ background: #292524; color: white; position: sticky; top: 0; }}
img {{ width: 120px; height: auto; background: #e7e5e4; }}
.status-ocr_required {{ background: #dcfce7; }}
.status-rejected_quality {{ background: #fee2e2; }}
.status-duplicate_of, .status-superseded_by {{ background: #e0e7ff; }}
.status-partial_occluded, .status-review_possible_erasure, .status-unknown_form_review, .status-quarantine_wrong_patient {{ background: #fef3c7; }}
.status-skipped_low_value {{ background: #f5f5f4; }}
</style>
</head>
<body>
<h1>{html.escape(ip_number)}</h1>
<p>{len(ordered)} photo pages. Blur is the Laplacian variance at width 720; a higher number is sharper.</p>
<table>
<thead><tr>
<th>Thumb</th><th>Page</th><th>PDF</th><th>Src</th><th>Status</th><th>Form code</th><th>Title</th><th>Tier</th>
<th>Rotation</th><th>Session</th><th>Scan time</th><th>User</th>
<th>Blur</th><th>Brightness</th><th>Glare</th><th>Area</th>
<th>Sticker IP</th><th>UHID</th><th>Bed</th><th>Related</th><th>Reasons</th>
</tr></thead>
<tbody>
{"".join(rows)}
</tbody>
</table>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def sync_images(directory: Path, pages: list[WorkingPage], kind: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    keep: set[str] = set()
    for page in pages:
        if page.image is None:
            continue
        if kind == "png":
            if page.status != "ocr_required":
                continue
            name = f"{page.page_id}.png"
            save_png(directory / name, page.image)
        else:
            name = f"{page.page_id}.jpg"
            save_thumb(directory / name, page.image)
        keep.add(name)
    for existing in directory.iterdir():
        if existing.is_file() and existing.name not in keep:
            existing.unlink()
