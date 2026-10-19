from __future__ import annotations

import logging
import time
from collections import Counter
from pathlib import Path

import numpy as np
import typer

from apollo_preprocess.coverage import build_coverage
from apollo_preprocess.crop import split_documents
from apollo_preprocess.dedupe import dedupe
from apollo_preprocess.identity import ensure_tesseract, identify
from apollo_preprocess.ingest import SessionInfo, WorkingPage, collect_pdfs, ingest_pdf
from apollo_preprocess.models import load_config
from apollo_preprocess.orient import upright
from apollo_preprocess.quality import frame_reject_reason, measure, page_reject_reason
from apollo_preprocess.report import dump_model, manifest_for, sync_images, write_report
from apollo_preprocess.value import apply_value_status

app = typer.Typer(add_completion=False, no_args_is_help=True)
logger = logging.getLogger("apollo_preprocess")


class _Step:
    def __init__(self, step: str, page_id: str) -> None:
        self.step = step
        self.page_id = page_id
        self.started = 0.0

    def __enter__(self) -> _Step:
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        elapsed = (time.perf_counter() - self.started) * 1000
        logger.info("step=%s page=%s elapsed_ms=%.1f", self.step, self.page_id, elapsed)
        return False


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def _under_page(page: WorkingPage, image: np.ndarray, area: float, cfg) -> WorkingPage:
    under = WorkingPage(
        page_id=f"{page.page_id}-b",
        source_pdf=page.source_pdf,
        source_page=page.source_page,
        session_id=page.session_id,
        scan_time=page.scan_time,
        user=page.user,
        ip_number=page.ip_number,
        image=image,
        sha256=page.sha256,
        uhid=page.uhid,
        patient_name=page.patient_name,
        admission_date=page.admission_date,
        status="partial_occluded",
        terminal=True,
        related_page_id=page.page_id,
        reasons=["partial_occluded"],
        quality=measure(image, cfg, area),
    )
    page.related_page_id = under.page_id
    return under


def process_page(page: WorkingPage, cfg) -> list[WorkingPage]:
    produced = [page]
    try:
        if page.image is None:
            raise RuntimeError("no image")
        with _Step("quality_frame", page.page_id):
            page.quality = measure(page.image, cfg, 1.0)
            reason = frame_reject_reason(page.image, cfg)
            if reason:
                page.status = "rejected_quality"
                page.terminal = True
                page.reasons.append(reason)
                return produced
        with _Step("crop", page.page_id):
            parts = split_documents(page.image, cfg)
            page.image, area = parts[0]
            if len(parts) > 1:
                produced.append(_under_page(page, parts[1][0], parts[1][1], cfg))
        with _Step("quality_page", page.page_id):
            page.quality = measure(page.image, cfg, area)
            reason = page_reject_reason(page.image, page.quality, cfg)
            if reason:
                page.status = "rejected_quality"
                page.terminal = True
                page.reasons.append(reason)
                return produced
        with _Step("orient", page.page_id):
            page.image, page.rotation_applied, page.header_text = upright(page.image, cfg)
        with _Step("identity", page.page_id):
            found = identify(page.image, page.header_text, page.ip_number, cfg)
            page.form_code = found.form_code
            page.form_title = found.form_title
            page.form_family = found.form_family
            page.value_tier = found.value_tier
            page.sticker_ip = found.sticker_ip
            page.sticker_uhid = found.sticker_uhid
            page.sticker_bed = found.sticker_bed
            page.header_text = found.header_text
            for reason in found.reasons:
                if reason not in page.reasons:
                    page.reasons.append(reason)
            if found.quarantine:
                page.status = "quarantine_wrong_patient"
                page.terminal = True
    except Exception as exc:
        logger.exception("page_failed page=%s", page.page_id)
        page.status = "unknown_form_review"
        page.terminal = True
        page.reasons.append(f"processing_error:{type(exc).__name__}:{exc}"[:200])
    return produced


def run_preprocess(input_path: Path, out_dir: Path, config_path: Path) -> dict[str, int]:
    started = time.perf_counter()
    cfg = load_config(config_path)
    version = ensure_tesseract(cfg)
    logger.info("step=startup tesseract=%s config=%s elapsed_ms=0", version, config_path)
    pdfs = collect_pdfs(input_path)
    if not pdfs:
        raise FileNotFoundError(f"No PDF files in {input_path}")
    pages: list[WorkingPage] = []
    sessions: list[SessionInfo] = []
    for pdf in pdfs:
        with _Step("ingest", pdf.name):
            found_pages, found_sessions = ingest_pdf(pdf, cfg)
        logger.info("ingest pdf=%s photos=%s sessions=%s", pdf.name, len(found_pages), len(found_sessions))
        pages.extend(found_pages)
        sessions.extend(found_sessions)
    processed: list[WorkingPage] = []
    for page in pages:
        processed.extend(process_page(page, cfg))
    with _Step("dedupe", "*"):
        dedupe(processed, cfg)
    with _Step("value", "*"):
        apply_value_status(processed)
    by_ip: dict[str, list[WorkingPage]] = {}
    for page in processed:
        by_ip.setdefault(page.ip_number, []).append(page)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ip_number in sorted(by_ip):
        ip_pages = by_ip[ip_number]
        destination = out_dir / ip_number
        destination.mkdir(parents=True, exist_ok=True)
        manifest = manifest_for(ip_number, ip_pages)
        coverage = build_coverage(ip_number, ip_pages, sessions)
        dump_model(manifest, destination / "manifest.json")
        dump_model(coverage, destination / "coverage.json")
        write_report(destination / "report.html", ip_number, ip_pages)
        sync_images(destination / "pages", ip_pages, "png")
        sync_images(destination / "thumbs", ip_pages, "thumb")
        counts = Counter(page.status for page in ip_pages)
        logger.info("wrote ip=%s %s", ip_number, " ".join(f"{key}={counts[key]}" for key in sorted(counts)))
    elapsed = (time.perf_counter() - started) * 1000
    totals = Counter(page.status for page in processed)
    logger.info("step=finished pages=%s elapsed_ms=%.1f %s", len(processed), elapsed, dict(totals))
    return dict(totals)


@app.command()
def run(
    input: Path = typer.Option(..., "--input", exists=True, readable=True, help="PDF file or a folder of PDFs."),
    out: Path = typer.Option(..., "--out", help="Output directory."),
    config: Path = typer.Option(
        Path("config/apollo_chennai.yaml"),
        "--config",
        exists=True,
        readable=True,
        help="Hospital config YAML.",
    ),
) -> None:
    _configure_logging()
    run_preprocess(input, out, config)


if __name__ == "__main__":
    app()
