from __future__ import annotations

from apollo_preprocess.ingest import SessionInfo, WorkingPage
from apollo_preprocess.models import Coverage, CoverageCapture, CoverageForm

_STATUS_ORDER = [
    "ocr_required",
    "review_possible_erasure",
    "unknown_form_review",
    "skipped_low_value",
    "duplicate_of",
    "superseded_by",
    "quarantine_wrong_patient",
    "rejected_quality",
    "partial_occluded",
]


def _capture_status(pages: list[WorkingPage]) -> str:
    present = {page.status for page in pages if page.status}
    ordered = [status for status in _STATUS_ORDER if status in present]
    return "+".join(ordered)


def build_coverage(ip_number: str, pages: list[WorkingPage], sessions: list[SessionInfo]) -> Coverage:
    own = [session for session in sessions if session.ip_number == ip_number]
    own.sort(key=lambda session: (session.scan_time, session.session_id))
    demo = own[0] if own else None
    groups: dict[tuple[str, str], list[WorkingPage]] = {}
    for page in pages:
        if page.ip_number != ip_number:
            continue
        if not page.form_code and not page.form_title:
            continue
        key = (page.form_code or "", page.form_title or "")
        groups.setdefault(key, []).append(page)
    forms: list[CoverageForm] = []
    for group in groups.values():
        by_session: dict[str, list[WorkingPage]] = {}
        for page in group:
            by_session.setdefault(page.session_id, []).append(page)
        captures: list[CoverageCapture] = []
        for session_id, session_pages in by_session.items():
            session_pages.sort(key=lambda page: (page.source_pdf, page.source_page, page.page_id))
            captures.append(
                CoverageCapture(
                    session_id=session_id,
                    scan_time=session_pages[0].scan_time,
                    page_ids=[page.page_id for page in session_pages],
                    status=_capture_status(session_pages),
                )
            )
        captures.sort(key=lambda capture: (capture.scan_time, capture.session_id))
        forms.append(
            CoverageForm(
                form_code=group[0].form_code,
                form_title=group[0].form_title,
                value_tier=group[0].value_tier,
                sessions=captures,
                service_dates=[],
            )
        )
    forms.sort(key=lambda form: (form.form_code or "", form.form_title or ""))
    return Coverage(
        ip_number=ip_number,
        uhid=demo.uhid if demo else None,
        patient_name=demo.patient_name if demo else None,
        admission_date=demo.admission_date if demo else None,
        forms=forms,
    )
