from __future__ import annotations

import json
import re
import shutil
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from apollo_preprocess.cli import _configure_logging, run_preprocess

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "apollo_chennai.yaml"
TEMPLATE = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
JOBS = ROOT / "out" / "web"
TAKEN_DIR = ROOT / "out" / "taken"
HOST = "127.0.0.1"
PORT = 8765

TAKEN = {"ocr_required"}
REVIEW = {"review_possible_erasure", "quarantine_wrong_patient", "unknown_form_review", "partial_occluded"}

_QUALITY = {"blur": "photo too blurry", "blank": "blank frame", "dark": "photo too dark", "glare": "too much glare"}


def _page(results: str = "") -> bytes:
    html = TEMPLATE.replace("<!--RESULTS-->", results)
    return html.encode("utf-8")


def _esc(value: object) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _parse_pdf(content_type: str, body: bytes) -> tuple[str, bytes]:
    match = re.search(r"boundary=([^;]+)", content_type, flags=re.IGNORECASE)
    if not match:
        raise ValueError("Upload a PDF.")
    boundary = match.group(1).strip().strip('"')
    marker = ("--" + boundary).encode()
    for part in body.split(marker):
        if b'name="pdf"' not in part and b"name='pdf'" not in part:
            continue
        header, _, data = part.partition(b"\r\n\r\n")
        if data.endswith(b"\r\n"):
            data = data[:-2]
        header_text = header.decode("utf-8", errors="replace")
        filename = "upload.pdf"
        found = re.search(r'filename="([^"]*)"', header_text) or re.search(r"filename='([^']*)'", header_text)
        if found and found.group(1).strip():
            filename = Path(found.group(1)).name
        if not filename.lower().endswith(".pdf"):
            filename += ".pdf"
        return filename, data
    raise ValueError("Choose a PDF file.")


def _page_ref(number: object) -> str:
    return f'<a href="#page-{_esc(number)}">page {_esc(number)}</a>'


def _why(page: dict, related: object, related_status: object = None) -> str:
    status = page.get("status")
    reasons = page.get("reasons") or []
    form = page.get("form_title") or page.get("form_code") or "this form"
    if status == "duplicate_of":
        if "exact_hash" in reasons:
            lead = f"Identical image to {_page_ref(related)}"
        else:
            lead = f"Same sheet as {_page_ref(related)} (the latest photo of it)"
        if related_status in TAKEN:
            return f"{lead}, which was kept."
        return f"{lead}. That page itself was not sent to OCR ({_esc(related_status)})."
    if status == "superseded_by":
        return f"Older photo of the same sheet; {_page_ref(related)} has more entries and was kept."
    if status == "review_possible_erasure":
        return (
            f"Same sheet as {_page_ref(related)}, but writing present in an earlier photo is missing "
            "from a later one. Check by hand which copy is complete."
        )
    if status == "skipped_low_value":
        return f"{_esc(form)} is a low-value form and is not sent to OCR."
    if status == "rejected_quality":
        found = [_QUALITY[reason] for reason in reasons if reason in _QUALITY]
        return _esc("Unusable photo: " + (", ".join(found) or ", ".join(reasons)) + ".")
    if status == "quarantine_wrong_patient":
        return "Sticker shows a different patient's IP number."
    if status == "partial_occluded":
        return f"Sheet partly covered by {_page_ref(related)}."
    if status == "unknown_form_review":
        return _esc("Could not process: " + ", ".join(reasons))
    if status == "ocr_required" and page.get("copies"):
        copies = ", ".join(_page_ref(number) for number in page["copies"])
        return f"Also photographed as {copies} (those copies were dropped)."
    return ""


def _rows(pages: list[dict], decision: str, numbers: dict[str, object], statuses: dict[str, object]) -> str:
    body = []
    for page in pages:
        reasons = ", ".join(page.get("reasons") or [])
        form = page.get("form_title") or page.get("form_code") or ""
        src = page.get("image") or page.get("thumb") or ""
        css = "full" if page.get("image") else ""
        image = f'<img class="{css}" src="{_esc(src)}" alt="">' if src else ""
        number = page.get("source_page", "")
        related = numbers.get(page.get("related_page_id") or "", "")
        related_status = statuses.get(page.get("related_page_id") or "")
        body.append(
            f'<tr id="page-{_esc(number)}">'
            f"<td>{_esc(number)}</td>"
            f"<td>{image}</td>"
            f"<td>{_esc(decision)}</td>"
            f"<td>{_why(page, related, related_status)}</td>"
            f"<td>{_esc(form)}</td>"
            f"<td>{_esc(page.get('status'))}<br><small>{_esc(reasons)}</small></td>"
            "</tr>"
        )
    if not body:
        return "<p>None.</p>"
    return (
        "<table><thead><tr>"
        "<th>Page</th><th></th><th>Decision</th><th>Why</th><th>Form</th><th>Status</th>"
        "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def _render_job(payload: dict) -> str:
    if payload.get("error"):
        return f"<h2>Could not read that PDF</h2><p class='err'>{_esc(payload['error'])}</p>"
    blocks = [f"<p>{_esc(payload.get('filename') or '')}</p>"]
    for ip in payload.get("ips") or []:
        pages = ip.get("pages") or []
        numbers = {page["page_id"]: page.get("source_page", "") for page in pages if page.get("page_id")}
        statuses = {page["page_id"]: page.get("status") for page in pages if page.get("page_id")}
        for page in pages:
            if page.get("status") == "duplicate_of" and page.get("related_page_id"):
                for kept in pages:
                    if kept.get("page_id") == page["related_page_id"]:
                        kept.setdefault("copies", []).append(page.get("source_page"))
        taken = [page for page in pages if page.get("status") in TAKEN]
        review = [page for page in pages if page.get("status") in REVIEW]
        discarded = [page for page in pages if page.get("status") not in TAKEN | REVIEW]
        duplicates = sum(1 for page in discarded if page.get("status") in ("duplicate_of", "superseded_by"))
        low_value = sum(1 for page in discarded if page.get("status") == "skipped_low_value")
        unusable = sum(1 for page in discarded if page.get("status") == "rejected_quality")
        blocks.append(f"<h2>{_esc(ip.get('ip'))}</h2>")
        saved = ip.get("saved_dir") or ""
        blocks.append(
            f"<p>Taken {len(taken)}. Needs review {len(review)}. Discarded {len(discarded)} "
            f"({duplicates} repeat photos, {low_value} low-value forms, {unusable} unusable photos).</p>"
        )
        if saved:
            blocks.append(f"<p>Corrected images saved locally: <span class=\"path\">{_esc(saved)}</span></p>")
        blocks.append("<h2>Taken</h2>")
        blocks.append("<p>These are the corrected photos, after straightening.</p>")
        blocks.append(_rows(taken, "Taken", numbers, statuses))
        blocks.append("<h2>Needs review</h2>")
        blocks.append("<p>Not sent to OCR and not thrown away. Look at these and decide.</p>")
        blocks.append(_rows(review, "Review", numbers, statuses))
        blocks.append("<h2>Discarded</h2>")
        blocks.append(_rows(discarded, "Discarded", numbers, statuses))
    return "".join(blocks)


def _run_job(filename: str, data: bytes) -> str:
    job_id = uuid.uuid4().hex[:12]
    folder = JOBS / job_id
    folder.mkdir(parents=True, exist_ok=True)
    pdf_path = folder / filename
    pdf_path.write_bytes(data)
    try:
        run_preprocess(pdf_path, folder / "out", CONFIG)
        ips = []
        for manifest_path in sorted((folder / "out").glob("*/manifest.json")):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            ip_number = manifest["ip_number"]
            pages = []
            page_dir = folder / "out" / ip_number / "pages"
            local_dir = TAKEN_DIR / ip_number
            local_dir.mkdir(parents=True, exist_ok=True)
            for old in local_dir.glob("*.png"):
                old.unlink()
            for page in manifest["pages"]:
                thumb = folder / "out" / ip_number / "thumbs" / f"{page['page_id']}.jpg"
                full = page_dir / f"{page['page_id']}.png"
                image_url = ""
                if page["status"] in TAKEN and full.exists():
                    shutil.copy2(full, local_dir / full.name)
                    image_url = f"/job/{job_id}/{ip_number}/pages/{page['page_id']}.png"
                pages.append(
                    {
                        "page_id": page["page_id"],
                        "related_page_id": page.get("related_page_id"),
                        "source_page": page["source_page"],
                        "status": page["status"],
                        "form_code": page.get("form_code"),
                        "form_title": page.get("form_title"),
                        "reasons": page.get("reasons") or [],
                        "thumb": f"/job/{job_id}/{ip_number}/thumbs/{page['page_id']}.jpg" if thumb.exists() else "",
                        "image": image_url,
                    }
                )
            ips.append({"ip": ip_number, "pages": pages, "saved_dir": str(local_dir)})
        payload = {"filename": filename, "ips": ips}
    except Exception:
        payload = {"filename": filename, "error": traceback.format_exc(limit=4)}
    (folder / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    return job_id


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        print(f"[web] {self.address_string()} {fmt % args}")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path in ("/", ""):
            self._send(200, _page(), "text/html; charset=utf-8")
            return
        if path.startswith("/job/") and path.endswith("/"):
            path = path[:-1]
        job_match = re.fullmatch(r"/job/([0-9a-f]{12})", path)
        if job_match:
            result = JOBS / job_match.group(1) / "result.json"
            if not result.exists():
                self._send(404, _page("<p>That run was not found.</p>"), "text/html; charset=utf-8")
                return
            payload = json.loads(result.read_text(encoding="utf-8"))
            self._send(200, _page(_render_job(payload)), "text/html; charset=utf-8")
            return
        file_match = re.fullmatch(r"/job/([0-9a-f]{12})/([^/]+)/(thumbs|pages)/([^/]+\.(?:jpg|png))", path)
        if file_match:
            job_id, ip_number, kind, name = file_match.groups()
            target = (JOBS / job_id / "out" / ip_number / kind / name).resolve()
            root = (JOBS / job_id / "out").resolve()
            if root not in target.parents or not target.is_file():
                self._send(404, b"not found", "text/plain; charset=utf-8")
                return
            content_type = "image/png" if name.endswith(".png") else "image/jpeg"
            self._send(200, target.read_bytes(), content_type)
            return
        self._send(404, _page("<p>Not found.</p>"), "text/html; charset=utf-8")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/run":
            self._send(404, _page("<p>Not found.</p>"), "text/html; charset=utf-8")
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        try:
            filename, data = _parse_pdf(self.headers.get("Content-Type", ""), body)
            if not data.startswith(b"%PDF"):
                raise ValueError("That file is not a PDF.")
            job_id = _run_job(filename, data)
        except Exception as exc:
            self._send(400, _page(f"<p class='err'>{_esc(exc)}</p>"), "text/html; charset=utf-8")
            return
        self.send_response(303)
        self.send_header("Location", f"/job/{job_id}")
        self.send_header("Content-Length", "0")
        self.end_headers()


def main() -> None:
    _configure_logging()
    JOBS.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Open http://{HOST}:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
