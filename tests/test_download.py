"""Athena bundle download: gzip-wrapped zips, safe extraction, no URL in errors."""

from __future__ import annotations

import gzip
import http.server
import io
import threading
import zipfile

import pytest

from omop_agent.vocab.download import download_bundle, fetch_athena, unpack_bundle


def _bundle_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("CONCEPT.csv", "concept_id\tconcept_name\n8507\tMALE\n")
        z.writestr("VOCABULARY.csv", "vocabulary_id\n")
    return buf.getvalue()


@pytest.fixture
def server():
    payloads: dict[str, bytes] = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = payloads.get(self.path)
            if body is None:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    httpd = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}", payloads
    httpd.shutdown()


@pytest.mark.parametrize("wrap", [False, True])
def test_download_plain_and_gzip_wrapped(server, tmp_path, wrap):
    base, payloads = server
    payloads["/zip/secret-token"] = gzip.compress(_bundle_bytes()) if wrap else _bundle_bytes()
    out = download_bundle(f"{base}/zip/secret-token", tmp_path, log=lambda m: None)
    assert zipfile.is_zipfile(out)
    target = unpack_bundle(out, tmp_path / "vocab", log=lambda m: None)
    assert (target / "CONCEPT.csv").read_text().startswith("concept_id")
    assert not out.exists()


def test_errors_never_echo_the_link(server, tmp_path, monkeypatch):
    base, _ = server
    monkeypatch.setenv("ATHENA_DOWNLOAD_URL", f"{base}/zip/secret-token")
    logs: list[str] = []
    with pytest.raises(RuntimeError) as e:
        fetch_athena(tmp_path, log=logs.append)
    assert "secret-token" not in str(e.value) and all("secret-token" not in m for m in logs)
    monkeypatch.delenv("ATHENA_DOWNLOAD_URL")
    with pytest.raises(RuntimeError, match="ATHENA_DOWNLOAD_URL is not set"):
        fetch_athena(tmp_path)


def test_zip_slip_is_refused(tmp_path):
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("../evil.csv", "x")
    with pytest.raises(RuntimeError, match="outside"):
        unpack_bundle(bad, tmp_path / "vocab", log=lambda m: None)
