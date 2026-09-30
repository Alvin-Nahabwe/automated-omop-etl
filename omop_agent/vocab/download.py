"""Download and unpack an Athena vocabulary bundle.

Athena e-mails a link to the bundle once it is ready. The link grants access to your licensed
download, so treat it like a password: keep it in an environment variable (default
``ATHENA_DOWNLOAD_URL``) and never in the config file, logs or chat. This module never prints
it.

Athena may serve the zip gzip-encoded; the gzip layer is detected by its magic bytes and
removed while streaming.
"""

from __future__ import annotations

import os
import shutil
import urllib.request
import zipfile
import zlib
from pathlib import Path
from typing import Callable

GZIP_MAGIC = b"\x1f\x8b"
CHUNK = 1 << 20


def _redacted(url: str) -> str:
    from urllib.parse import urlparse

    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}/…"


def download_bundle(
    url: str,
    dest_dir: Path,
    log: Callable[[str], None] = print,
    timeout: float = 600.0,
) -> Path:
    """Stream the bundle to ``dest_dir/athena_bundle.zip`` (gzip layer removed). Returns the path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / "athena_bundle.zip"
    tmp = out.with_suffix(".part")
    req = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
    log(f"downloading the Athena bundle from {_redacted(url)}")
    written = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as fh:
            first = resp.read(CHUNK)
            gz = zlib.decompressobj(16 + zlib.MAX_WBITS) if first.startswith(GZIP_MAGIC) else None
            chunk = first
            while chunk:
                data = gz.decompress(chunk) if gz else chunk
                fh.write(data)
                written += len(data)
                if written // (100 << 20) != (written - len(data)) // (100 << 20):
                    log(f"  {written / 1e6:,.0f} MB")
                chunk = resp.read(CHUNK)
            if gz:
                fh.write(gz.flush())
    except Exception as e:  # never echo the URL
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"download failed ({type(e).__name__}); check that the Athena link has not expired") from None
    if not zipfile.is_zipfile(tmp):
        tmp.unlink(missing_ok=True)
        raise RuntimeError("the download is not a zip file; the Athena link may have expired or be incomplete")
    tmp.replace(out)
    log(f"  downloaded {out.stat().st_size / 1e6:,.0f} MB")
    return out


def unpack_bundle(bundle: Path, target: Path, log: Callable[[str], None] = print, keep_zip: bool = False) -> Path:
    """Extract the Athena CSVs into ``target`` (safely: no paths outside it). Returns ``target``."""
    target.mkdir(parents=True, exist_ok=True)
    root = target.resolve()
    with zipfile.ZipFile(bundle) as z:
        for info in z.infolist():
            dest = (target / info.filename).resolve()
            if root not in dest.parents and dest != root:
                raise RuntimeError(f"refusing to extract {info.filename!r} outside {target}")
            if info.is_dir():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, dest.open("wb") as dst:
                shutil.copyfileobj(src, dst, CHUNK)
    if not keep_zip:
        bundle.unlink()
    if not (target / "CONCEPT.csv").exists():
        raise RuntimeError(f"{target} has no CONCEPT.csv after unpacking: is this an Athena vocabulary bundle?")
    names = sorted(p.name for p in target.glob("*.csv"))
    log(f"  unpacked {len(names)} files into {target}")
    return target


def fetch_athena(dest: Path, url_env: str = "ATHENA_DOWNLOAD_URL", log: Callable[[str], None] = print) -> Path:
    url = os.environ.get(url_env, "").strip()
    if not url:
        raise RuntimeError(
            f"{url_env} is not set. Request a download at https://athena.ohdsi.org (Download tab), then put the "
            f"link from Athena's e-mail in the {url_env} environment variable."
        )
    bundle = download_bundle(url, dest, log)
    return unpack_bundle(bundle, dest, log)
