"""Politely download public documents listed in a CSV (url column) into knowledge/raw.

Usage: fetch_documents.py <urls.csv> <raw_dir> <log.json> [file-number offset]. One request at a time,
2 s apart, plain client name, robots.txt respected; a blocked URL is skipped, never worked around."""

import csv
import json
import re
import sys
import time
import urllib.error
import urllib.request
import urllib.robotparser
from pathlib import Path
from urllib.parse import urlparse

src, out, log = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
BASE = int(sys.argv[4]) if len(sys.argv) > 4 else 0
UA = "sprout-research-fetch/1.0 (local non-commercial research; polite single-file fetch)"
MAXB = 30_000_000
robots = {}


def allowed(url):
    p = urlparse(url)
    host = p.scheme + "://" + p.netloc
    if host not in robots:
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(host + "/robots.txt")
        try:
            req = urllib.request.Request(
                host + "/robots.txt", headers={"User-Agent": UA}
            )
            rp.parse(
                urllib.request.urlopen(req, timeout=20)
                .read()
                .decode("utf-8", "replace")
                .splitlines()
            )
        except Exception:
            rp = None  # no readable robots.txt: no stated block
        robots[host] = rp
    rp = robots[host]
    return True if rp is None else rp.can_fetch(UA, url)


def slug(url):
    p = urlparse(url)
    s = re.sub(r"[^A-Za-z0-9]+", "-", (p.netloc.replace("www.", "") + p.path)).strip(
        "-"
    )
    return s[:90]


rows = list(csv.DictReader(src.open(encoding="utf-8")))
res = []
for i, r in enumerate(rows):
    url = r["url"]
    rec = {"url": url, "status": "", "file": "", "reason": ""}
    try:
        if not allowed(url):
            rec["reason"] = "robots.txt disallows"
            res.append(rec)
            continue
        req = urllib.request.Request(
            url, headers={"User-Agent": UA, "Accept": "text/html,application/pdf"}
        )
        with urllib.request.urlopen(req, timeout=40) as resp:
            ct = resp.headers.get("Content-Type", "")
            rec["status"] = resp.status
            final = resp.geturl()
            data = resp.read(MAXB + 1)
        if len(data) > MAXB:
            rec["reason"] = "over 30 MB"
        elif data[:5] == b"%PDF-":
            ext = ".pdf"
        elif "html" in ct.lower():
            ext = ".html"
        else:
            rec["reason"] = f"unexpected content type {ct}"
        if not rec["reason"]:
            name = f"{BASE + i + 1:03d}-{slug(url)}{ext}"
            (out / name).write_bytes(data)
            rec["file"] = name
            rec["final_url"] = final
            rec["bytes"] = len(data)
    except urllib.error.HTTPError as e:
        rec["status"] = e.code
        rec["reason"] = f"HTTP {e.code}"
    except Exception as e:
        rec["reason"] = f"{type(e).__name__}: {e}"[:200]
    res.append(rec)
    log.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(i + 1, rec["status"], rec["file"] or rec["reason"], flush=True)
    time.sleep(2)
