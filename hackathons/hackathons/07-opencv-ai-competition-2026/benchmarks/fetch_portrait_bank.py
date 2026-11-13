"""Fetch a small, licence-clean portrait bank from Wikimedia Commons.

Only files under CC0 / Public domain / CC BY 4.0 / CC BY 3.0 are kept.  Each
downloaded image is reduced to the single, largest YuNet detection and stored
as ``data/samples/faces/ext_<n>.jpg`` with full attribution in
``data/samples/faces/external_manifest.json``.

Run once with network access; the resulting crops are committed so that all
benchmarks and demos are offline and deterministic afterwards.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
FACE_DIR = ROOT / "data" / "samples" / "faces"
MODEL = ROOT / "data" / "models" / "face_detection_yunet_2023mar.onnx"
API = "https://commons.wikimedia.org/w/api.php?"
ALLOWED = {"CC0", "Public domain", "PD", "CC BY 4.0", "CC BY 3.0"}
CATEGORIES = ["Portrait photographs of women", "Portrait photographs of men"]


def api(params: dict) -> dict:
    req = urllib.request.Request(
        API + urllib.parse.urlencode(params),
        headers={"User-Agent": "VIGIL5-research/1.0 (local hackathon benchmark)"},
    )
    return json.load(urllib.request.urlopen(req, timeout=45))


def candidates() -> list[dict]:
    out = []
    for cat in CATEGORIES:
        data = api(
            {
                "action": "query",
                "generator": "categorymembers",
                "gcmtitle": f"Category:{cat}",
                "gcmtype": "file",
                "gcmlimit": "100",
                "prop": "imageinfo",
                "iiprop": "url|extmetadata",
                "iiurlwidth": "640",
                "format": "json",
            }
        )
        for page in data.get("query", {}).get("pages", {}).values():
            info = (page.get("imageinfo") or [{}])[0]
            meta = info.get("extmetadata", {})
            lic = meta.get("LicenseShortName", {}).get("value", "")
            if lic not in ALLOWED or not info.get("thumburl"):
                continue
            out.append(
                {
                    "title": page["title"],
                    "license": lic,
                    "artist": meta.get("Artist", {}).get("value", "unknown"),
                    "url": info["thumburl"],
                    "page": info.get("descriptionurl", ""),
                }
            )
    return out


def main() -> None:
    FACE_DIR.mkdir(parents=True, exist_ok=True)
    det = cv2.FaceDetectorYN_create(str(MODEL), "", (320, 320), 0.8, 0.3, 20)
    kept, seen = [], set()
    for cand in candidates():
        if cand["title"] in seen:
            continue
        seen.add(cand["title"])
        try:
            req = urllib.request.Request(cand["url"], headers={"User-Agent": "VIGIL5-research/1.0"})
            raw = urllib.request.urlopen(req, timeout=45).read()
            img = cv2.imdecode(cv2.UMat(__import__("numpy").frombuffer(raw, dtype="uint8")).get(), cv2.IMREAD_COLOR)
        except Exception as exc:  # network / decode failures are non-fatal
            print(f"[skip] {cand['title'][:50]}: {type(exc).__name__}")
            continue
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = min(1.0, 640 / max(h, w))
        if scale < 1:
            img = cv2.resize(img, None, fx=scale, fy=scale)
        det.setInputSize((img.shape[1], img.shape[0]))
        _, faces = det.detect(img)
        if faces is None or len(faces) != 1:
            continue
        face = faces[0]
        x, y, bw, bh = face[:4]
        side = max(bw, bh) * 1.45
        cx, cy = x + bw / 2, y + bh / 2
        x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        x1, y1 = int(min(img.shape[1], cx + side / 2)), int(min(img.shape[0], cy + side / 2))
        crop = img[y0:y1, x0:x1]
        if min(crop.shape[:2]) < 100:
            continue
        crop = cv2.resize(crop, (320, 320), interpolation=cv2.INTER_AREA)
        name = f"ext_{len(kept):02d}.jpg"
        cv2.imwrite(str(FACE_DIR / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        kept.append({**cand, "file": name, "det_score": float(face[14])})
        if len(kept) >= 12:
            break
    (FACE_DIR / "external_manifest.json").write_text(json.dumps(kept, indent=2))
    print(f"kept {len(kept)} portraits")


if __name__ == "__main__":
    main()
