"""Build the local face bank and labelled evaluation corpus.

This script is deterministic and fully offline after the models in
``data/models`` exist.  It extracts face crops from images that ship with the
OpenCV source tree (already present on this machine), clusters the crops with
SFace embeddings to avoid identity leakage, and emits:

* ``data/samples/faces/<identity>.jpg``  clean reference crops
* ``data/samples/faces/manifest.json``   provenance + photometric stats
* ``data/eval/detection/...``            labelled positives / negatives
* ``data/eval/recognition_pairs.json``   genuine / impostor pairs

The evaluation corpus is synthetic (real faces composited onto synthetic
backgrounds with exact ground-truth boxes).  It is a *sanity benchmark*, not a
replacement for WIDER FACE / LFW; the submission documents that limitation.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FACE_DIR = ROOT / "data" / "samples" / "faces"
EVAL_DIR = ROOT / "data" / "eval"

SOURCES = [
    "/media/pope/projecteo/github_proj/contri/opencv/samples/data/messi5.jpg",
    "/media/pope/projecteo/github_proj/contri/opencv/doc/js_tutorials/js_assets/lena.jpg",
]

# Licence-clean portraits fetched once by benchmarks/fetch_portrait_bank.py.
EXTERNAL = sorted((ROOT / "data" / "samples" / "faces").glob("ext_*.jpg"))

DETECTOR = ROOT / "data" / "models" / "face_detection_yunet_2023mar.onnx"
RECOGNIZER = ROOT / "data" / "models" / "face_recognition_sface_2021dec.onnx"


def _detector() -> "cv2.FaceDetectorYN":
    return cv2.FaceDetectorYN_create(str(DETECTOR), "", (320, 320), 0.7, 0.3, 100)


def extract_crops() -> list[dict]:
    """Detect faces in the source images and save padded, square-ish crops."""
    FACE_DIR.mkdir(parents=True, exist_ok=True)
    det = _detector()
    rec = cv2.FaceRecognizerSF_create(str(RECOGNIZER), "")
    records: list[dict] = []
    sources = [(str(s), None) for s in SOURCES] + [(str(p), p) for p in EXTERNAL]
    for src, pre_cropped in sources:
        img = cv2.imread(src)
        if img is None:
            print(f"[skip] unreadable {src}")
            continue
        h, w = img.shape[:2]
        scale = min(1.0, 900 / max(h, w))
        work = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1 else img
        det.setInputSize((work.shape[1], work.shape[0]))
        _, faces = det.detect(work)
        if faces is None:
            continue
        for k, face in enumerate(faces):
            if pre_cropped is not None:
                # Already a tight portrait: keep it whole as a clean reference.
                crop = work
                emb = rec.feature(rec.alignCrop(work, face)).reshape(-1)
                tag = hashlib.sha1(src.encode()).hexdigest()[:8]
                name = f"clean_{Path(src).stem}_{tag}.jpg"
                cv2.imwrite(str(FACE_DIR / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 96])
                records.append(
                    {
                        "file": name,
                        "source": src,
                        "face_index": k,
                        "det_score": float(face[14]),
                        "embedding": (emb / (np.linalg.norm(emb) + 1e-9)).round(5).tolist(),
                    }
                )
                continue
            x, y, bw, bh = face[:4] / scale
            cx, cy = x + bw / 2, y + bh / 2
            side = max(bw, bh) * 1.55
            x0 = int(max(0, cx - side / 2))
            y0 = int(max(0, cy - side / 2))
            x1 = int(min(w, cx + side / 2))
            y1 = int(min(h, cy + side / 2))
            crop = img[y0:y1, x0:x1]
            if min(crop.shape[:2]) < 48:
                continue
            crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
            emb = rec.feature(rec.alignCrop(work, face)).reshape(-1)
            tag = hashlib.sha1(f"{src}:{k}".encode()).hexdigest()[:8]
            name = f"src_{Path(src).stem[:18]}_{tag}.jpg"
            cv2.imwrite(str(FACE_DIR / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 96])
            records.append(
                {
                    "file": name,
                    "source": src,
                    "face_index": k,
                    "det_score": float(face[14]),
                    "embedding": (emb / (np.linalg.norm(emb) + 1e-9)).round(5).tolist(),
                }
            )
    return records


def cluster_identities(records: list[dict], threshold: float = 0.42) -> list[dict]:
    """Greedy cosine clustering; keep the sharpest crop per cluster as reference."""
    clusters: list[list[dict]] = []
    for rec in sorted(records, key=lambda r: -r["det_score"]):
        vec = np.asarray(rec["embedding"], dtype=np.float32)
        for group in clusters:
            ref = np.asarray(group[0]["embedding"], dtype=np.float32)
            if float(np.dot(vec, ref)) >= threshold:
                group.append(rec)
                break
        else:
            clusters.append([rec])

    for idx, group in enumerate(clusters):
        ident = f"id{idx:02d}"
        for rec in group:
            rec["identity"] = ident
        group.sort(key=lambda r: -r["det_score"])
        ref = group[0]
        img = cv2.imread(str(FACE_DIR / ref["file"]))
        cv2.imwrite(str(FACE_DIR / f"ref_{ident}.jpg"), img)
        ref["reference"] = f"ref_{ident}.jpg"
    return clusters


def make_background(rng: random.Random, w: int, h: int) -> np.ndarray:
    """Random but realistic-ish indoor/outdoor background with structure."""
    a = np.array([rng.randint(20, 90), rng.randint(25, 95), rng.randint(30, 110)], np.float32)
    b = np.array([rng.randint(90, 190), rng.randint(90, 190), rng.randint(90, 200)], np.float32)
    ramp = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    bg = (a[None, None, :] * (1 - ramp) + b[None, None, :] * ramp)
    bg = np.repeat(bg, w, axis=1)
    # Perspective floor + wall separation.
    horizon = int(h * rng.uniform(0.42, 0.62))
    bg[horizon:] *= 0.55
    for _ in range(rng.randint(2, 5)):
        x = rng.randint(0, w)
        cv2.line(bg, (x, 0), (x + rng.randint(-40, 40), horizon), (60, 60, 65), rng.randint(8, 30))
    bg += np.random.default_rng(rng.randint(0, 10**6)).normal(0, 6, bg.shape)
    return np.clip(bg, 0, 255).astype(np.uint8)


def compose_face(
    bg: np.ndarray,
    face: np.ndarray,
    center: tuple[int, int],
    size: int,
    rng: random.Random,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Alpha-composite a face crop onto a background, returning the GT box."""
    h, w = bg.shape[:2]
    face = cv2.resize(face, (size, size), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.5:
        face = cv2.flip(face, 1)
    angle = rng.uniform(-13, 13)
    mat = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
    face = cv2.warpAffine(face, mat, (size, size), borderMode=cv2.BORDER_REFLECT)
    gain = rng.uniform(0.72, 1.18)
    face = np.clip(face.astype(np.float32) * gain + rng.uniform(-18, 18), 0, 255).astype(np.uint8)
    if rng.random() < 0.35:
        k = rng.choice([3, 5])
        face = cv2.GaussianBlur(face, (k, k), rng.uniform(0.4, 1.4))

    yy, xx = np.mgrid[0:size, 0:size]
    mask = (((xx - size / 2) / (size * 0.46)) ** 2 + ((yy - size / 2) / (size * 0.5)) ** 2) <= 1
    mask = mask.astype(np.float32)
    mask = cv2.GaussianBlur(mask, (0, 0), size * 0.055)
    mask = cv2.merge([mask, mask, mask])

    cx, cy = center
    x0, y0 = cx - size // 2, cy - size // 2
    x1, y1 = x0 + size, y0 + size
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(w, x1), min(h, y1)
    sx1, sy1 = sx0 + (dx1 - dx0), sy0 + (dy1 - dy0)
    if dx1 <= dx0 or dy1 <= dy0:
        return bg, (0, 0, 0, 0)
    roi = bg[dy0:dy1, dx0:dx1].astype(np.float32)
    patch = face[sy0:sy1, sx0:sx1].astype(np.float32)
    alpha = mask[sy0:sy1, sx0:sx1]
    bg[dy0:dy1, dx0:dx1] = np.clip(patch * alpha + roi * (1 - alpha), 0, 255).astype(np.uint8)
    return bg, (dx0, dy0, dx1 - dx0, dy1 - dy0)


def tight_box(image: np.ndarray) -> tuple[float, float, float, float]:
    """Normalised tight face box inside a reference crop (exact GT)."""
    det = _detector()
    h, w = image.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(image)
    if faces is None:
        return (0.15, 0.15, 0.7, 0.7)
    x, y, bw, bh = faces[0][:4]
    return (float(x / w), float(y / h), float(bw / w), float(bh / h))


def build_detection_set(refs: list[dict], seed: int = 7) -> dict:
    """60 positives + 40 negatives at 640x360 with exact boxes."""
    rng = random.Random(seed)
    pos_dir = EVAL_DIR / "detection" / "positive"
    neg_dir = EVAL_DIR / "detection" / "negative"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)
    labels = []
    tight_cache: dict[str, tuple[float, float, float, float]] = {}
    for i in range(60):
        ref = rng.choice(refs)
        face = cv2.imread(str(FACE_DIR / ref["reference"]))
        if ref["reference"] not in tight_cache:
            tight_cache[ref["reference"]] = tight_box(face)
        bg = make_background(rng, 640, 360)
        size = rng.choice([40, 56, 72, 96, 128, 160, 210])
        cx = rng.randint(int(size * 0.35), 640 - int(size * 0.35))
        cy = rng.randint(int(size * 0.35), 340)
        canvas, paste = compose_face(bg, face, (cx, cy), size, rng)
        tx, ty, tw, th = tight_cache[ref["reference"]]
        box = (
            int(paste[0] + tx * paste[2]),
            int(paste[1] + ty * paste[3]),
            int(tw * paste[2]),
            int(th * paste[3]),
        )
        if rng.random() < 0.5:
            canvas = cv2.convertScaleAbs(canvas, alpha=rng.uniform(0.65, 1.25), beta=rng.uniform(-25, 25))
        if rng.random() < 0.5:
            canvas = cv2.GaussianBlur(canvas, (3, 3), rng.uniform(0.2, 0.9))
        name = f"pos_{i:03d}.jpg"
        cv2.imwrite(str(pos_dir / name), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
        labels.append({"file": f"positive/{name}", "has_face": True, "box": list(box), "size": size, "identity": ref["identity"]})
    for i in range(40):
        bg = make_background(rng, 640, 360)
        # Distractor blobs / text / edges: hard negatives, no human faces.
        for _ in range(rng.randint(2, 6)):
            p1 = (rng.randint(0, 640), rng.randint(0, 360))
            p2 = (p1[0] + rng.randint(-120, 120), p1[1] + rng.randint(-90, 90))
            cv2.rectangle(bg, p1, p2, tuple(int(v) for v in rng.choices(range(40, 230), k=3)), -1)
        cv2.putText(bg, f"CAM-{i:02d}", (20, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)
        name = f"neg_{i:03d}.jpg"
        cv2.imwrite(str(neg_dir / name), bg, [cv2.IMWRITE_JPEG_QUALITY, 88])
        labels.append({"file": f"negative/{name}", "has_face": False, "box": None, "size": 0})
    meta = {"seed": seed, "root": str(EVAL_DIR / "detection"), "items": labels}
    (EVAL_DIR / "detection" / "labels.json").write_text(json.dumps(meta, indent=2))
    return meta


def build_recognition_pairs(refs: list[dict], seed: int = 11) -> dict:
    """Genuine pairs (same id, different augmentation) and impostor pairs."""
    rng = random.Random(seed)
    base = []
    for ref in refs:
        img = cv2.imread(str(FACE_DIR / ref["reference"]))
        emb = None
        base.append({"id": ref["identity"], "img": img})
    pairs = []
    for i in range(60):
        a = rng.choice(base)
        if i % 2 == 0:
            b = a
            genuine = True
        else:
            b = rng.choice([x for x in base if x["id"] != a["id"]])
            genuine = False
        aug = b["img"].copy()
        aug = cv2.resize(aug, None, fx=rng.uniform(0.75, 1.15), fy=rng.uniform(0.75, 1.15))
        aug = cv2.convertScaleAbs(aug, alpha=rng.uniform(0.75, 1.2), beta=rng.uniform(-20, 20))
        h, w = aug.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-8, 8), 1.0)
        aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        pairs.append({"a_id": a["id"], "b_id": b["id"], "genuine": genuine, "index": i})
        cv2.imwrite(str(EVAL_DIR / "recognition" / f"a_{i:03d}.jpg"), a["img"])
        cv2.imwrite(str(EVAL_DIR / "recognition" / f"b_{i:03d}.jpg"), aug)
    meta = {"pairs": pairs}
    (EVAL_DIR / "recognition_pairs.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "recognition").mkdir(parents=True, exist_ok=True)
    records = extract_crops()
    clusters = cluster_identities(records)
    refs = [g[0] for g in clusters]
    manifest = {
        "note": "Face crops derived from images shipped with the OpenCV source tree; see source paths.",
        "n_detections": len(records),
        "n_identities": len(clusters),
        "references": [
            {"identity": r["identity"], "file": r["reference"], "source": r["source"]} for r in refs
        ],
        "records": [{k: v for k, v in r.items() if k != "embedding"} for r in records],
    }
    (FACE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    det = build_detection_set(refs)
    rec = build_recognition_pairs(refs)
    print(
        json.dumps(
            {
                "detections": len(records),
                "identities": len(clusters),
                "detection_images": len(det["items"]),
                "recognition_pairs": len(rec["pairs"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
