"""Face perception with a graceful degradation chain.

Backends, in order of preference
--------------------------------
detector:   ``yunet``  -> cv2.FaceDetectorYN (OpenCV 5 objdetect)
            ``dnn``    -> YuNet ONNX decoded manually through cv2.dnn
            ``skin``   -> YCrCb skin + contour proposal (no model required)
recogniser: ``sface``  -> cv2.FaceRecognizerSF (OpenCV 5)
            ``dnn``    -> SFace ONNX through cv2.dnn + 5-point alignment
            ``lbp``    -> LBP spatial histogram descriptor

Every component reports which path executed so telemetry can prove whether the
run was degraded.  A missing model file never raises at construction time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.config import VisionConfig
from src.perception.quality import FaceQuality, LivenessEstimator, LivenessResult, assess_face
from src.telemetry import BBox, embedding_digest

# ArcFace-style 112x112 reference landmarks used by SFace.
REF_LANDMARKS = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


@dataclass
class FaceHit:
    face_id: int
    box: tuple[int, int, int, int]
    score: float
    landmarks: np.ndarray | None = None
    embedding: np.ndarray | None = None
    identity: str | None = None
    identity_score: float | None = None
    quality: FaceQuality | None = None
    liveness: float = 0.0
    liveness_reason: str = "unknown"
    digest: str | None = None
    forensic: bool = False

    def as_record(self, store_embedding: bool = False) -> dict:
        x, y, w, h = self.box
        record = {
            "face_id": self.face_id,
            "bbox": BBox(x=x, y=y, w=w, h=h),
            "det_score": float(min(1.0, max(0.0, self.score))),
            "landmarks": (
                [{"x": float(p[0]), "y": float(p[1])} for p in np.asarray(self.landmarks).reshape(-1, 2)]
                if self.landmarks is not None
                else []
            ),
            "identity": self.identity,
            "identity_score": None if self.identity_score is None else float(self.identity_score),
            "quality": float(self.quality.score) if self.quality else 0.0,
            "liveness": float(self.liveness),
            "embedding_digest": self.digest,
            "forensic": self.forensic,
        }
        return record


@dataclass
class _Track:
    face_id: int
    box: tuple[int, int, int, int]
    misses: int = 0
    age: int = 1
    liveness: LivenessEstimator = field(default_factory=LivenessEstimator)
    identity_votes: dict[str, int] = field(default_factory=dict)
    embedding: np.ndarray | None = None
    last_embed_frame: int = -999


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    x0, y0 = max(ax0, bx0), max(ay0, by0)
    x1, y1 = min(ax0 + aw, bx0 + bw), min(ay0 + ah, by0 + bh)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class FaceTracker:
    """Tiny IoU tracker: stable ids for telemetry and per-track liveness."""

    def __init__(self, max_misses: int = 5) -> None:
        self.max_misses = max_misses
        self._tracks: list[_Track] = []
        self._next_id = 1

    def update(self, detections: list[tuple[tuple[int, int, int, int], float]]) -> list[int]:
        assigned: list[int] = []
        used: set[int] = set()
        for box, _score in detections:
            best_idx, best_iou = -1, 0.25
            for idx, track in enumerate(self._tracks):
                if idx in used:
                    continue
                score = _iou(box, track.box)
                if score > best_iou:
                    best_idx, best_iou = idx, score
            if best_idx >= 0:
                track = self._tracks[best_idx]
                track.box, track.misses, track.age = box, 0, track.age + 1
                used.add(best_idx)
                assigned.append(track.face_id)
            else:
                track = _Track(face_id=self._next_id, box=box)
                self._next_id += 1
                self._tracks.append(track)
                used.add(len(self._tracks) - 1)
                assigned.append(track.face_id)
        for idx, track in enumerate(self._tracks):
            if idx not in used:
                track.misses += 1
        self._tracks = [t for t in self._tracks if t.misses <= self.max_misses]
        return assigned

    def track(self, face_id: int) -> _Track | None:
        return next((t for t in self._tracks if t.face_id == face_id), None)


class FacePipeline:
    """Model-backed face detection + embedding + gallery matching."""

    def __init__(self, cfg: VisionConfig, gallery: dict[str, np.ndarray] | None = None) -> None:
        self.cfg = cfg
        self.detector_backend = "skin"
        self.recognizer_backend = "lbp"
        self.degraded = True
        self._detector = None
        self._dnn_det = None
        self._recognizer = None
        self._dnn_rec = None
        self._gallery: dict[str, np.ndarray] = {}
        self._tracker = FaceTracker()
        self._frame_no = 0
        self._notes: list[str] = []
        self._init_detector()
        self._init_recognizer()
        if gallery:
            for label, vec in gallery.items():
                self.enroll(label, vec)
        self.degraded = self.detector_backend == "skin" or self.recognizer_backend == "lbp"

    # ------------------------------------------------------------------ init
    def _init_detector(self) -> None:
        path = Path(self.cfg.detector_model)
        if path.exists():
            try:
                self._detector = cv2.FaceDetectorYN_create(
                    str(path), "", (320, 320), self.cfg.det_threshold, self.cfg.nms_threshold, self.cfg.max_faces
                )
                self.detector_backend = "yunet"
                return
            except Exception as exc:  # pragma: no cover - depends on build
                self._notes.append(f"yunet init failed: {type(exc).__name__}")
            try:
                self._dnn_det = cv2.dnn.readNetFromONNX(str(path))
                self._dnn_det.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self._dnn_det.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.detector_backend = "dnn"
                return
            except Exception as exc:  # pragma: no cover
                self._notes.append(f"dnn detector init failed: {type(exc).__name__}")
        self._notes.append("detector model missing: skin/contour fallback")
        self.detector_backend = "skin"

    def _init_recognizer(self) -> None:
        path = Path(self.cfg.recognizer_model)
        if path.exists():
            try:
                self._recognizer = cv2.FaceRecognizerSF_create(str(path), "")
                self.recognizer_backend = "sface"
                return
            except Exception as exc:  # pragma: no cover
                self._notes.append(f"sface init failed: {type(exc).__name__}")
            try:
                self._dnn_rec = cv2.dnn.readNetFromONNX(str(path))
                self._dnn_rec.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self._dnn_rec.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.recognizer_backend = "dnn"
                return
            except Exception as exc:  # pragma: no cover
                self._notes.append(f"dnn recognizer init failed: {type(exc).__name__}")
        self._notes.append("recogniser model missing: LBP descriptor fallback")
        self.recognizer_backend = "lbp"

    def reset_tracking(self) -> None:
        """Clear tracker state between independent images/sequences."""
        self._tracker = FaceTracker()
        self._frame_no = 0

    @property
    def notes(self) -> list[str]:
        return list(self._notes)

    @property
    def descriptor_dim(self) -> int:
        return {"sface": 128, "dnn": 128, "lbp": 1024}[self.recognizer_backend]

    # -------------------------------------------------------------- detection
    def detect(
        self,
        frame: np.ndarray,
        roi: tuple[int, int, int, int] | None = None,
        target_side: int | None = None,
        score_threshold: float | None = None,
    ) -> list[tuple[tuple[int, int, int, int], float, np.ndarray | None]]:
        """Detect faces inside an optional ROI, resized to ``target_side``."""
        if frame.ndim != 3:
            raise ValueError("frame must be HxWx3 BGR")
        H, W = frame.shape[:2]
        if roi is None:
            x0, y0, x1, y1 = 0, 0, W, H
        else:
            x, y, w, h = (int(v) for v in roi)
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            return []
        crop = frame[y0:y1, x0:x1]
        ch, cw = crop.shape[:2]
        side = int(target_side or max(ch, cw))
        side = max(128, min(1920, side))
        scale = min(1.0, side / max(ch, cw)) if max(ch, cw) > side else side / max(ch, cw)
        if abs(scale - 1.0) > 1e-3:
            work = cv2.resize(crop, (max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))), interpolation=cv2.INTER_AREA)
        else:
            work = crop

        thresh = self.cfg.det_threshold if score_threshold is None else score_threshold
        if self.detector_backend == "yunet":
            raw = self._detect_yunet(work, thresh)
        elif self.detector_backend == "dnn":
            raw = self._detect_dnn(work, thresh)
        else:
            raw = self._detect_skin(work, thresh)

        inv = 1.0 / scale
        out = []
        for box, score, landmarks in raw:
            bx = int(round(x0 + box[0] * inv))
            by = int(round(y0 + box[1] * inv))
            bw = int(round(box[2] * inv))
            bh = int(round(box[3] * inv))
            bw, bh = max(1, min(bw, W - bx)), max(1, min(bh, H - by))
            lms = landmarks * inv + np.array([x0, y0], dtype=np.float32) if landmarks is not None else None
            if min(bw, bh) < 16:
                continue
            out.append(((bx, by, bw, bh), float(score), lms))
        out.sort(key=lambda t: -t[1])
        return out[: self.cfg.max_faces]

    def _detect_yunet(self, work: np.ndarray, thresh: float) -> list[tuple[tuple[int, int, int, int], float, np.ndarray]]:
        h, w = work.shape[:2]
        self._detector.setInputSize((w, h))
        self._detector.setScoreThreshold(float(thresh))
        _, faces = self._detector.detect(work)
        if faces is None:
            return []
        result = []
        for row in faces:
            x, y, bw, bh = row[:4]
            result.append(((int(x), int(y), int(bw), int(bh)), float(row[14]), row[4:14].reshape(5, 2).copy()))
        return result

    def _detect_dnn(self, work: np.ndarray, thresh: float) -> list[tuple[tuple[int, int, int, int], float, np.ndarray]]:
        """Manual YuNet decode (cv2.dnn fallback, verified against the API)."""
        size = 320
        img = cv2.resize(work, (size, size), interpolation=cv2.INTER_AREA)
        blob = cv2.dnn.blobFromImage(img, 1.0, (size, size), (0, 0, 0), swapRB=False, crop=False)
        self._dnn_det.setInput(blob)
        names = self._dnn_det.getUnconnectedOutLayersNames()
        outs = dict(zip(names, self._dnn_det.forward(names)))
        boxes, scores, landmarks = [], [], []
        for stride in (8, 16, 32):
            cls = outs[f"cls_{stride}"][0, :, 0]
            obj = outs[f"obj_{stride}"][0, :, 0]
            bbox = outs[f"bbox_{stride}"][0]
            kps = outs[f"kps_{stride}"][0]
            conf = np.sqrt(np.clip(cls, 0, 1) * np.clip(obj, 0, 1))
            grid = size // stride
            for idx in np.where(conf >= thresh)[0]:
                row, col = divmod(int(idx), grid)
                cx = (col + bbox[idx, 0]) * stride
                cy = (row + bbox[idx, 1]) * stride
                bw = math.exp(float(np.clip(bbox[idx, 2], -3, 3))) * stride
                bh = math.exp(float(np.clip(bbox[idx, 3], -3, 3))) * stride
                sx, sy = work.shape[1] / size, work.shape[0] / size
                boxes.append([(cx - bw / 2) * sx, (cy - bh / 2) * sy, bw * sx, bh * sy])
                scores.append(float(conf[idx]))
                pts = []
                for k in range(5):
                    pts.append([(col + kps[idx, 2 * k]) * stride * sx, (row + kps[idx, 2 * k + 1]) * stride * sy])
                landmarks.append(np.asarray(pts, dtype=np.float32))
        if not boxes:
            return []
        keep = cv2.dnn.NMSBoxes(boxes, scores, thresh, self.cfg.nms_threshold)
        keep = np.asarray(keep).reshape(-1).tolist()
        return [(tuple(int(v) for v in boxes[i]), scores[i], landmarks[i]) for i in keep]

    def _detect_skin(self, work: np.ndarray, thresh: float) -> list[tuple[tuple[int, int, int, int], float, np.ndarray]]:
        """Model-free fallback: YCrCb skin mask + face-like contour filtering."""
        ycrcb = cv2.cvtColor(work, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = work.shape[:2]
        out = []
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            area = bw * bh
            if area < (h * w) * 0.01 or bw < 20 or bh < 20:
                continue
            ratio = bw / max(1, bh)
            if not 0.55 <= ratio <= 1.25:
                continue
            fill = cv2.contourArea(contour) / area
            score = float(min(0.55, 0.25 + 0.35 * fill))
            if score < thresh:
                continue
            out.append(((x, y, bw, bh), score, None))
        return out

    # ------------------------------------------------------------ recognition
    def embed(self, frame: np.ndarray, hit: FaceHit) -> np.ndarray | None:
        """L2-normalised embedding for a detected face."""
        if self.recognizer_backend == "sface":
            row = np.zeros((1, 15), dtype=np.float32)
            row[0, :4] = hit.box
            if hit.landmarks is not None:
                row[0, 4:14] = np.asarray(hit.landmarks).reshape(-1)
            row[0, 14] = hit.score
            try:
                aligned = self._recognizer.alignCrop(frame, row)
                feat = self._recognizer.feature(aligned).reshape(-1).astype(np.float32)
            except cv2.error:
                return None
        elif self.recognizer_backend == "dnn":
            aligned = self._align(frame, hit)
            if aligned is None:
                return None
            blob = cv2.dnn.blobFromImage(aligned, 1.0, (112, 112), (0, 0, 0), swapRB=True, crop=False)
            self._dnn_rec.setInput(blob)
            feat = self._dnn_rec.forward().reshape(-1).astype(np.float32)
        else:
            aligned = self._align(frame, hit)
            if aligned is None:
                return None
            feat = self._lbp(aligned)
        norm = float(np.linalg.norm(feat))
        return feat / norm if norm > 1e-9 else None

    def _align(self, frame: np.ndarray, hit: FaceHit, size: int = 112) -> np.ndarray | None:
        x, y, w, h = hit.box
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        if hit.landmarks is not None and len(hit.landmarks) == 5:
            src = np.asarray(hit.landmarks, dtype=np.float32) - np.array([x0, y0], dtype=np.float32)
            mat, _ = cv2.estimateAffinePartial2D(src, REF_LANDMARKS, method=cv2.LMEDS)
            if mat is not None:
                return cv2.warpAffine(crop, mat, (size, size), borderMode=cv2.BORDER_REPLICATE)
        return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _lbp(aligned: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY) if aligned.ndim == 3 else aligned
        gray = cv2.resize(gray, (96, 96))
        center = gray[1:-1, 1:-1].astype(np.int16)
        code = np.zeros_like(center, dtype=np.uint8)
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
        for bit, (dy, dx) in enumerate(offsets):
            neigh = gray[1 + dy : 95 + dy, 1 + dx : 95 + dx].astype(np.int16)
            code |= ((neigh >= center) << bit).astype(np.uint8)
        feats = []
        for gy in range(3):
            for gx in range(3):
                cell = code[gy * 31 : (gy + 1) * 31, gx * 31 : (gx + 1) * 31]
                hist = np.bincount(cell.reshape(-1), minlength=256).astype(np.float32)
                hist /= hist.sum() + 1e-9
                feats.append(np.sqrt(hist))
        vec = np.concatenate(feats)
        return vec / (np.linalg.norm(vec) + 1e-9)

    # ---------------------------------------------------------------- gallery
    def enroll(self, label: str, embedding: np.ndarray, consent: bool = True) -> bool:
        """Enroll an identity. Refused without consent by default."""
        if self.cfg.require_consent_for_enrollment and not consent:
            self._notes.append(f"enrollment refused (no consent): {label}")
            return False
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-9:
            return False
        self._gallery[label] = vec / norm
        return True

    def match(self, embedding: np.ndarray) -> tuple[str | None, float | None]:
        if not self._gallery or embedding is None:
            return None, None
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        vec /= float(np.linalg.norm(vec)) + 1e-9
        best_label, best_score = None, -1.0
        for label, ref in self._gallery.items():
            score = float(np.dot(vec, ref))
            if score > best_score:
                best_label, best_score = label, score
        if best_score < self.cfg.recognition_threshold:
            return None, best_score
        return best_label, best_score

    @property
    def gallery_size(self) -> int:
        return len(self._gallery)

    # --------------------------------------------------------------- analyze
    def analyze(
        self,
        frame: np.ndarray,
        motion=None,
        roi: tuple[int, int, int, int] | None = None,
        target_side: int | None = None,
        embed: bool = True,
        score_threshold: float | None = None,
        forensic: bool = False,
    ) -> list[FaceHit]:
        """Detect, score, embed and identify all faces in one call."""
        raw = self.detect(frame, roi=roi, target_side=target_side, score_threshold=score_threshold)
        ids = self._tracker.update([(b, s) for b, s, _ in raw])
        self._frame_no += 1
        hits: list[FaceHit] = []
        for (box, score, landmarks), face_id in zip(raw, ids):
            quality = assess_face(frame, box, landmarks, self.cfg.min_face_px)
            x, y, w, h = box
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
            crop = frame[y0:y1, x0:x1]
            flow = motion.flow_in(box, frame.shape) if motion is not None else None
            track = self._tracker.track(face_id)
            live: LivenessResult = (
                track.liveness.update(crop, flow) if track else LivenessEstimator().update(crop, flow)
            )
            emb = None
            identity, id_score = None, None
            cached = (
                track is not None
                and track.embedding is not None
                and not forensic
                and self._frame_no - track.last_embed_frame < self.cfg.embed_stride
            )
            if cached:
                emb = track.embedding
            elif embed and min(w, h) >= self.cfg.min_face_px and quality.score > 0.2:
                emb = self.embed(frame, FaceHit(face_id=face_id, box=box, score=score, landmarks=landmarks))
                if track is not None and emb is not None:
                    track.embedding = emb
                    track.last_embed_frame = self._frame_no
            if emb is not None:
                identity, id_score = self.match(emb)
            if track and identity:
                track.identity_votes[identity] = track.identity_votes.get(identity, 0) + 1
            hits.append(
                FaceHit(
                    face_id=face_id,
                    box=box,
                    score=score,
                    landmarks=landmarks,
                    embedding=emb,
                    identity=identity,
                    identity_score=id_score,
                    quality=quality,
                    liveness=live.score,
                    liveness_reason=live.reason,
                    digest=embedding_digest(emb),
                    forensic=forensic,
                )
            )
        return hits
