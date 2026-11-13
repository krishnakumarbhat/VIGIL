"""AWS integration with an offline-first contract.

The agent treats cloud inspection as an *optional escalation hook*.  When
``aws_enabled`` is false (the default, and the only mode used by the local
benchmarks) the deterministic :class:`OfflineCloud` is used, so everything
works with no credentials and no network.  :class:`AwsCloud` is a thin,
defensively coded boto3 client for Rekognition, CloudWatch, S3 and Bedrock.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.config import VisionConfig
from src.telemetry import CloudRecord, utc_now


class OfflineCloud:
    """Deterministic mock that mirrors the AWS response shape."""

    provider = "offline-mock"

    def __init__(self, cfg: VisionConfig) -> None:
        self.cfg = cfg
        self.calls = 0

    def __call__(self, frame: np.ndarray, faces: list[Any], reason: str) -> CloudRecord:
        self.calls += 1
        digest = hashlib.sha256(np.ascontiguousarray(frame)).hexdigest()[:12]
        known = [f for f in faces if getattr(f, "identity", None)]
        unknown = len(faces) - len(known)
        summary = (
            f"Escalation {digest}: {len(faces)} face(s), {unknown} unknown; "
            f"reason={reason}. No biometric payload left the device."
        )
        return CloudRecord(
            provider=self.provider,
            latency_ms=0.0,
            rekognition_faces=len(faces),
            matched=bool(known) if faces else None,
            incident_id=f"inc-{digest}",
            bedrock_summary=summary,
        )


class AwsCloud:
    """Guarded AWS escalation path.

    * Rekognition ``DetectFaces`` runs on in-memory JPEG bytes (no S3 upload).
    * CloudWatch receives one metric datapoint per escalation.
    * Bedrock (Nova Lite by default) writes a short operator summary.
    * S3 is used only for the JSON incident record, never for raw frames,
      unless ``VIGIL_UPLOAD_FRAMES=1`` is explicitly set.
    """

    provider = "aws"

    def __init__(self, cfg: VisionConfig, session: Any | None = None) -> None:
        if not cfg.aws_enabled:
            raise ValueError("AwsCloud requires aws_enabled=true")
        import boto3
        from botocore.config import Config as BotoConfig

        conf = BotoConfig(retries={"max_attempts": 3, "mode": "standard"}, connect_timeout=3, read_timeout=8)
        self.cfg = cfg
        self.session = session or boto3.Session(region_name=cfg.aws_region)
        self.rekognition = self.session.client("rekognition", config=conf)
        self.cloudwatch = self.session.client("cloudwatch", config=conf)
        self.bedrock = self.session.client("bedrock-runtime", config=conf)
        self.s3 = self.session.client("s3", config=conf) if cfg.aws_bucket else None
        self.calls = 0

    def __call__(self, frame: np.ndarray, faces: list[Any], reason: str) -> CloudRecord:
        self.calls += 1
        t0 = time.perf_counter()
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality])
        if not ok:
            return CloudRecord(provider="aws-degraded", bedrock_summary="JPEG encode failed")
        data = buf.tobytes()
        try:
            rek = self.rekognition.detect_faces(Image={"Bytes": data}, Attributes=["DEFAULT"])
            n_faces = len(rek.get("FaceDetails", []))
        except Exception:
            n_faces = 0
        known = [f for f in faces if getattr(f, "identity", None)]
        summary = self._summarise(len(faces), len(known), reason)
        incident_id = f"inc-{hashlib.sha256(data).hexdigest()[:12]}"
        latency = (time.perf_counter() - t0) * 1000.0
        try:
            self.cloudwatch.put_metric_data(
                Namespace="VIGIL5/Agent",
                MetricData=[
                    {"MetricName": "Escalations", "Value": 1, "Unit": "Count"},
                    {"MetricName": "EscalationLatency", "Value": latency, "Unit": "Milliseconds"},
                    {"MetricName": "UnknownFaces", "Value": max(0, len(faces) - len(known)), "Unit": "Count"},
                ],
            )
        except Exception:
            pass
        if self.s3 is not None:
            try:
                self.s3.put_object(
                    Bucket=self.cfg.aws_bucket,
                    Key=f"incidents/{incident_id}.json",
                    Body=json.dumps(
                        {
                            "incident_id": incident_id,
                            "reason": reason,
                            "faces": len(faces),
                            "unknown": len(faces) - len(known),
                            "rekognition_faces": n_faces,
                            "summary": summary,
                            "generated_utc": utc_now(),
                        }
                    ).encode(),
                    ContentType="application/json",
                    ServerSideEncryption="AES256",
                )
            except Exception:
                pass
        return CloudRecord(
            provider=self.provider,
            latency_ms=round(latency, 2),
            rekognition_faces=n_faces,
            matched=bool(known) if faces else None,
            incident_id=incident_id,
            bedrock_summary=summary,
        )

    def _summarise(self, n_faces: int, n_known: int, reason: str) -> str:
        prompt = (
            "You are a security-operations assistant. Write one neutral sentence (max 40 words) "
            f"summarising: {n_faces} faces, {n_known} matched to consenting staff, trigger={reason}. "
            "Do not identify people or speculate about demographics."
        )
        try:
            response = self.bedrock.converse(
                modelId=self.cfg.bedrock_model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": 80, "temperature": 0.1},
            )
            text = response["output"]["message"]["content"][0]["text"].strip()
            return text[:500]
        except Exception:
            return f"{n_faces} face(s), {n_known} matched; trigger={reason}. Escalated for human review."


def build_cloud(cfg: VisionConfig) -> OfflineCloud | AwsCloud:
    """Return a working cloud hook, falling back to the mock on any failure."""
    if not cfg.aws_enabled:
        return OfflineCloud(cfg)
    try:
        return AwsCloud(cfg)
    except Exception:
        return OfflineCloud(cfg)


def upload_evidence(path: str | Path, cfg: VisionConfig, key: str | None = None) -> str | None:
    """Upload a JSON/text artefact (never a raw frame by default)."""
    if not cfg.aws_enabled or not cfg.aws_bucket:
        return None
    try:
        import boto3

        client = boto3.client("s3", region_name=cfg.aws_region)
        object_key = key or f"evidence/{Path(path).name}"
        client.upload_file(str(path), cfg.aws_bucket, object_key, ExtraArgs={"ServerSideEncryption": "AES256"})
        return object_key
    except Exception:
        return None
