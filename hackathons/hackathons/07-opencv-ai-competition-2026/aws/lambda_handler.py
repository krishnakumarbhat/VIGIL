"""AWS Lambda entry point for VIGIL-5.

Two invocation shapes are supported:

* ``{"image_b64": "...", "mock": true}`` – fully offline, useful for smoke tests
  and for CI where no AWS credentials exist.
* ``{"bucket": "...", "key": "frames/0001.jpg"}`` – reads the object with S3,
  processes one frame and (unless ``mock``) writes the telemetry record back to
  ``s3://<bucket>/telemetry/<run_id>/<frame>.json``.

The heavy OpenCV runtime is created lazily and reused across warm invocations.
Cloud failures are logged and never fail the frame: the local decision is the
product, the cloud is an optional escalation channel.
"""

from __future__ import annotations

import base64
import json
import os
import time
from typing import Any

import cv2
import numpy as np

_ENGINE = None


def _engine():
    """Build (or reuse) the process-wide vision engine."""
    global _ENGINE
    if _ENGINE is None:
        from src.config import load_config
        from src.engine import VisionEngine

        _ENGINE = VisionEngine(load_config())
    return _ENGINE


def _decode(event: dict[str, Any]) -> np.ndarray:
    """Decode a base64 JPEG/PNG or fetch one frame from S3."""
    if event.get("image_b64"):
        raw = base64.b64decode(event["image_b64"])
        frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    elif event.get("bucket") and event.get("key"):
        import boto3

        obj = boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1")).get_object(
            Bucket=event["bucket"], Key=event["key"]
        )
        frame = cv2.imdecode(np.frombuffer(obj["Body"].read(), dtype=np.uint8), cv2.IMREAD_COLOR)
    else:
        raise ValueError("event needs image_b64 or bucket+key")
    if frame is None:
        raise ValueError("could not decode frame")
    return frame


def _emit_metrics(record: dict[str, Any], wall_ms: float) -> None:
    """Publish one metric datapoint; best-effort only."""
    try:
        import boto3

        boto3.client("cloudwatch", region_name=os.getenv("AWS_REGION", "us-east-1")).put_metric_data(
            Namespace="VIGIL5/Lambda",
            MetricData=[
                {"MetricName": "FrameLatency", "Value": wall_ms, "Unit": "Milliseconds"},
                {"MetricName": "Faces", "Value": len(record["faces"]), "Unit": "Count"},
                {"MetricName": "Forensic", "Value": int(bool(record["agent"]["forensic"])), "Unit": "Count"},
                {"MetricName": "Escalations", "Value": int(bool(record["agent"]["escalated"])), "Unit": "Count"},
            ],
        )
    except Exception:
        pass


def _store(bucket: str, record: dict[str, Any]) -> str | None:
    """Persist the validated telemetry record (never the raw frame)."""
    try:
        import boto3

        key = f"telemetry/{record['run_id']}/{record['frame_index']:06d}.json"
        boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1")).put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(record).encode(),
            ContentType="application/json",
            ServerSideEncryption="AES256",
        )
        return key
    except Exception:
        return None


def _summarise(record: dict[str, Any]) -> str | None:
    """Optional Bedrock operator summary; skipped without credentials."""
    try:
        import boto3

        client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-east-1"))
        prompt = (
            "Write one neutral sentence for a security operator. "
            f"Faces={len(record['faces'])}, forensic={record['agent']['forensic']}, "
            f"escalated={record['agent']['escalated']}, reasons="
            f"{[r['code'] for r in record['agent']['reasons']]}. Do not identify people."
        )
        response = client.converse(
            modelId=os.getenv("VIGIL_BEDROCK_MODEL", "amazon.nova-lite-v1:0"),
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 80, "temperature": 0.1},
        )
        return response["output"]["message"]["content"][0]["text"].strip()[:500]
    except Exception:
        return None


def handler(event: dict[str, Any], context: Any | None = None) -> dict[str, Any]:
    """Lambda handler: one frame in, one validated telemetry record out."""
    started = time.perf_counter()
    mock = bool(event.get("mock", os.getenv("VIGIL_AWS_MOCK", "1") == "1"))
    try:
        frame = _decode(event)
        frame_index = int(event.get("frame_index", 0))
        outcome = _engine().process_frame(frame, source=str(event.get("source", "lambda")), index=frame_index)
        record = json.loads(outcome.telemetry.model_dump_json())
        wall_ms = (time.perf_counter() - started) * 1000.0
        stored = None
        summary = None
        if not mock:
            if event.get("bucket"):
                stored = _store(event["bucket"], record)
            _emit_metrics(record, wall_ms)
            if event.get("summarise"):
                summary = _summarise(record)
        body = {
            "schema_version": record["schema_version"],
            "run_id": record["run_id"],
            "frame_index": record["frame_index"],
            "backend": record["backend"],
            "faces": record["faces"],
            "agent": record["agent"],
            "verification": record["verification"],
            "mock": mock,
            "stored": stored,
            "summary": summary,
            "lambda_ms": round(wall_ms, 2),
            "telemetry": record,
        }
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(body),
        }
    except Exception as exc:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": type(exc).__name__, "detail": str(exc)[:200], "mock": mock}),
        }
