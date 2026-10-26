"""Static and offline checks for the AWS deployment artefacts."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from aws.cool_evidence import build_evidence
from aws.lambda_handler import handler

ROOT = Path(__file__).resolve().parents[1]
AWS = ROOT / "aws"


def test_task_definitions_are_valid_and_arch_specific() -> None:
    arm = json.loads((AWS / "ecs-task-definition-arm64.json").read_text())
    x86 = json.loads((AWS / "ecs-task-definition-x86.json").read_text())
    assert arm["runtimePlatform"]["cpuArchitecture"] == "ARM64"
    assert x86["runtimePlatform"]["cpuArchitecture"] == "X86_64"
    for task in (arm, x86):
        assert task["requiresCompatibilities"] == ["FARGATE"]
        container = task["containerDefinitions"][0]
        assert container["essential"] is True
        assert container["logConfiguration"]["logDriver"] == "awslogs"
        env = {item["name"] for item in container["environment"]}
        assert {"VIGIL_THREADS", "VIGIL_AWS_MOCK", "VIGIL_AWS_BUCKET"} <= env


def test_cloudformation_parses_and_has_core_resources() -> None:
    class CfnLoader(yaml.SafeLoader):
        pass

    CfnLoader.add_multi_constructor("!", lambda loader, suffix, node: None)
    template = yaml.load((AWS / "cloudformation.yaml").read_text(), Loader=CfnLoader)
    resources = template["Resources"]
    assert "VisionFunction" in resources
    assert resources["VisionFunction"]["Properties"]["PackageType"] == "Image"
    assert "EvidenceBucket" in resources
    assert "ErrorAlarm" in resources
    assert template["Parameters"]["Architecture"]["Default"] == "arm64"


def test_terraform_is_structurally_complete() -> None:
    main = (AWS / "terraform" / "main.tf").read_text()
    variables = (AWS / "terraform" / "variables.tf").read_text()
    outputs = (AWS / "terraform" / "outputs.tf").read_text()
    assert main.count("{") == main.count("}")
    assert 'resource "aws_lambda_function" "vision"' in main
    assert 'resource "aws_s3_bucket" "evidence"' in main
    assert 'architectures = ["arm64"]' in main
    assert 'variable "image_uri"' in variables
    assert 'output "evidence_bucket"' in outputs


def test_dockerfiles_target_both_arches() -> None:
    arm = (AWS / "Dockerfile.arm64").read_text()
    x86 = (AWS / "Dockerfile.x86").read_text()
    assert "FROM --platform=linux/arm64" in arm
    assert "--platform=linux/amd64" in x86
    for text in (arm, x86):
        assert "COOL_WHEEL_URL" in text or "OPENCV_VERSION" in text
        assert "lambda_handler" in text


def test_lambda_handler_offline_smoke() -> None:
    frame = np.full((96, 96, 3), 120, dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", frame)
    assert ok
    event = {
        "image_b64": base64.b64encode(buffer.tobytes()).decode(),
        "mock": True,
        "frame_index": 4,
        "source": "unit-test",
    }
    response = handler(event)
    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert body["mock"] is True
    assert body["frame_index"] == 4
    assert body["telemetry"]["schema_version"] == "vigil.telemetry/1.0"
    assert body["telemetry"]["source"] == "unit-test"


def test_lambda_handler_rejects_empty_event() -> None:
    response = handler({})
    assert response["statusCode"] == 400
    assert "ValueError" in response["body"]


def _payload(machine: str, load: float = 2.0) -> dict:
    return {
        "environment": {
            "machine": machine,
            "platform": "Linux",
            "python": "3.11",
            "opencv": "5.0.0",
            "opencv_module": "/opt/cool/cv2",
            "build_info_sha256": "abc123",
            "simd": ["NEON"],
            "ipp": [],
            "pinned_cores": [0, 1],
            "load_average_1m": load,
            "timestamp_utc": "2026-10-22T00:00:00Z",
        },
        "benchmarks": {
            "detection": {"roc_auc": 0.99, "cpu_fps": 50.0, "cpu_latency_ms_p50": 18.0},
            "recognition": {"roc_auc": 1.0},
            "agent": {"full_agent": {"wall_fps": 40.0, "f1": 0.7, "event_recall": 1.0}},
            "dispatch": {"mean_speedup": 1.6},
            "thread_scaling": {"1": {"fps": 40.0}},
        },
    }


def test_cool_evidence_refuses_unverified_claims() -> None:
    x86 = build_evidence(_payload("x86_64", load=40.0))
    assert x86["verified_on_arm"] is False
    assert x86["cool_verified"] is False
    assert x86["status"] == "unavailable"
    assert "not arm64" in x86["status_reason"]
    assert any("not arm64" in caveat for caveat in x86["caveats"])
    assert any("load average" in caveat for caveat in x86["caveats"])

    arm_no_id = build_evidence(_payload("aarch64"))
    assert arm_no_id["verified_on_arm"] is True
    assert arm_no_id["cool_verified"] is False

    arm_cool = build_evidence(_payload("aarch64", load=1.0), cool_build_id="cool-2026-10")
    assert arm_cool["cool_verified"] is True
    assert arm_cool["status"] == "measured"
    assert arm_cool["measurements"]["full_agent_wall_fps"] == 40.0
    assert arm_cool["measurements"]["detection_roc_auc"] == 0.99
