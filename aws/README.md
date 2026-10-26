# VIGIL-5 AWS deployment suite

Everything here is **deployable but deliberately not deployed** from the build
environment: no AWS credentials or Graviton host were available. The local
benchmark and tests are real; the cloud numbers are marked pending rather than
invented. See the verification matrix at the bottom.

## What is included

| Path | Purpose |
|---|---|
| `lambda_handler.py` | One-frame Lambda entry point; base64 or S3 input, validated telemetry out, optional S3/CloudWatch/Bedrock. |
| `Dockerfile.arm64` | Graviton/Lambda arm64 image; accepts a COOL wheel URL. |
| `Dockerfile.x86` | x86_64 reference image. |
| `ecs-task-definition-arm64.json` | Fargate ARM64 task for continuous video ingestion. |
| `ecs-task-definition-x86.json` | Same for x86_64. |
| `terraform/` | S3 evidence bucket, ECR repos, IAM, arm64 Lambda, alarms, ECS cluster. |
| `cloudformation.yaml` | Equivalent single-stack YAML template. |
| `cool_evidence.py` | Converts `benchmarks.json` into an honest COOL/Graviton provenance record. |
| `run_cool_bench.sh` | Builds the target image and runs `benchmarks/run_eval.py` inside it. |
| `requirements.txt` | Runtime dependencies; OpenCV is installed by the Dockerfiles. |

## Build and push

```bash
AWS_REGION=us-east-1
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
aws ecr create-repository --repository-name vigil5-arm64 --region "$AWS_REGION"

aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com"

# COOL wheel URL comes from the AWS Marketplace subscription/entitlement.
docker buildx build --platform linux/arm64 \
  --build-arg COOL_WHEEL_URL="$COOL_WHEEL_URL" \
  -f aws/Dockerfile.arm64 \
  -t "$ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/vigil5-arm64:latest" \
  --push .

docker buildx build --platform linux/amd64 \
  -f aws/Dockerfile.x86 \
  -t "$ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/vigil5-x86:latest" \
  --push .
```

## Deploy

Terraform:

```bash
cd aws/terraform
terraform init
terraform apply \
  -var account_suffix="$ACCOUNT" \
  -var image_uri="$ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/vigil5-arm64:latest"
```

CloudFormation (alternative):

```bash
aws cloudformation deploy \
  --stack-name vigil5 \
  --template-file aws/cloudformation.yaml \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides \
      ImageUri="$ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/vigil5-arm64:latest" \
      Architecture=arm64
```

## Invoke

Offline smoke test (no credentials, returns a real telemetry record):

```bash
python3 - <<'PY'
import base64, json, cv2
from aws.lambda_handler import handler
frame = cv2.imread("data/samples/faces/src_lena_36fe10b1.jpg")
ok, buf = cv2.imencode(".jpg", frame)
event = {"image_b64": base64.b64encode(buf).decode(), "mock": True}
response = handler(event)
print(response["statusCode"])
print(json.loads(response["body"])["backend"])
PY
```

S3 event:

```bash
aws lambda invoke --function-name vigil5-vision \
  --payload '{"bucket":"vigil5-evidence-...","key":"frames/0001.jpg"}' \
  /tmp/vigil5-response.json
```

ECS:

```bash
aws ecs register-task-definition --cli-input-json file://aws/ecs-task-definition-arm64.json
aws ecs run-task --cluster vigil5-cluster \
  --task-definition vigil5-vision-arm64 --launch-type FARGATE \
  --network-configuration 'awsvpcConfiguration={subnets=[subnet-...],securityGroups=[sg-...],assignPublicIp=DISABLED}'
```

## COOL / Graviton evidence

`run_cool_bench.sh` refuses to run arm64 on x86 unless `ALLOW_EMULATED=1`, and
`cool_evidence.py` only sets `cool_verified=true` when the machine is aarch64
**and** a COOL build id is supplied. The JSON record lists the OpenCV build
hash, SIMD dispatch lines, pinned cores, load average and the measured metrics.

```bash
COOL_WHEEL_URL="https://.../cool-opencv.whl" \
COOL_BUILD_ID="$(date -u +%Y%m%dT%H%M%SZ)-cool" \
aws/run_cool_bench.sh arm64
```

## Security and privacy

- Evidence bucket is private, AES256-encrypted and versioned.
- Only JSON telemetry/incident records are uploaded; raw frames stay on device
  unless `VIGIL_UPLOAD_FRAMES=1` is explicitly set in a custom build.
- Rekognition runs on in-memory JPEG bytes, never on an S3 copy of the frame.
- Consent is mandatory for gallery enrollment; `consent.json` labels are the
  only source of truth.
- IAM policies are least-privilege for S3/Rekognition/CloudWatch/Bedrock; no
  wildcard S3 resource.
- No secrets are committed. Model paths and bucket names come from environment
  variables.

## Verification matrix

| Claim | Status |
|---|---|
| Local OpenCV 5.0.0 detection/recognition/agent benchmarks | **Executed** (`outputs/benchmarks.json`) |
| pytest suite | **Executed** (`python3 -m pytest`) |
| Lambda handler offline smoke test | **Executed** (`tests/test_aws_assets.py`) |
| Docker/ECS/Terraform/CloudFormation syntax | **Static validation only** |
| `terraform apply` / `aws cloudformation deploy` | **Pending** — no AWS credentials |
| Graviton + COOL latency/FPS | **Not measured** — no Arm host or COOL entitlement; see `outputs/cool_evidence.json` (`status: unavailable`) |

The canonical local record is built by `scripts/build_cool_evidence.py`, which
merges the benchmark, the measured x86 levers and the QEMU aarch64 probe. The
`cool_evidence.py` tool below remains the arm64-runner converter; it is the one
that can ever set `cool_verified=true`, and only on native Arm with a COOL build
id.

Costs if deployed for a demo: arm64 Lambda at 2 GB is roughly $0.000033 per
second of execution; a 2 vCPU / 4 GB Fargate task is roughly $0.08 per hour in
`us-east-1` (verify current pricing). Delete the stack afterwards:
`terraform destroy` or `aws cloudformation delete-stack`.
