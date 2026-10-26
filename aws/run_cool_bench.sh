#!/usr/bin/env bash
# Build and run the VIGIL-5 benchmark inside the target architecture.
#
#   COOL_WHEEL_URL="https://.../cool-opencv.whl" COOL_BUILD_ID="cool-..." \
#     aws/run_cool_bench.sh arm64
#
# On an x86 laptop the arm64 image needs buildx + QEMU and the numbers are
# emulated, not Graviton. Pass --allow-emulated only to check that the image
# builds; do not publish those numbers as Arm evidence.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCH="${1:-arm64}"
ALLOW_EMULATED="${ALLOW_EMULATED:-0}"

case "$ARCH" in
  arm64)
    DOCKERFILE="$ROOT/aws/Dockerfile.arm64"
    PLATFORM="linux/arm64"
    ;;
  x86)
    DOCKERFILE="$ROOT/aws/Dockerfile.x86"
    PLATFORM="linux/amd64"
    ;;
  *)
    echo "usage: $0 [arm64|x86]" >&2
    exit 2
    ;;
esac

HOST_ARCH="$(uname -m)"
if [[ "$ARCH" == "arm64" && "$HOST_ARCH" != "aarch64" && "$ALLOW_EMULATED" != "1" ]]; then
  echo "FAIL: arm64 benchmark requested on $HOST_ARCH." >&2
  echo "Run this on Graviton (or set ALLOW_EMULATED=1 for a build-only check)." >&2
  exit 3
fi

if [[ -z "${COOL_WHEEL_URL:-}" ]]; then
  echo "WARN: COOL_WHEEL_URL is unset. The image will use upstream OpenCV;" >&2
  echo "      cool_evidence.py will record status=pending, not COOL-verified." >&2
fi

IMAGE="vigil5-bench:${ARCH}"
echo "Building $IMAGE for $PLATFORM"
docker buildx build \
  --platform "$PLATFORM" \
  --build-arg COOL_WHEEL_URL="${COOL_WHEEL_URL:-}" \
  -f "$DOCKERFILE" \
  -t "$IMAGE" \
  --load \
  "$ROOT"

echo "Running benchmarks inside $IMAGE"
docker run --rm \
  --platform "$PLATFORM" \
  --user "$(id -u):$(id -g)" \
  -v "$ROOT:/work" \
  -w /work \
  -e VIGIL_BENCH_REPS="${VIGIL_BENCH_REPS:-3}" \
  -e VIGIL_THREADS="${VIGIL_THREADS:-1}" \
  --entrypoint /var/lang/bin/python3 \
  "$IMAGE" benchmarks/run_eval.py

COOL_ARGS=(--output "$ROOT/outputs/cool_evidence_${ARCH}.json" --expected-arch arm64)
if [[ -n "${COOL_BUILD_ID:-}" ]]; then
  COOL_ARGS+=(--cool-build-id "$COOL_BUILD_ID")
fi
if [[ "$ARCH" == "arm64" ]]; then
  COOL_ARGS+=(--require-arm)
fi

python3 "$ROOT/aws/cool_evidence.py" "$ROOT/outputs/benchmarks.json" "${COOL_ARGS[@]}"
echo "COOL evidence written to outputs/cool_evidence_${ARCH}.json"
