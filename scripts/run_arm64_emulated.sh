#!/usr/bin/env bash
# Run the portable OpenCV probe on a real aarch64 userland under QEMU user-mode.
#
#   bash scripts/run_arm64_emulated.sh [--out outputs/arch_probe_aarch64.json]
#
# This gives *functional* aarch64 evidence (NEON build, same detections) and a
# transparent timing datapoint.  It is NOT a Graviton benchmark: QEMU translates
# every instruction, so its timings must never be presented as Arm speedup.
# The script refuses to write a "verified" COOL record.
#
# Everything is cached under ${VIGIL_ARM_CACHE:-/tmp/opencode/vigil-arm64};
# nothing is installed into the system Python and no virtualenv is created.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="${VIGIL_ARM_CACHE:-/tmp/opencode/vigil-arm64}"
OUT="$ROOT/outputs/arch_probe_aarch64.json"
if [[ "${1:-}" == "--out" && -n "${2:-}" ]]; then
  OUT="$2"
fi

QEMU_VERSION="v7.2.0-1"
PY_VERSION="3.11.17"
PY_RELEASE="20261021"
UBUNTU_BASE="ubuntu-base-22.04.5-base-arm64"
OPENCV_WHEEL="opencv-python-headless==5.0.0.93"
NUMPY_WHEEL="numpy==2.3.5"

mkdir -p "$CACHE"
cd "$CACHE"

fetch() {  # fetch <url> <file>
  if [[ ! -s "$2" ]]; then
    echo "fetching $2"
    curl -fL --retry 4 --retry-all-errors --connect-timeout 20 -o "$2" "$1"
  fi
}

fetch "https://github.com/multiarch/qemu-user-static/releases/download/${QEMU_VERSION}/qemu-aarch64-static" qemu-aarch64-static
chmod +x qemu-aarch64-static
fetch "https://cdimage.ubuntu.com/ubuntu-base/releases/22.04/release/${UBUNTU_BASE}.tar.gz" ubuntu-base.tar.gz
fetch "https://github.com/astral-sh/python-build-standalone/releases/download/${PY_RELEASE}/cpython-${PY_VERSION}%2B${PY_RELEASE}-aarch64-unknown-linux-gnu-install_only_stripped.tar.gz" python-arm64.tar.gz

if [[ ! -x armroot/bin/true && ! -f armroot/lib/ld-linux-aarch64.so.1 ]]; then
  mkdir -p armroot
  tar -xzf ubuntu-base.tar.gz -C armroot
fi
if [[ ! -x armpy/bin/python3.11 ]]; then
  mkdir -p armpy
  tar -xzf python-arm64.tar.gz -C armpy --strip-components=1
fi
if [[ ! -d armlibs/cv2 ]]; then
  mkdir -p armwheels armlibs
  pip download --no-deps --only-binary=:all: \
    --platform manylinux_2_28_aarch64 --python-version 311 \
    -d armwheels "$OPENCV_WHEEL" "$NUMPY_WHEEL"
  for wheel in armwheels/*.whl; do unzip -oq "$wheel" -d armlibs; done
fi

echo "qemu: $("$CACHE/qemu-aarch64-static" --version | head -1)"
echo "WARNING: aarch64 under QEMU user-mode; timings are emulated, not Graviton." >&2

cd "$ROOT"
PYTHONPATH="$CACHE/armlibs" "$CACHE/qemu-aarch64-static" -L "$CACHE/armroot" \
  "$CACHE/armpy/bin/python3.11" benchmarks/arch_probe.py \
  --out "$OUT" --emulated --detection-images 6

python3 - "$OUT" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
assert payload["environment"]["machine"] == "aarch64", payload["environment"]
assert payload["environment"]["emulated"] is True
print(f"aarch64 probe written: {sys.argv[1]} "
      f"(detection p50 {payload['detection_yunet_640']['p50_ms']} ms under emulation)")
print("Not a Graviton number. Use aws/run_cool_bench.sh on real Arm hardware for that.")
PY
