# VIGIL-5 / Track 07 — one-command entry points.
#
#   make all        tests + full benchmark + annotated demo + local COOL levers + evidence
#   make test       pytest (no plugin autoload)
#   make bench      benchmarks/run_eval.py -> outputs/benchmarks.json
#   make demo       compose the clip and run the agent -> outputs/demo_annotated.mp4
#   make cool-local measured x86 optimisation levers -> outputs/cool_local.json
#   make probe-x86  portable probe, native
#   make arm64      aarch64 probe under QEMU (emulated; not Graviton)
#   make evidence   merge artefacts -> outputs/cool_evidence.json
#   make cool-arm64 real Graviton path; needs COOL_WHEEL_URL and an arm64 builder
#
# COOL is not installed here. `make cool-arm64` is the recipe that would produce
# verified evidence; it refuses to label emulated runs as Arm.

PYTHON ?= python3
export PYTEST_DISABLE_PLUGIN_AUTOLOAD := 1

.PHONY: all test bench demo cool-local probe-x86 arm64 evidence cool-arm64 help

help:
	@sed -n '1,18p' Makefile

all: test bench demo cool-local probe-x86 evidence

test:
	$(PYTHON) -m pytest -p no:cacheprovider

bench:
	$(PYTHON) benchmarks/run_eval.py

demo:
	$(PYTHON) src/generate_demo_video.py \
		--out data/demo/demo.mp4 \
		--annotated-out outputs/demo_annotated.mp4 \
		--telemetry outputs/telemetry/demo.jsonl \
		--summary outputs/demo_summary.json

cool-local:
	$(PYTHON) benchmarks/cool_local.py --out outputs/cool_local.json

probe-x86:
	$(PYTHON) benchmarks/arch_probe.py --out outputs/arch_probe_x86_64.json

arm64:
	bash scripts/run_arm64_emulated.sh --out outputs/arch_probe_aarch64.json

evidence:
	$(PYTHON) scripts/build_cool_evidence.py

cool-arm64:
	@test -n "$(COOL_WHEEL_URL)" || { echo "COOL_WHEEL_URL is required (COOL entitlement)"; exit 2; }
	COOL_WHEEL_URL="$(COOL_WHEEL_URL)" COOL_BUILD_ID="$(COOL_BUILD_ID)" bash aws/run_cool_bench.sh arm64
