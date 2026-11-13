# Devpost submission form — VIGIL-5 (Track 07)

Paste-ready fields for the OpenCV AI Competition 2026 project page.
Source of truth for every number: `outputs/DEVPOST_SUBMISSION.md` (report) and
`outputs/benchmarks.json` (raw evidence). Nothing here is a new claim.

**Deadline:** Oct 26, 2026 23:59 PDT = **2026-10-27 06:59 UTC = 2026-10-27 12:29 IST**.
Devpost locks the page at that instant; upload the video and save the draft well before it.

---

## Project name

```
VIGIL-5
```

## Tagline (≤ 200 chars)

```
An autonomous vision agent on OpenCV 5 that decides when a view is not good enough — zoom, re-scan, roll back, or escalate — and emits one validated JSON record per frame.
```

## Elevator pitch / short description

```
VIGIL-5 is not a detector with a UI. It runs a closed SEE → REASON → ACT → VERIFY loop
over OpenCV 5: a resolution ladder, ROI zoom, forensic re-scan, rollback when verification
fails, and human escalation after persistent ambiguity. Compute is spent only when the
current view is ambiguous, so a static 320 px detector runs at 216 FPS but catches 0% of
events; VIGIL-5 runs at 24.8 FPS and catches 100%, with identity 0.999 and per-frame
CPU p50 of 36 ms.
```

## Inspiration

```
Demo videos of face pipelines always look impressive and never show the failure: the
subject walks behind a pillar, the frame is too small, the match is ambiguous — and the
system silently returns a confident wrong answer. We had an OpenCV 5 build, a 288-frame
scene with occlusion, distance changes and an unknown visitor, and one question: what does
a vision agent look like if "I am not sure" is a legal output?
```

## What it does

```
Per frame, VIGIL-5 emits a validated JSON record (Pydantic) containing detections,
identity matches, quality metrics, the decision taken and the reason for it.

The loop:
  SEE    — YuNet detection at a base resolution, quick quality score
  REASON — is this good enough? (face size, blur, occlusion, margin between identities)
  ACT    — resolution ladder step, ROI zoom, forensic re-scan, or replay with a stronger path
  VERIFY — re-score the new result; roll back if it did not improve
  ESCALATE — flag for a human after persistent ambiguity instead of guessing

On the 288-frame demo this yields event recall 1.00 with identity 0.999, versus 0.00
event recall for the fixed-resolution baseline, at 24.8 wall FPS and 36.2 ms CPU p50
per frame on four pinned x86 cores.
```

## How we built it

```
- OpenCV 5.0.0 (Python 3.10+) with YuNet (228 KB) and SFace (37 MB) from the OpenCV Zoo,
  loaded through ONNX Runtime-free cv2.dnn so the same code path builds on arm64.
- A typed agent core: every stage returns a Pydantic record, so telemetry is validated at
  the boundary rather than debugged after the fact.
- A budget object (CPU-ms + escalation counters) that the ACT stage must charge against;
  the agent cannot zoom for free.
- A fused preprocessing kernel (resize + gray + Laplacian + blur in one pass) integrated
  into the loop after A/B measurement: 45.04 → 36.21 ms CPU p50, ~20% wall-FPS gain,
  F1 change 0.7309 → 0.7303.
- Benchmarks are scripted, repeated and interleaved (9+ reps/operator, 3 reps/timing
  block, spread retained in JSON); logs/experiments.tsv is append-only.
- AWS delivery: base64/S3 Lambda handler, arm64 + x86 Dockerfiles, ECS Fargate task
  definitions, Terraform and CloudFormation — offline-validated only.
```

## Challenges

```
1. Honest arm64 numbers. The COOL criterion wants measured Graviton results; no Graviton
   host was reachable. We ran the portable probe on a real aarch64 userland under QEMU,
   got identical detection counts, and reported the 265× latency gap as QEMU overhead —
   not as an Arm speedup. The report has a section titled "What is not demonstrated".
2. The F1 paradox. Full-agent box F1 (0.731) is below the static detector (0.753) because
   forensic recovery finds partially occluded slivers whose boxes miss the GT IoU gate,
   while event recall goes 0.00 → 1.00. We kept the lower F1 and documented why.
3. Host noise. Load average ranged 1.5–87 during the week; timings are reported as
   least-contended p50 with the spread retained, and a "loaded host" pass is included
   rather than discarded.
4. Rollback. Early versions could chain zooms and make a frame worse; the VERIFY stage now
   keeps the previous result unless the new score strictly improves.
```

## Accomplishments that we're proud of

```
- 100% event recall on the 288-frame occlusion/distance demo at 24.8 FPS, with a 0.00
  event-recall baseline that runs 8.7× faster — the agent trades throughput for the
  thing the task actually measures.
- YuNet detection ROC-AUC 0.9917 (P 1.000, R 0.9833, F1 0.9916 on 100 labelled images);
  SFace TAR@FAR 1e-2 = 1.000 on 60 pairs as a regression check.
- A fully scripted evidence chain: every row in the report maps to a JSON field and a
  reproducible command; sha256 digests for the artifacts are in the evidence index.
- Refused to claim the COOL award: 62/100 self-scored, 0 of the 30 Graviton points claimed.
```

## What we learned

```
- Adaptive compute is a scheduling problem, not a model problem: the biggest win was
  deciding *where* to look (ROI at 320 px, ~3.0× latency reduction with equal recall),
  not adding a bigger network.
- Verification needs a rollback, or the agent's extra attempts can lower quality.
- Cost telemetry inside the loop (CPU-ms, escalations) changes what the agent does; an
  uncharged zoom is a zoom that never stops.
- Measurement discipline is a feature: interleaved A/B, retained spread and a published
  caveat list made the honest numbers defensible.
```

## What's next

```
- Build the arm64 image on a real Graviton host and replace the emulated probe with
  measured COOL evidence (the recipe and refusal guard are already in the repo).
- Live AWS deployment: the Lambda handler and IaC validate offline; next is a real account,
  S3 event ingestion and CloudWatch alarms under load.
- Replace the synthetic corpus with consented, in-the-wild footage and a certified PAD
  model for liveness.
- Learn the escalation threshold per deployment instead of using the fixed ambiguity rule.
```

## Built with

```
python, opencv, numpy, pydantic, psutil, matplotlib, onnx, yunet, sface, docker,
terraform, aws-lambda, amazon-s3, amazon-ecs, ecs-fargate, qemu, ffmpeg, git
```

## Links

| Field | Value |
|---|---|
| Video demo | `outputs/demo_annotated.mp4` (12 s, 288 frames @ 24 fps, 1.9 MB) — upload directly or host unlisted |
| Code repository | **TODO: publish this directory to a public GitHub repo** (not currently a git repo) |
| Architecture | `outputs/architecture_diagram.png` |
| Technical report | `outputs/DEVPOST_SUBMISSION.md` (paste as the "additional information" / project details body) |
| Evidence | `outputs/benchmarks.json`, `outputs/benchmarks.csv`, `outputs/cool_evidence.json`, `logs/experiments.tsv` |

## Submission checklist (manual, ~30 min + upload)

1. Create the Devpost project, paste the fields above (name, tagline, elevator pitch).
2. Paste `outputs/DEVPOST_SUBMISSION.md` into the project details body; it renders as
   Markdown and carries the evidence index.
3. Upload `outputs/demo_annotated.mp4` as the video, or host it unlisted and paste the URL.
   Optional: render a 60–90 s narrated cut — 12 s is short for a judging pass.
4. Upload `outputs/architecture_diagram.png` as the thumbnail.
5. Publish the repo (public, or shared with the judges) and paste the URL. Exclude
   `data/models/*.onnx` if size is a problem; the README documents the download.
6. Pick the tracks: **Overall**, **Agentic Vision**. Do **not** enter **Best Use of COOL** —
   no verified Graviton evidence exists and the report says so.
7. Save the draft early, then submit before **2026-10-27 12:29 IST**.

## Known gap to close before claiming test coverage

`python3 -m pytest -q tests` fails locally with
`ModuleNotFoundError: No module named 'langsmith'`. Either install `langsmith` into the
shared interpreter and record the real pass count, or state "tests require the optional
`langsmith` dependency" on the page. Do not claim "61 passing" until that run is green.
