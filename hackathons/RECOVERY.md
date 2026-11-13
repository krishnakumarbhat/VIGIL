# Recovery note — 2026-11-13

## What happened

The local `gpu /` tree in `heaphero2` was deleted on the assumption that every
large file was inside the per-project archives in `no_mo_compressed/`. That was
only partly true: the archives are an older snapshot and covered 727 large files,
while 620 large files (4.21 GB) existed only in the local tree — untracked and
never committed anywhere.

## What was recovered

An `ntfsundelete` scan of the NTFS partition, followed by sha256 verification of
every candidate against `MANIFEST.tsv`, restored:

- **549 files / 2.95 GB** — byte-for-byte verified (sha256 equal to the manifest)
- **71 files / 1.24 GB** — unrecoverable, clusters already overwritten; listed
  with their original sha256 in `MISSING.tsv` (30 in 06-playground, 20 in
  02-gemma-4, 18 in 05-name-that-llm, 3 in 01-nfl)

## Current coverage

- 2,877 of 2,948 manifest files exist in the local `heaphero2/gpu /` tree
- all files <= 512 KB (2,221) are in this branch
- the external archives still hold the 727 large files from the older snapshot
- `MISSING.tsv` is the authoritative list of data that no longer exists anywhere

## Restore

1. clone this branch for the small files
2. `bash hackathons/restore.sh <archive_dir> <target>` for the archived large files
3. verify anything with the `sha256` column in `MANIFEST.tsv`
