#!/usr/bin/env bash
# Rebuild the original "gpu /" tree from this branch + the compressed archives.
# usage: restore.sh <archive_dir> <target_dir>
set -euo pipefail
ARCHIVE_DIR=${1:?usage: restore.sh <archive_dir> <target_dir>}
TARGET=${2:?usage: restore.sh <archive_dir> <target_dir>}
HERE=$(cd "$(dirname "$0")" && pwd)

mkdir -p "$TARGET"
echo "1/2 extracting archives from $ARCHIVE_DIR"
shopt -s nullglob
count=0
for archive in "$ARCHIVE_DIR"/*.tar.zst; do
  echo "    $(basename "$archive")"
  tar -I zstd -xf "$archive" -C "$TARGET"
  count=$((count + 1))
done
echo "    extracted $count archives"

echo "2/2 copying branch files (small files, docs, configs)"
cp -a "$HERE"/. "$TARGET"/

echo "done -> $TARGET"
echo "verify against $HERE/MANIFEST.tsv (column 3 = sha256)"
