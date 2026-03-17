#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 <frames_dir> <output.mp4> [fps]" >&2
  exit 1
fi

frames_dir=$1
output=$2
fps=${3:-24}

ffmpeg -y \
  -framerate "${fps}" \
  -i "${frames_dir}/frame_%04d.png" \
  -c:v libx264 \
  -pix_fmt yuv420p \
  -crf 18 \
  "${output}"
