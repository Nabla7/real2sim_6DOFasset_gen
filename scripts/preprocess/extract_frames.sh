#!/usr/bin/env bash
#
# Extract frames from a video for COLMAP / SfM.
# - Uses only system ffmpeg/ffprobe (no Python deps).
# - Writes to a subfolder: <out_dir>/<video_stem>/images/%06d.jpg by default.
# - Optionally deduplicates near-identical frames (mpdecimate) and/or selects by scene changes.
#
# Example:
#   ./scripts/preprocess/extract_frames.sh /path/to/video.mov
#
# Defaults are tuned for SfM:
# - moderate FPS (3)
# - downscale to 1920px width (keeps aspect)
# - high JPEG quality
#
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  extract_frames.sh <video_path> [options]

Options:
  --out_dir <dir>        Base output directory (default: ./data/video_frames)
  --name <stem>          Output subfolder name (default: derived from video filename)
  --fps <float>          Target extraction FPS (default: 3)
  --width <int>          Resize to this width (keep aspect). 0 = keep original (default: 1920)
  --format <jpg|png>     Output image format (default: jpg)
  --jpeg_quality <1-31>  ffmpeg q:v for JPEG (lower=better). Default: 2
  --dedupe               Enable near-duplicate removal (mpdecimate). Default: off
  --scene <float>        Enable scene-change selection (0.0-1.0), e.g. 0.2. Default: off
  --dry_run              Print ffmpeg command without executing
  -h, --help             Show this help

Outputs:
  <out_dir>/<name>/
    images/000000.jpg ...
    info.txt            (basic metadata)

Notes:
  - For SfM, overlap matters more than raw FPS. Start with 2–5 fps.
  - If you get too many similar frames, use --dedupe (or increase --scene).
  - ffmpeg auto-rotates based on metadata by default (good for phone videos).
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: '$1' not found. Please install it (e.g., sudo apt-get install -y ffmpeg)." >&2
    exit 1
  fi
}

VIDEO="${1:-}"
if [[ -z "${VIDEO}" || "${VIDEO}" == "-h" || "${VIDEO}" == "--help" ]]; then
  usage
  exit 0
fi
shift || true

require_cmd ffmpeg
require_cmd ffprobe

OUT_DIR="./data/video_frames"
NAME=""
FPS="3"
WIDTH="1920"
FORMAT="jpg"
JPEG_QUALITY="2"
DEDUPE="0"
SCENE=""
DRY_RUN="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out_dir) OUT_DIR="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --fps) FPS="$2"; shift 2 ;;
    --width) WIDTH="$2"; shift 2 ;;
    --format) FORMAT="$2"; shift 2 ;;
    --jpeg_quality) JPEG_QUALITY="$2"; shift 2 ;;
    --dedupe) DEDUPE="1"; shift 1 ;;
    --scene) SCENE="$2"; shift 2 ;;
    --dry_run) DRY_RUN="1"; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "$VIDEO" ]]; then
  echo "ERROR: video not found: $VIDEO" >&2
  exit 1
fi

if [[ -z "$NAME" ]]; then
  base="$(basename "$VIDEO")"
  NAME="${base%.*}"
  # sanitize: spaces -> underscores
  NAME="${NAME// /_}"
fi

OUT_PATH="${OUT_DIR%/}/${NAME}"
IMG_DIR="${OUT_PATH}/images"
mkdir -p "$IMG_DIR"

INFO_PATH="${OUT_PATH}/info.txt"
{
  echo "video=$VIDEO"
  echo "out=$OUT_PATH"
  echo "fps=$FPS"
  echo "width=$WIDTH"
  echo "format=$FORMAT"
  echo "dedupe=$DEDUPE"
  echo "scene=${SCENE:-off}"
  echo ""
  echo "ffprobe:"
  ffprobe -hide_banner -v error \
    -select_streams v:0 \
    -show_entries stream=codec_name,width,height,avg_frame_rate,r_frame_rate,duration \
    -of default=nw=1 "$VIDEO" || true
} > "$INFO_PATH"

vf_parts=()

# 1) base sampling strategy
if [[ -n "$SCENE" ]]; then
  # Scene-change selection in addition to FPS. This is helpful if camera is static for long periods.
  # We still cap by FPS to avoid bursts of too many frames.
  vf_parts+=("select='gt(scene,${SCENE})',fps=${FPS}")
else
  vf_parts+=("fps=${FPS}")
fi

# 2) optional de-duplication of near-identical frames
if [[ "$DEDUPE" = "1" ]]; then
  # mpdecimate keeps frames with significant changes.
  vf_parts+=("mpdecimate")
fi

# 3) resize (keeps aspect)
if [[ "$WIDTH" != "0" ]]; then
  vf_parts+=("scale=${WIDTH}:-2:flags=lanczos")
fi

VF="$(IFS=, ; echo "${vf_parts[*]}")"

pattern="${IMG_DIR}/%06d.${FORMAT}"

cmd=(ffmpeg -hide_banner -y -i "$VIDEO" -vf "$VF")
if [[ "$FORMAT" == "jpg" || "$FORMAT" == "jpeg" ]]; then
  cmd+=(-q:v "$JPEG_QUALITY")
fi
cmd+=("$pattern")

echo "=== Extracting frames ==="
echo "Output: $IMG_DIR"
echo "Filter: $VF"
echo "Command:"
printf '  %q' "${cmd[@]}"
echo ""

if [[ "$DRY_RUN" = "1" ]]; then
  echo "(dry-run) Not executing."
  exit 0
fi

"${cmd[@]}"

count=$(ls -1 "$IMG_DIR" 2>/dev/null | wc -l | tr -d ' ')
echo "Done. Extracted ${count} frames."
echo "Info written to: $INFO_PATH"

