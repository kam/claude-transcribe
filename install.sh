#!/usr/bin/env bash
# Install the transcribe skill into ~/.claude and build its binaries.
#   ./install.sh              symlink the skill (edits in this checkout apply immediately)
#   ./install.sh --copy       copy the skill instead of symlinking
#   ./install.sh --no-build   link/copy only, skip the Swift builds
#   ./install.sh --uninstall  remove the skill from ~/.claude (leaves model caches)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CLAUDE="${CLAUDE_HOME:-$HOME/.claude}"
MODE="link"
BUILD=1
for arg in "$@"; do
  case "$arg" in
    --copy) MODE="copy" ;;
    --no-build) BUILD=0 ;;
    --uninstall) MODE="uninstall" ;;
    *) echo "unknown option: $arg"; exit 1 ;;
  esac
done

DST="$CLAUDE/skills/transcribe"
SRC="$HERE/skills/transcribe"

# Refuse to delete anything at $DST that is not this skill.
remove_dst() {
  [ -e "$DST" ] || [ -L "$DST" ] || return 0
  if [ -L "$DST" ] || grep -qx 'name: transcribe' "$DST/SKILL.md" 2>/dev/null; then
    rm -rf "$DST"
  else
    echo "$DST exists and is not the transcribe skill; remove it by hand first." >&2
    exit 1
  fi
}

if [ "$MODE" = "uninstall" ]; then
  remove_dst
  echo "Removed $DST. Model caches left in place:"
  echo "  ~/Library/Application Support/FluidAudio/Models"
  echo "  ~/Documents/huggingface/models/argmaxinc"
  exit 0
fi

if [ "$BUILD" = 1 ]; then
  if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
    echo "Swift backends need Apple Silicon; installing the skill with the Python backend only." >&2
    BUILD=0
  elif ! command -v swift >/dev/null; then
    echo "swift not found. Install Xcode Command Line Tools: xcode-select --install" >&2
    exit 1
  fi
fi

if [ "$BUILD" = 1 ]; then
  for pkg in fluid-transcribe swift-transcribe; do
    echo "==> Building $pkg (first build fetches dependencies; a few minutes)"
    (cd "$HERE/skills/transcribe/tools/$pkg" && swift build -c release --product "$pkg")
  done
fi

mkdir -p "$CLAUDE/skills"
remove_dst
if [ "$MODE" = "copy" ]; then
  # Copy sources without the SwiftPM caches, then just the release binaries.
  rsync -a --exclude '.build' "$SRC/" "$DST/"
  for pkg in fluid-transcribe swift-transcribe; do
    bin="$SRC/tools/$pkg/.build/release/$pkg"
    if [ -x "$bin" ]; then
      mkdir -p "$DST/tools/$pkg/.build/release"
      cp "$bin" "$DST/tools/$pkg/.build/release/$pkg"
    fi
  done
else
  ln -s "$SRC" "$DST"
fi

for pkg in fluid-transcribe swift-transcribe; do
  if [ ! -x "$SRC/tools/$pkg/.build/release/$pkg" ]; then
    echo "Warning: $pkg not built; that backend will not work (the Python backend, /transcribe --python, still does)." >&2
    echo "         Build later with: ./install.sh" >&2
  fi
done
echo "Installed ($MODE) at $DST. Restart Claude Code, then run: /transcribe <audio-file>"
echo "First transcription downloads models (~2 GB for both backends, a few minutes). Later runs start in seconds."
