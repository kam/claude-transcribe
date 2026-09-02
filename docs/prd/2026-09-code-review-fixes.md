---
status: approved
created: 2026-09-02
updated: 2026-09-02
lane: change
---

# Transcribe skill: code-review fixes

## Problem
Three review passes (code-review, swift-developer, python-developer) on the initial skill found data-loss and late-failure bugs in all three backends: relative input paths wrote to `/`, empty transcripts were saved silently when word timings were absent, and the Python backend failed minutes late on a missing token or ffmpeg.

## Decision
Fix in place across all three backends, keeping the two Swift tools as separate packages. A shared Swift package for the duplicated helpers was rejected: two files is a tolerable cost for a skill this size.

## Change
- `fluid-transcribe/Sources/main.swift` → absolute input path, stderr errors, output-path and speaker-count preflight, empty-audio guard, plain-text fallback without word timings, pause-split merge, unconditional model-loading messages
- `swift-transcribe/Sources/main.swift` → same, plus `--fast`/`--model` conflict rejected
- `transcribe.py` → ffmpeg/HF_TOKEN preflight, write-before-print, FIRST_EXCEPTION wait, pipeline None guard, argument and output-path validation, temp-WAV cleanup on all paths, single-pass alignment, pause-split merge
- `install.sh` → refuse to delete a foreign directory, `--copy` excludes SwiftPM caches, warn when binaries are absent
- `SKILL.md` → standard `argument-hint`/`$ARGUMENTS`, per-backend gap-word note
- `.gitignore` → exclude `.claude/`

## Docs
- `SKILL.md` updated in the change. `README.md` unchanged: no flag or install step changed.

## Out of scope
- Shared Swift package for duplicated helpers (rejected above)
- Streaming URL overloads for multi-day audio
- Restricting `--model` to a fixed list (raw Hugging Face repo ids stay allowed)

## Acceptance
- [x] Given a bare relative filename, when either Swift tool runs, then the transcript lands next to the input (verified 2026-09-02 with a synthesized clip)
- [x] Given `--fast --model X`, when either whisper backend runs, then it exits with a usage error
- [x] Given a missing output directory, when any backend runs, then it fails before inference
- [x] Both Swift packages build; `py_compile` and `bash -n` pass

## Open questions
None.
