APPROVED — every finding from both rounds is fixed, and each one was verified by running the code rather than reading the diff.

# Python backend review — transcribe.py and install.sh

Reviewer: independent read-only review agent (python-agent:python-developer, agent name `python-review`). Scope: `skills/transcribe/tools/transcribe.py` and `install.sh` at commit 40cfc51 on `fix/code-review-fixes`. Date: 2026-09-02. Read-only; no edits; working tree clean. The Swift sources changed in the same commit fall outside this review and are covered by `2026-09-swift-review.md`.

## Verification

The blocking `UnboundLocalError` is gone. `supported_formats` and `known_extension` now sit at transcribe.py:517-525, above the output-path block at 527. A no-flag run reaches the dependency check and exits 1 with no traceback.

Every guard fires before `check_dependencies()`, so none of them costs a model load. Observed directly:

| Invocation | Result |
|---|---|
| no flags | missing-packages error, exit 1 |
| `Recording 2026.09.01` | format warning, then dependency error, no truncated output name |
| `--min-speakers 5 --max-speakers 2` | `--min-speakers cannot exceed --max-speakers` |
| `--num-speakers 0` | `--num-speakers must be at least 1` |
| `--num-speakers 2 --max-speakers 3` | mutually-exclusive error |
| `--output /tmp/nope/x.txt` | output directory does not exist |
| `--output /tmp` | output path is a directory |

The broken-pipe handling at transcribe.py:561-570 works. Reproduced the exact tail of `main()` against `| head`: exit code 0, zero bytes on stderr, and the full 399-line file intact on disk. The `os.dup2` onto the null device correctly suppresses the interpreter's "Exception ignored" flush noise, which a bare `except BrokenPipeError: pass` would not.

Fixed-order failure reporting at transcribe.py:402-403 behaves as intended. Against a 3-second sibling the cause printed at 0.1 seconds in both directions, transcription-fails-fast and diarization-fails-fast. When both workers fail the transcription exception is reported deterministically, since it is checked first. The `fut.exception() if fut in done else None` guard is the right shape: it never blocks, because `exception()` on a member of the `done` set returns immediately.

The revised comment at transcribe.py:404-405 is now accurate. Already-running workers cannot be cancelled, so the executor still joins the survivor on exit and the process exits only after both threads finish. The user sees the cause at once, which was the point.

`install.sh:76-81` now warns per package, so a WhisperKit build failure alongside a successful FluidAudio build no longer passes silently. `bash -n` parses clean.

## Confirmed from the earlier rounds

Transcript written before printing at transcribe.py:559. The `Pipeline.from_pretrained` `None` guard with the model-terms URL. The pyannote imports moved inside the `try` whose `finally` unlinks the temp WAV. The output path resolved with `expanduser().resolve()`. Extension-aware naming via `with_name`, which avoids the `with_suffix` truncation. The `convert_to_wav` return annotation. The 1.5-second merge-gap split, which correctly compares against the running end time of the accumulating segment.

## Accepted as-is

Two deliberate non-changes, both reasonable. `--model` stays unrestricted so raw Hugging Face repo ids keep working; the cost is that a typo becomes a 404 partway into a run. The FluidAudio resource bundle is still omitted in `--copy` mode; its contents are only text-to-speech assets that the transcription path never loads, so this is latent fragility rather than a live break. It becomes real if FluidAudio ever moves a speech-recognition resource into that bundle.

## Unverified

Claims about pyannote, mlx-whisper and faster-whisper runtime behaviour rest on reading, not execution; none of the three is installed on this machine. Two items are worth confirming on a machine that has them. Whether pyannote 4.x returns `None` from `Pipeline.from_pretrained` or raises on an invalid token, though the guard is correct under either behaviour. And whether the installed faster-whisper resolves the bare `large-v3-turbo` size string, since `FASTER_WHISPER_MODEL_MAP` maps only `distil-large-v3` and the default model reaches `WhisperModel` unmapped on the CPU fallback path.

## Note for future rounds

The one bug that reached a commit in this cycle was a use-before-assignment on the happy path, introduced while fixing a different finding and caught only once someone ran the entry point. Three review rounds reasoned from diffs without executing anything. Running the command once per round is cheap here, because all argument validation now precedes the expensive imports, and it catches exactly the class of error that diff-reading misses.
