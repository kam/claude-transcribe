APPROVED — the regression is fixed, verified independently, and no findings remain open.

# Swift transcribe tools: code review

Reviewer: independent read-only review agent (swift-agent:swift-developer, agent name `swift-review`). Branch `fix/code-review-fixes`, final commit `40cfc51`. Scope: the two SwiftPM executables at `skills/transcribe/tools/fluid-transcribe` and `skills/transcribe/tools/swift-transcribe`, plus their `Package.swift` manifests. Date: 2026-09-02.

## Verdict

APPROVED. Three review rounds found one high-severity data-loss bug, three medium findings, eight low findings, and one regression introduced by a fix. All are resolved or consciously accepted. Nothing remains open.

## Round 3: the output-path regression

The fix matches what the previous round recommended. Both tools now build a `rawOutput` string, absolutize it through `URL(fileURLWithPath:).standardizedFileURL.path`, then validate. See fluid-transcribe main.swift:42-48 and swift-transcribe main.swift:59-65.

I re-derived the behavior rather than accepting the report, running the resolve-and-validate logic against six inputs:

| Input | Resolves to | Outcome |
|---|---|---|
| `notes.txt` | cwd/notes.txt | accepted, regression fixed |
| `./notes.txt` | cwd/notes.txt | accepted |
| `sub/notes.txt` with no `sub/` | cwd/sub/notes.txt | rejected, full parent path in message |
| `/tmp/notes.txt` | unchanged | accepted |
| `/private/tmp/` | /tmp | rejected as a directory target |
| `~/Documents` | /Users/kam/Documents | rejected as a directory target |

Two edge cases beyond the reported ones also behave. A trailing slash is normalized away by `standardizedFileURL` and then caught by the directory check rather than producing a confusing parent path. An empty `-o ""` resolves to the current directory and is likewise caught by the directory check, so it gives a clean error instead of the `NSException` that `URL(fileURLWithPath:)` historically raised on empty input.

## Findings resolved across the review

**High.** swift-transcribe silently wrote an empty transcript when word timestamps were absent. WhisperKit attaches word timings only when its decoder emits alignment weights, and SpeakerKit's subsegment strategy discards every segment lacking them, so a full successful run could produce a one-byte file and exit 0. The fallback at swift-transcribe main.swift:203-212 now recovers the plain text and warns on stderr, matching the guard fluid-transcribe already had.

**Medium.** The output path was validated only after inference completed, so a typo cost an entire run. Both tools now check the destination before loading models. Same-speaker runs merged without any limit, collapsing a monologue into a single line spanning the whole file; `mergeConsecutive` now splits after a 1.5 second pause in both tools. The whole-file in-memory Float32 array remains by design, since sharing one decode across both pipelines is incompatible with the libraries' streaming URL overloads, and the tradeoff is now documented with its 64 KB per second cost.

**Low.** Speaker counts below 1 are rejected before inference, which is stricter than FluidAudio's silent clamp to 1. Zero-sample audio is caught after decode. fluid-transcribe prints model-loading stages unconditionally with download-size hints, and the misleading "(offline)" label is gone; `--verbose` still gates two messages, so the flag is not dead.

**Accepted as-is.** The `openai_whisper-` prefix on `--model`, skipping model cleanup at process exit, and not adopting the streaming URL overloads. All three are reasonable for a short-lived CLI.

## Verified clean

These were investigated and found sound. They should not be re-reviewed without new evidence.

- **Task-group cancellation on partial failure.** Both libraries honor cooperative cancellation, so a throw in one pipeline does not leave the process waiting on the other. FluidAudio checks in `ChunkProcessor`, `TdtDecoderV3`, `AsrManager+Pipeline`, `OfflineSegmentationProcessor`, and `OfflineEmbeddingExtractor`; WhisperKit in `TranscribeTask` and `TextDecoder`; SpeakerKit in `PyannoteDiarizer`.
- **fluid-transcribe's speaker attribution.** The containment scan cannot mis-attribute overlapping speech, because `postProcessing.exclusiveSegments` defaults to true and trims output segments to be non-overlapping. The nearest-segment fallback correctly covers the resulting gaps.
- **`buildWordTimings`** is a real exported FluidAudio API, not a missing local function.
- **The model identifier** `openai_whisper-large-v3-v20240930` is OpenAI's large-v3-turbo release and matches the help text. WhisperKit's separate `_turbo` suffix denotes an Argmax compute variant.
- **Clean-machine reproducibility.** Both `Package.resolved` files are tracked in git, so the `from:` version ranges resolve to pinned revisions.
- **The `install.sh --copy` path.** It copies only the executable and omits `FluidAudio_FluidAudio.bundle`, but `Bundle.module` is referenced solely from TTS code these tools never reach, and `otool -L` shows no `@rpath` dependencies, so the NemoTextProcessing xcframework links statically.

## Known limitations

Two items are correct today but rest on current library behavior rather than a guarantee.

- The missing resource bundle in `--copy` installs would become a `fatalError` if FluidAudio ever touched `Bundle.module` on an ASR or diarization path. Copying the whole `.build/release` directory would remove the dependency on that assumption.
- swift-transcribe's task closures capture `self` (main.swift:87, 94, 123) while fluid-transcribe hoists locals first (main.swift:75-76). Both compile under the Swift 5 language mode the manifests declare. Enabling `swiftLanguageMode(.v6)` would require giving swift-transcribe the fluid-transcribe treatment.

## Process note

This review was read-only throughout. The reviewer made no edits to the repository in any round; the working tree is clean at `40cfc51`. Its only writes were throwaway Swift scripts under `/tmp`, used to verify Foundation path behavior empirically rather than from memory, since removed.

The final commit also changed `skills/transcribe/tools/transcribe.py`, covering executor error reporting, a format warning, and `BrokenPipeError` handling. That file was outside this reviewer's brief and is covered by `2026-09-python-review.md`.
