---
name: transcribe
description: Transcribe audio files with speaker diarization, locally on Apple Silicon. Default is FluidAudio (NVIDIA Parakeet TDT v3 + pyannote community-1 on CoreML, ~100x realtime, English and 24 European languages). `--whisper` switches to WhisperKit large-v3-turbo + SpeakerKit for Japanese and other languages. Python fallback for other platforms. Supports mp3, wav, m4a, flac. Invoke with /transcribe <audio-file-path>.
argument-hint: "[--whisper|--python] <audio-file>"
---

# Audio Transcription with Speaker Diarization

Three backends, all producing the same transcript format:

| Backend | Engine | Use when |
|---|---|---|
| **fluid** (default) | FluidAudio: Parakeet TDT 0.6B v3 + pyannote community-1, CoreML | English or one of 24 European languages |
| **whisper** | WhisperKit large-v3-turbo + SpeakerKit, CoreML | Japanese, Chinese, or any language Parakeet lacks |
| **python** | mlx-whisper / faster-whisper + pyannote.audio | Non-Apple hardware |

## Usage

```
/transcribe path/to/audio.mp3
/transcribe --whisper path/to/japanese.m4a
/transcribe --python path/to/audio.mp3
```

Arguments: `$ARGUMENTS`. Pick the backend from them; the remaining token is the audio path. If the user names a language outside Parakeet's set (see below) and did not pass `--whisper`, use the whisper backend anyway and say so.

## Execution

### fluid backend (default)

```bash
"${CLAUDE_SKILL_DIR}/tools/fluid-transcribe/.build/release/fluid-transcribe" "<audio-file>"
```

Flags:
- `--num-speakers N` — exact speaker count (auto-detected if omitted)
- `--output path` — custom output path
- `--verbose`

Parakeet v3 languages: bg, cs, da, de, el, en, es, et, fi, fr, hr, hu, it, lt, lv, mt, nl, pl, pt, ro, ru, sk, sl, sv, uk. No language flag: the model detects within that set.

First run downloads and compiles the models (~4 min, ~1 GB for this backend). Later runs start in seconds.

Rebuild if the binary is missing:

```bash
cd "${CLAUDE_SKILL_DIR}/tools/fluid-transcribe" && swift build -c release --product fluid-transcribe
```

### whisper backend

```bash
"${CLAUDE_SKILL_DIR}/tools/swift-transcribe/.build/release/swift-transcribe" "<audio-file>"
```

Defaults to `large-v3-v20240930` (large-v3-turbo; Argmax's CoreML repo has nothing newer). Flags:
- `--language ja` — skip language detection (pass it whenever the language is known)
- `--model large-v3` — full large-v3; slower, marginal gain
- `--fast` — distil-large-v3, English only. Mutually exclusive with `--model` (the tool rejects both).
- `--num-speakers N`, `--output path`, `--verbose`

Rebuild if the binary is missing:

```bash
cd "${CLAUDE_SKILL_DIR}/tools/swift-transcribe" && swift build -c release --product swift-transcribe
```

### python backend

```bash
python3 -u "${CLAUDE_SKILL_DIR}/tools/transcribe.py" "<audio-file>"
```

Defaults to `large-v3-turbo` and `pyannote/speaker-diarization-community-1`. Prerequisites:

```bash
pip install faster-whisper "pyannote.audio>=4" torch torchaudio
pip install mlx-whisper  # Apple Silicon only, optional
```

`ffmpeg` on PATH, `HF_TOKEN` set, and the model terms accepted at https://huggingface.co/pyannote/speaker-diarization-community-1. The script checks all three before starting.

## Output Format

```
[00:00 - 01:20] Speaker 1: Hello, welcome to the meeting.

[01:20 - 02:38] Speaker 2: Thanks for having me.
```

Consecutive segments from the same speaker are merged. A `.transcript.txt` file is saved alongside the input audio unless `--output` is given.

Words that fall outside every diarization turn are attributed differently per backend: fluid picks the nearest turn, python the turn with most overlap, whisper uses SpeakerKit's subsegment strategy. Expect boundary words to shift speaker between backends.

## Performance (M5 Pro, 2026-09-02, 53 s two-speaker clip, warm models)

| Backend | Speakers found | Total | Speed |
|---|---|---|---|
| fluid | 2 (correct) | 0.5 s | ~105x realtime |
| whisper (1.1.0) | 3 | 8.4 s | ~6x realtime |

Parakeet v3 averages 5.66 WER on the Open ASR Leaderboard against large-v3-turbo's 7.03, but it misses the odd homophone ("rolls" for "roles") and splits some product names. When proper nouns matter more than speed, run `--whisper` as a second pass.

## Alternatives assessed and rejected (2026-09-02)

- **Apple Speech framework** (`yap`): 60x realtime but five word errors on the test clip and no diarization.
- **parakeet-mlx**: same model as the fluid backend, slower, no diarization. Removed.
- **senko**: MIT diarizer, faster than community-1, but cannot label overlapping speech. Revisit if speed on multi-hour files becomes the constraint.
- **SpeakerKit Pro (Sortformer)**: paid, and worse DER than the free pyannote path.
