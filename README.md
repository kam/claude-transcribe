# claude-transcribe

A Claude Code skill that transcribes audio files with speaker labels, entirely on your Mac. No API keys, no upload.

```
/transcribe meeting.m4a
```

```
[00:00 - 00:12] Speaker 1: Good morning everyone, and thanks for joining the quarterly review.

[00:12 - 00:19] Speaker 2: Thanks. Quick question on churn. Is that gross or net?
```

The default backend runs at roughly 100x realtime on an M-series Mac: an hour of audio in under a minute, including diarization.

## Backends

| Backend | Engine | Use when |
|---|---|---|
| **fluid** (default) | [FluidAudio](https://github.com/FluidInference/FluidAudio): NVIDIA Parakeet TDT 0.6B v3 + pyannote community-1, Core ML | English or one of 24 European languages |
| **whisper** | [WhisperKit](https://github.com/argmaxinc/argmax-oss-swift) large-v3-turbo + SpeakerKit, Core ML | Japanese, Chinese, or any language Parakeet lacks |
| **python** | mlx-whisper / faster-whisper + pyannote.audio | Linux or Intel Macs |

Parakeet v3 averages 5.66 WER on the Open ASR Leaderboard against large-v3-turbo's 7.03, at a fraction of the compute. Whisper still handles proper nouns better and covers about 100 languages, so it stays as the second pass.

Measured on an M5 Pro with a 53-second two-speaker clip, models cached:

| Backend | Speakers found | Time | Speed |
|---|---|---|---|
| fluid | 2 (correct) | 0.5 s | ~100x realtime |
| whisper | 3 | 8.4 s | ~6x realtime |

## Requirements

- Apple Silicon Mac, macOS 14 or later, for the Swift backends
- Xcode Command Line Tools (`xcode-select --install`), Swift 5.10 or later
- About 2 GB of disk for models, downloaded on first use
- Python backend only: Python 3.10+, `ffmpeg`, a Hugging Face token

## Install

```bash
git clone https://github.com/kam/claude-transcribe.git
cd claude-transcribe
./install.sh
```

The script builds both Swift binaries and symlinks `skills/transcribe` into `~/.claude/skills/`. Restart Claude Code afterwards. Options:

| Flag | Effect |
|---|---|
| `--copy` | Copy the skill instead of symlinking |
| `--no-build` | Skip the Swift builds |
| `--uninstall` | Remove the skill from `~/.claude` |

Set `CLAUDE_HOME` to install somewhere other than `~/.claude`.

The first transcription downloads and compiles the Core ML models. Expect a few minutes, then seconds on later runs. Models are cached under `~/Library/Application Support/FluidAudio/Models` and `~/Documents/huggingface/models/argmaxinc`.

## Use

```
/transcribe path/to/audio.mp3
/transcribe --whisper path/to/interview.m4a
/transcribe --python path/to/audio.wav
```

Supported inputs: mp3, wav, m4a, flac, and anything else AVFoundation decodes. The transcript is printed and saved next to the input as `<name>.transcript.txt`.

Claude picks the backend from your request. Mention the language and it will switch to Whisper for anything outside Parakeet's set (bg, cs, da, de, el, en, es, et, fi, fr, hr, hu, it, lt, lv, mt, nl, pl, pt, ro, ru, sk, sl, sv, uk).

### Running the binaries directly

```bash
skills/transcribe/tools/fluid-transcribe/.build/release/fluid-transcribe audio.m4a [--num-speakers 2] [--output out.txt]
skills/transcribe/tools/swift-transcribe/.build/release/swift-transcribe audio.m4a [--language ja] [--model large-v3] [--num-speakers 2]
python3 skills/transcribe/tools/transcribe.py audio.m4a [--model large-v3-turbo] [--language en] [--num-speakers 2]
```

`--num-speakers` helps when auto-detection over-splits a recording with similar voices.

### Python backend setup

```bash
pip install faster-whisper "pyannote.audio>=4" torch torchaudio
pip install mlx-whisper   # Apple Silicon only
brew install ffmpeg
export HF_TOKEN=...       # after accepting the terms at
                          # https://huggingface.co/pyannote/speaker-diarization-community-1
```

## How it works

Each Swift binary loads the audio once, runs transcription and diarization concurrently in a task group, then assigns each word to the speaker segment covering its midpoint. Consecutive words from the same speaker merge into one line. Everything stays on the Apple Neural Engine and GPU via Core ML.

## Alternatives assessed

Rejected in September 2026, recorded so the next review starts from evidence:

- **Apple Speech framework** (`SpeechTranscriber`, via `yap`): 60x realtime but five word errors on the test clip where WhisperKit made none, and no speaker diarization.
- **parakeet-mlx**: same Parakeet model, slower than the Core ML port, no diarization.
- **senko**: MIT diarizer, faster than pyannote community-1 with better VoxConverse DER, but it cannot label overlapping speech.
- **SpeakerKit Pro (Sortformer)**: paid, and worse DER on AMI than the free pyannote pipeline.

## License

MIT. Models carry their own licenses: Parakeet TDT v3 is CC-BY-4.0, pyannote community-1 is CC-BY-4.0, Whisper is MIT. FluidAudio is Apache-2.0 and WhisperKit is MIT.
