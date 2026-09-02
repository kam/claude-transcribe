#!/usr/bin/env python3
"""
Audio transcription with speaker diarization.

Uses mlx-whisper (Apple Silicon GPU) or faster-whisper (CPU fallback) for
transcription, and pyannote.audio (community-1 pipeline, pyannote.audio >= 4.0) for speaker diarization.
Requires HF_TOKEN environment variable for pyannote model access.

Usage:
    python3 transcribe.py <audio_file> [--model large-v3-turbo] [--language en] [--num-speakers N] [--min-speakers N] [--max-speakers N] [--output path] [--fast]
"""

import argparse
import concurrent.futures
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"

# Force line-buffered stdout so progress is visible in real time
# (os.environ["PYTHONUNBUFFERED"] doesn't affect the current process)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Backend detection — done once at module load so we report it early
# ---------------------------------------------------------------------------
try:
    import mlx_whisper as _mlx_whisper  # noqa: F401
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

# MLX model name mapping
FASTER_WHISPER_MODEL_MAP = {
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
}

MLX_MODEL_MAP = {
    "tiny": "mlx-community/whisper-tiny",
    "base": "mlx-community/whisper-base",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "distil-large-v3": "mlx-community/distil-whisper-large-v3",
}


def check_dependencies():
    """Check that all required packages are installed."""
    missing = []

    if not MLX_AVAILABLE:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            missing.append("faster-whisper")

    try:
        import pyannote.audio  # noqa: F401
    except ImportError:
        missing.append("pyannote.audio")
    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append("torch")

    if missing:
        print(f"ERROR: Missing required packages: {', '.join(missing)}", file=sys.stderr)
        print(f"Install with: pip install {' '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not found on PATH. Install with: brew install ffmpeg", file=sys.stderr)
        sys.exit(1)

    # Fail here, not minutes later inside the diarization thread after a full transcription.
    if not os.environ.get("HF_TOKEN"):
        print(
            "ERROR: HF_TOKEN environment variable is not set.\n"
            "Set it with: export HF_TOKEN='your_huggingface_token'\n"
            "You must also accept the model terms at:\n"
            f"  https://huggingface.co/{PYANNOTE_MODEL}",
            file=sys.stderr,
        )
        sys.exit(1)

    backend = "mlx-whisper (Apple Silicon GPU)" if MLX_AVAILABLE else "faster-whisper (CPU)"
    print(f"Transcription backend: {backend}")


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


# ---------------------------------------------------------------------------
# WAV conversion (runs in background so model loading overlaps with it)
# ---------------------------------------------------------------------------

def convert_to_wav(audio_path: str) -> str | None:
    """
    Convert audio to 16 kHz mono WAV for pyannote compatibility.
    Always converts (even .wav inputs) to ensure correct sample rate and channel count.
    Returns the temp WAV path. The caller is responsible for deleting the returned temp file.
    """
    print("[diarize] Converting to 16kHz mono WAV for diarization compatibility...")
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_wav.close()
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", tmp_wav.name],
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        os.unlink(tmp_wav.name)
        print(f"[diarize] ffmpeg error: {e.stderr.decode(errors='replace')}", file=sys.stderr)
        raise
    except Exception:
        os.unlink(tmp_wav.name)
        raise
    print("[diarize] WAV conversion complete.")
    return tmp_wav.name


# ---------------------------------------------------------------------------
# Transcription (runs in parallel with diarization)
# ---------------------------------------------------------------------------

def run_transcription(audio_path: str, model_size: str, language: str | None) -> tuple[list[dict], str]:
    """
    Transcribe audio using mlx-whisper or faster-whisper.
    Returns (segments, detected_language).
    """
    t0 = time.monotonic()

    if MLX_AVAILABLE:
        segments, detected_language = _transcribe_mlx(audio_path, model_size, language)
    else:
        segments, detected_language = _transcribe_faster_whisper(audio_path, model_size, language)

    elapsed = time.monotonic() - t0
    print(f"[transcribe] Transcription completed in {elapsed:.1f}s ({len(segments)} segments)")
    return segments, detected_language


def _transcribe_mlx(audio_path: str, model_size: str, language: str | None) -> tuple[list[dict], str]:
    """Transcribe using mlx-whisper (Apple Silicon GPU)."""
    import mlx_whisper

    hf_repo = MLX_MODEL_MAP.get(model_size, f"mlx-community/whisper-{model_size}-mlx")
    print(f"[transcribe] Loading MLX Whisper model ({hf_repo})...")

    kwargs: dict = {"path_or_hf_repo": hf_repo, "verbose": False}
    if language:
        kwargs["language"] = language

    print("[transcribe] Transcribing audio (MLX)...")
    result = mlx_whisper.transcribe(audio_path, **kwargs)

    detected_language = result.get("language", "unknown")
    print(f"[transcribe] Detected language: {detected_language}")

    segments = [
        {
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
        }
        for seg in result.get("segments", [])
    ]

    for seg in segments:
        print(f"[transcribe]   [{format_timestamp(seg['start'])} - {format_timestamp(seg['end'])}] {seg['text']}")

    return segments, detected_language


def _transcribe_faster_whisper(audio_path: str, model_size: str, language: str | None) -> tuple[list[dict], str]:
    """Transcribe using faster-whisper (CPU/CUDA fallback)."""
    import torch
    from faster_whisper import WhisperModel

    # CTranslate2 has no MPS support, so Apple Silicon without mlx-whisper runs on CPU.
    if torch.cuda.is_available():
        device, compute_type = "cuda", "float16"
    else:
        device, compute_type = "cpu", "int8"

    print(f"[transcribe] Using device: {device} (compute_type: {compute_type})")
    resolved_model = FASTER_WHISPER_MODEL_MAP.get(model_size, model_size)
    print(f"[transcribe] Loading Whisper model ({resolved_model})...")
    model = WhisperModel(resolved_model, device=device, compute_type=compute_type)

    print("[transcribe] Transcribing audio (faster-whisper)...")
    whisper_kwargs: dict = {"beam_size": 1, "vad_filter": True}
    if language:
        whisper_kwargs["language"] = language

    segments_iter, info = model.transcribe(audio_path, **whisper_kwargs)

    detected_language = info.language
    print(f"[transcribe] Detected language: {detected_language} (probability: {info.language_probability:.2f})")

    segments = []
    for segment in segments_iter:
        seg = {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
        }
        segments.append(seg)
        print(f"[transcribe]   [{format_timestamp(segment.start)} - {format_timestamp(segment.end)}] {segment.text.strip()}")

    return segments, detected_language


# ---------------------------------------------------------------------------
# Diarization (runs in parallel with transcription)
# ---------------------------------------------------------------------------

def run_diarization(
    audio_path: str,
    wav_future: concurrent.futures.Future,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
) -> object:
    """
    Run pyannote speaker diarization.
    Waits for wav_future to resolve before starting (gives the path to use).
    Returns the pyannote Annotation object.
    """
    import torch
    from pyannote.audio import Pipeline

    t0 = time.monotonic()
    wav_path: str | None = None
    try:
        # Load pipeline while WAV conversion may still be running.
        # pyannote MUST NOT run on MPS — known timestamp bugs on Apple Silicon.
        print("[diarize] Loading pyannote diarization pipeline...")
        pyannote_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline = Pipeline.from_pretrained(PYANNOTE_MODEL, token=os.environ["HF_TOKEN"])
        pipeline.to(pyannote_device)

        # Now wait for WAV conversion to be ready
        wav_path = wav_future.result()  # blocks until convert_to_wav() returns
        diarization_audio_path = wav_path

        print("[diarize] Running speaker diarization...")
        diarization_kwargs: dict = {}
        if num_speakers is not None:
            diarization_kwargs["num_speakers"] = num_speakers
        if min_speakers is not None:
            diarization_kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            diarization_kwargs["max_speakers"] = max_speakers

        diarization_output = pipeline(diarization_audio_path, **diarization_kwargs)

        # Extract the Annotation object from DiarizeOutput if needed
        if hasattr(diarization_output, "speaker_diarization"):
            diarization = diarization_output.speaker_diarization
        else:
            diarization = diarization_output

        elapsed = time.monotonic() - t0
        print(f"[diarize] Diarization completed in {elapsed:.1f}s")
        return diarization
    finally:
        # If the pipeline load failed, the WAV may still be in flight; wait so it can be removed.
        if wav_path is None:
            try:
                wav_path = wav_future.result()
            except Exception:
                wav_path = None
        if wav_path is not None:
            try:
                os.unlink(wav_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def align_segments(whisper_segments: list[dict], diarization) -> list[dict]:
    """Align whisper segments with speaker labels from pyannote."""
    print("Aligning transcription with speaker labels...")

    turns = [(turn.start, turn.end, spk) for turn, _, spk in diarization.itertracks(yield_label=True)]

    diarized_segments = []
    for ws in whisper_segments:
        ws_start = ws["start"]
        ws_end = ws["end"]
        ws_mid = (ws_start + ws_end) / 2

        # Speaker at the midpoint wins; otherwise the turn with the most overlap.
        speaker = "Unknown"
        best_overlap = 0
        for t_start, t_end, spk in turns:
            if t_start <= ws_mid <= t_end:
                speaker = spk
                break
            overlap = min(ws_end, t_end) - max(ws_start, t_start)
            if overlap > best_overlap:
                best_overlap = overlap
                speaker = spk

        diarized_segments.append(
            {
                "start": ws_start,
                "end": ws_end,
                "speaker": speaker,
                "text": ws["text"],
            }
        )

    # Rename speakers to friendly names (Speaker 1, Speaker 2, etc.)
    speaker_map: dict = {}
    speaker_counter = 1
    for seg in diarized_segments:
        if seg["speaker"] not in speaker_map and seg["speaker"] != "Unknown":
            speaker_map[seg["speaker"]] = f"Speaker {speaker_counter}"
            speaker_counter += 1
        seg["speaker"] = speaker_map.get(seg["speaker"], "Unknown Speaker")

    print(f"Identified {len(speaker_map)} speakers")
    return diarized_segments


# ---------------------------------------------------------------------------
# Main pipeline coordinator
# ---------------------------------------------------------------------------

def transcribe_and_diarize(
    audio_path: str,
    model_size: str = "large-v3-turbo",
    language: str | None = None,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[dict]:
    """
    Run transcription and diarization concurrently, then align segments.

    Pipeline:
      1. WAV conversion starts immediately in a background thread.
      2. Transcription and diarization run concurrently.
         - Diarization loads its pipeline eagerly, then waits for WAV conversion.
      3. Segments are aligned after both tasks complete.
    """
    total_start = time.monotonic()

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Step 1: Start WAV conversion immediately
        wav_future: concurrent.futures.Future = executor.submit(convert_to_wav, audio_path)

        # Step 2: Run transcription and diarization concurrently
        transcription_future: concurrent.futures.Future = executor.submit(
            run_transcription, audio_path, model_size, language
        )
        diarization_future: concurrent.futures.Future = executor.submit(
            run_diarization,
            audio_path,
            wav_future,
            num_speakers,
            min_speakers,
            max_speakers,
        )

        # Wait for both to complete (re-raise any exceptions)
        whisper_segments, detected_language = transcription_future.result()
        diarization = diarization_future.result()

    total_elapsed = time.monotonic() - total_start
    print(f"Total pipeline time: {total_elapsed:.1f}s")

    # Step 3: Align
    return align_segments(whisper_segments, diarization)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_transcript(segments: list[dict]) -> str:
    """Format segments into a readable transcript."""
    lines = []

    # Merge consecutive segments from the same speaker
    merged: list[dict] = []
    for seg in segments:
        if merged and merged[-1]["speaker"] == seg["speaker"]:
            merged[-1]["end"] = seg["end"]
            merged[-1]["text"] += " " + seg["text"]
        else:
            merged.append(dict(seg))

    for seg in merged:
        ts_start = format_timestamp(seg["start"])
        ts_end = format_timestamp(seg["end"])
        lines.append(f"[{ts_start} - {ts_end}] {seg['speaker']}: {seg['text']}")

    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio with speaker diarization"
    )
    parser.add_argument("audio_file", help="Path to the audio file")
    parser.add_argument(
        "--model",
        default="large-v3-turbo",
        help="Whisper model size (tiny, base, small, medium, large-v2, large-v3, large-v3-turbo, distil-large-v3). Default: large-v3-turbo",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language code (e.g., en, es, fr). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Exact number of speakers (if known)",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="Minimum number of speakers",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Maximum number of speakers",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path. Default: <audio_file>.transcript.txt",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use the distilled model (distil-large-v3) for faster transcription.",
    )
    args = parser.parse_args()

    if args.fast and args.model != parser.get_default("model"):
        parser.error("--fast and --model are mutually exclusive")
    model_size = "distil-large-v3" if args.fast else args.model
    if args.fast:
        print(f"Fast mode: using distilled model ({model_size})")

    # Validate input file
    audio_path = Path(args.audio_file).resolve()
    if not audio_path.exists():
        print(f"ERROR: Audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    supported_formats = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".wma", ".aac", ".webm", ".mp4"}
    if audio_path.suffix.lower() not in supported_formats:
        print(
            f"WARNING: Unrecognized format '{audio_path.suffix}'. "
            f"Supported: {', '.join(sorted(supported_formats))}",
            file=sys.stderr,
        )

    check_dependencies()

    print(f"Processing: {audio_path}")
    print("=" * 60)

    # Run parallel transcription + diarization pipeline
    segments = transcribe_and_diarize(
        str(audio_path),
        model_size=model_size,
        language=args.language,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )

    # Format output
    transcript = format_transcript(segments)

    print("=" * 60)
    print("DIARISED TRANSCRIPT")
    print("=" * 60)
    print(transcript)
    print("=" * 60)

    # Save to file
    output_path = Path(args.output) if args.output else audio_path.with_suffix(".transcript.txt")
    output_path.write_text(transcript, encoding="utf-8")
    print(f"\nTranscript saved to: {output_path}")


if __name__ == "__main__":
    main()
