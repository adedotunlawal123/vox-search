#!/usr/bin/env python3
"""
transcribe.py -- GPU-accelerated audio transcription for the vox-search pipeline.

Automatically picks the fastest backend and device available on whatever machine
it is deployed to:

  * faster-whisper (CTranslate2) when installed -- several times faster than
    openai-whisper at the same accuracy. float16 on CUDA, int8 on CPU.
  * openai-whisper (PyTorch) as a fallback -- CUDA, Apple MPS, or CPU.

Run ``python transcribe.py --list-devices`` to see what this machine will use.

Examples
--------
    python transcribe.py audio.mp3
    python transcribe.py audio.mp3 --model large-v3 -o Learn_Transformer
    python transcribe.py lecture.m4a --output-format srt -o lecture.srt
    python transcribe.py recordings/*.mp3 -o transcripts/
    python transcribe.py audio.mp3 --device cpu --model small   # force CPU

Install
-------
    pip install faster-whisper                 # recommended (GPU + CPU)
    pip install openai-whisper                 # fallback backend

For CUDA with faster-whisper you also need the cuBLAS/cuDNN runtime libraries:
    pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

MODELS = [
    "tiny", "tiny.en", "base", "base.en", "small", "small.en",
    "medium", "medium.en", "large-v1", "large-v2", "large-v3",
    "large-v3-turbo", "turbo", "distil-large-v3",
]

OUTPUT_FORMATS = ["txt", "json", "srt", "vtt"]


# --------------------------------------------------------------------------- #
# Backend / device discovery
# --------------------------------------------------------------------------- #

def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def cuda_device_count() -> int:
    """Visible CUDA devices, preferring CTranslate2 so torch is not required."""
    if _installed("ctranslate2"):
        try:
            import ctranslate2
            return ctranslate2.get_cuda_device_count()
        except Exception:
            pass
    if _installed("torch"):
        try:
            import torch
            return torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception:
            pass
    # Last resort: the driver is there even if no ML runtime is installed yet.
    return 1 if _nvidia_smi(["--query-gpu=name", "--format=csv,noheader"]) else 0


def mps_available() -> bool:
    """Apple Silicon GPU (only usable by the openai-whisper backend)."""
    if not _installed("torch"):
        return False
    try:
        import torch
        return torch.backends.mps.is_available()
    except Exception:
        return False


def _nvidia_smi(args: list[str]) -> str:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return ""
    try:
        out = subprocess.run([exe, *args], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def gpu_description() -> str:
    """Human-readable name of GPU 0, or '' if none."""
    if _installed("torch"):
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                return f"{props.name} ({props.total_memory / 1024**3:.1f} GB)"
        except Exception:
            pass
    smi = _nvidia_smi(["--query-gpu=name,memory.total", "--format=csv,noheader"])
    if smi:
        name, _, mem = smi.splitlines()[0].partition(",")
        return f"{name.strip()} ({mem.strip()})"
    return ""


def preload_cuda_libraries(device: str) -> None:
    """Make pip-installed NVIDIA runtime libraries discoverable.

    CTranslate2 links against cuBLAS and cuDNN lazily -- often not until the first
    decode -- but the ``nvidia-*-cu12`` wheels drop those libraries inside
    site-packages, where the OS loader does not look. Windows needs the directories
    added to the DLL search path; Linux needs the shared objects pulled into the
    process up front, since LD_LIBRARY_PATH cannot be changed after startup.

    A no-op when the libraries came from a system CUDA install (they are already on
    the loader path) or when running on CPU.
    """
    if device != "cuda":
        return
    try:
        spec = importlib.util.find_spec("nvidia")
    except (ImportError, ValueError):
        return
    roots = list(spec.submodule_search_locations) if spec and spec.submodule_search_locations else []

    windows = sys.platform == "win32"
    pending = []
    for root in roots:
        if windows:
            for directory in sorted(glob.glob(os.path.join(root, "*", "bin"))):
                try:
                    os.add_dll_directory(directory)
                except OSError:
                    pass
                pending += sorted(glob.glob(os.path.join(directory, "*.dll")))
        else:
            pending += sorted(glob.glob(os.path.join(root, "*", "lib", "lib*.so*")))

    # Loading each library by absolute path puts it in the process under its plain
    # base name, which is what CTranslate2 later asks the loader for. On Windows the
    # DLL search path alone is not enough: CTranslate2's runtime lookup does not
    # consult directories added via os.add_dll_directory.
    #
    # Two passes, because a library may need a sibling loaded first (cuDNN's
    # sublibraries, for instance) and sorted order does not guarantee that.
    mode = 0 if windows else ctypes.RTLD_GLOBAL
    for _ in range(2):
        failed = []
        for lib in pending:
            try:
                ctypes.CDLL(lib, mode=mode)
            except OSError:
                failed.append(lib)
        if not failed:
            break
        pending = failed


def pick_backend(requested: str) -> str:
    have_faster = _installed("faster_whisper")
    have_openai = _installed("whisper")

    if requested == "faster-whisper":
        if not have_faster:
            raise SystemExit("faster-whisper is not installed. Run: pip install faster-whisper")
        return requested
    if requested == "openai-whisper":
        if not have_openai:
            raise SystemExit("openai-whisper is not installed. Run: pip install openai-whisper")
        return requested

    if have_faster:
        return "faster-whisper"
    if have_openai:
        return "openai-whisper"
    raise SystemExit(
        "No transcription backend found.\n"
        "  pip install faster-whisper      (recommended, GPU-accelerated)\n"
        "  pip install openai-whisper      (fallback)"
    )


def pick_device(requested: str, backend: str) -> str:
    if requested != "auto":
        return requested
    if cuda_device_count() > 0:
        return "cuda"
    if backend == "openai-whisper" and mps_available():
        return "mps"
    return "cpu"


def pick_compute_type(requested: str, device: str) -> str:
    """CTranslate2 quantization. float16 is the sweet spot on any modern GPU."""
    if requested != "auto":
        return requested
    return "float16" if device == "cuda" else "int8"


def print_device_report() -> None:
    have_faster = _installed("faster_whisper")
    have_openai = _installed("whisper")
    backend = "faster-whisper" if have_faster else "openai-whisper" if have_openai else None
    n_cuda = cuda_device_count()
    gpu = gpu_description()

    print("Backends:")
    print(f"  faster-whisper  {'installed' if have_faster else 'not installed'}")
    print(f"  openai-whisper  {'installed' if have_openai else 'not installed'}")
    print("Devices:")
    print(f"  CUDA GPUs       {n_cuda}{f' -> {gpu}' if gpu else ''}")
    print(f"  Apple MPS       {'available' if mps_available() else 'not available'}")
    print(f"  CPU threads     {os.cpu_count()}")
    if backend is None:
        print("\nSelected: nothing -- install a backend first (see --help).")
        return
    device = pick_device("auto", backend)
    print(
        f"\nSelected: backend={backend} device={device} "
        f"compute_type={pick_compute_type('auto', device)}"
    )
    if device == "cpu" and n_cuda:
        print("Note: a GPU was detected but the backend cannot use it -- check the CUDA runtime libs.")


# --------------------------------------------------------------------------- #
# Engines
# --------------------------------------------------------------------------- #

class FasterWhisperEngine:
    """CTranslate2-backed Whisper. Fast on GPU, respectable on CPU."""

    name = "faster-whisper"

    def __init__(self, model, device, compute_type, cpu_threads=0, batch_size=0):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(
            model, device=device, compute_type=compute_type, cpu_threads=cpu_threads
        )
        self.batch_size = batch_size
        self.pipeline = None
        if batch_size:
            try:
                from faster_whisper import BatchedInferencePipeline

                self.pipeline = BatchedInferencePipeline(model=self.model)
            except ImportError:
                print(
                    "warning: --batch-size needs faster-whisper >= 1.1; "
                    "falling back to sequential decoding",
                    file=sys.stderr,
                )
                self.batch_size = 0

    def transcribe(self, path, *, language, task, beam_size, vad, on_segment):
        kwargs = dict(language=language, task=task, beam_size=beam_size, vad_filter=vad)
        if self.pipeline is not None:
            segments, info = self.pipeline.transcribe(
                str(path), batch_size=self.batch_size, **kwargs
            )
        else:
            segments, info = self.model.transcribe(str(path), **kwargs)

        collected = []
        for seg in segments:  # lazy generator -- work happens as we iterate
            item = {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
            collected.append(item)
            on_segment(item, info.duration)

        return {
            "text": " ".join(s["text"] for s in collected).strip(),
            "segments": collected,
            "language": info.language,
            "duration": info.duration,
        }


class OpenAIWhisperEngine:
    """Reference PyTorch implementation. Supports CUDA, Apple MPS and CPU."""

    name = "openai-whisper"

    def __init__(self, model, device, compute_type=None, cpu_threads=0, batch_size=0):
        import whisper

        if cpu_threads:
            import torch

            torch.set_num_threads(cpu_threads)
        self.device = device
        self.model = whisper.load_model(model, device=device)

    def transcribe(self, path, *, language, task, beam_size, vad, on_segment):
        result = self.model.transcribe(
            str(path),
            language=language,
            task=task,
            beam_size=beam_size if beam_size > 1 else None,
            fp16=(self.device == "cuda"),
            verbose=False,
        )
        collected = [
            {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
            for s in result.get("segments", [])
        ]
        duration = collected[-1]["end"] if collected else 0.0
        for item in collected:
            on_segment(item, duration)

        return {
            "text": result["text"].strip(),
            "segments": collected,
            "language": result.get("language") or language,
            "duration": duration,
        }


ENGINES = {
    "faster-whisper": FasterWhisperEngine,
    "openai-whisper": OpenAIWhisperEngine,
}


# --------------------------------------------------------------------------- #
# Output formatting
# --------------------------------------------------------------------------- #

def _timestamp(seconds: float, millis_sep: str = ",") -> str:
    ms = max(0, int(round(seconds * 1000)))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{millis_sep}{ms:03d}"


def format_clock(seconds: float) -> str:
    """Compact stamp for humans: 4:05, or 1:23:45 once the audio passes an hour."""
    total = max(0, int(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def render(result: dict, fmt: str) -> str:
    if fmt == "txt":
        return result["text"] + "\n"
    if fmt == "json":
        return json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if fmt == "srt":
        blocks = [
            f"{i}\n{_timestamp(s['start'])} --> {_timestamp(s['end'])}\n{s['text']}"
            for i, s in enumerate(result["segments"], start=1)
        ]
        return "\n\n".join(blocks) + "\n"
    if fmt == "vtt":
        blocks = [
            f"{_timestamp(s['start'], '.')} --> {_timestamp(s['end'], '.')}\n{s['text']}"
            for s in result["segments"]
        ]
        return "WEBVTT\n\n" + "\n\n".join(blocks) + "\n"
    raise ValueError(f"unknown output format: {fmt}")


def resolve_output_path(out, audio: Path, fmt: str, n_inputs: int) -> Path | None:
    """Where to write this file's transcript, or None for stdout."""
    if out is None:
        return None
    out = Path(out)
    treat_as_dir = out.is_dir() or str(out).endswith(("/", os.sep)) or n_inputs > 1
    if treat_as_dir:
        out.mkdir(parents=True, exist_ok=True)
        return out / f"{audio.stem}.{fmt}"
    if out.parent != Path(""):
        out.parent.mkdir(parents=True, exist_ok=True)
    return out


# --------------------------------------------------------------------------- #
# Public helper (for import from the RAG pipeline)
# --------------------------------------------------------------------------- #

def transcribe_audio(
    path,
    model="base",
    *,
    backend="auto",
    device="auto",
    compute_type="auto",
    language=None,
    task="transcribe",
    beam_size=5,
    vad=True,
    cpu_threads=0,
    batch_size=0,
):
    """Transcribe a single file, returning text, segments and the resolved settings.

    The returned dict has ``text``, ``segments``, ``language``, ``duration``, plus the
    ``backend``/``device``/``model`` that were actually used (useful when they were
    left as ``"auto"``).

    Importable counterpart to the CLI, e.g. from ``intel_audio_logic.py``::

        from transcribe import transcribe_audio
        text = transcribe_audio("lecture.mp3", model="large-v3")["text"]
    """
    backend = pick_backend(backend)
    device = pick_device(device, backend)
    preload_cuda_libraries(device)
    engine = ENGINES[backend](
        model,
        device=device,
        compute_type=pick_compute_type(compute_type, device),
        cpu_threads=cpu_threads,
        batch_size=batch_size,
    )
    result = engine.transcribe(
        path,
        language=language,
        task=task,
        beam_size=beam_size,
        vad=vad,
        on_segment=lambda *_: None,
    )
    # Report what was actually resolved, so callers can log it instead of "auto".
    result.update(audio=str(path), model=model, backend=backend, device=device)
    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="transcribe.py",
        description="GPU-accelerated audio transcription (faster-whisper / openai-whisper).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python transcribe.py audio.mp3\n"
            "  python transcribe.py audio.mp3 --model large-v3 -o Learn_Transformer\n"
            "  python transcribe.py recordings/*.mp3 -o transcripts/ --output-format srt\n"
            "  python transcribe.py --list-devices\n"
        ),
    )
    p.add_argument("audio", nargs="*", help="audio/video file(s) to transcribe")
    p.add_argument(
        "-m", "--model", default="base", metavar="NAME",
        help=f"whisper model size (default: base). Common: {', '.join(MODELS[:8])}, large-v3, turbo",
    )
    p.add_argument(
        "-o", "--output", metavar="PATH",
        help="output file, or a directory (implied when several inputs are given). "
             "Omit to print to stdout",
    )
    p.add_argument(
        "-f", "--output-format", default="txt", choices=OUTPUT_FORMATS,
        help="txt (plain transcript), json (with timestamps), srt or vtt subtitles (default: txt)",
    )
    p.add_argument(
        "-d", "--device", default="auto", choices=["auto", "cuda", "cpu", "mps"],
        help="compute device (default: auto -- CUDA if present, then MPS, then CPU)",
    )
    p.add_argument(
        "--backend", default="auto", choices=["auto", "faster-whisper", "openai-whisper"],
        help="inference backend (default: auto -- prefers faster-whisper)",
    )
    p.add_argument(
        "--compute-type", default="auto",
        choices=["auto", "float16", "bfloat16", "float32", "int8", "int8_float16"],
        help="faster-whisper quantization (default: auto -- float16 on GPU, int8 on CPU)",
    )
    p.add_argument("-l", "--language", help="spoken language code, e.g. en (default: auto-detect)")
    p.add_argument(
        "--task", default="transcribe", choices=["transcribe", "translate"],
        help="transcribe in the source language, or translate to English (default: transcribe)",
    )
    p.add_argument(
        "--beam-size", type=int, default=5, metavar="N",
        help="beam search width; 1 = greedy and fastest (default: 5)",
    )
    p.add_argument(
        "--batch-size", type=int, default=0, metavar="N",
        help="faster-whisper batched decoding, a big GPU speedup (try 8 or 16; default: off)",
    )
    p.add_argument(
        "--no-vad", dest="vad", action="store_false",
        help="disable voice-activity filtering of silence (on by default)",
    )
    p.add_argument("--cpu-threads", type=int, default=0, metavar="N", help="CPU threads (default: auto)")
    p.add_argument("-v", "--verbose", action="store_true", help="print each segment with timestamps")
    p.add_argument("-q", "--quiet", action="store_false", dest="progress", help="suppress progress output")
    p.add_argument("--list-devices", action="store_true", help="report available backends/devices and exit")
    return p


CUDA_ERROR_MARKERS = ("cublas", "cudnn", "cuda", "cublaslt", "nvrtc", "out of memory")


def cuda_hint(device: str, exc: BaseException) -> None:
    """Advice for CUDA failures -- usually missing runtime libraries or OOM.

    Only fires on errors that actually look like CUDA problems, so unrelated
    failures (a bad model name, a Hub download error) are not misdiagnosed.
    """
    if device != "cuda":
        return
    message = str(exc).lower()
    if not any(marker in message for marker in CUDA_ERROR_MARKERS):
        return
    if "out of memory" in message:
        print(
            "hint: the GPU ran out of memory. Try a smaller --model, a smaller\n"
            "      --batch-size, --compute-type int8_float16, or --device cpu",
            file=sys.stderr,
        )
        return
    print(
        "hint: this usually means the CUDA runtime is missing. Install it with\n"
        "      pip install nvidia-cublas-cu12 nvidia-cudnn-cu12\n"
        "      or rerun with --device cpu",
        file=sys.stderr,
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_devices:
        print_device_report()
        return 0

    if not args.audio:
        build_parser().print_usage(sys.stderr)
        print("transcribe.py: error: no audio files given (try --list-devices)", file=sys.stderr)
        return 2

    paths = [Path(a) for a in args.audio]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        print("transcribe.py: error: file not found: " + ", ".join(missing), file=sys.stderr)
        return 2

    backend = pick_backend(args.backend)
    device = pick_device(args.device, backend)
    compute_type = pick_compute_type(args.compute_type, device)

    if device == "mps" and backend == "faster-whisper":
        print(
            "transcribe.py: error: faster-whisper has no MPS support; "
            "use --backend openai-whisper or --device cpu",
            file=sys.stderr,
        )
        return 2

    # Anything chatty goes to stderr so `-o -`/piping stdout stays clean.
    log = (lambda *a: print(*a, file=sys.stderr)) if args.progress else (lambda *a: None)
    gpu = gpu_description() if device == "cuda" else ""
    log(f"backend={backend} model={args.model} device={device}"
        f"{f' ({gpu})' if gpu else ''} compute_type={compute_type}")

    preload_cuda_libraries(device)

    load_start = time.perf_counter()
    try:
        engine = ENGINES[backend](
            args.model,
            device=device,
            compute_type=compute_type,
            cpu_threads=args.cpu_threads,
            batch_size=args.batch_size,
        )
    except Exception as exc:  # missing CUDA libs, bad model name, OOM at load...
        print(f"transcribe.py: error: could not load model: {exc}", file=sys.stderr)
        cuda_hint(device, exc)
        return 1
    log(f"model loaded in {time.perf_counter() - load_start:.1f}s")

    failures = 0
    for audio in paths:
        state = {"last": -1.0}

        def on_segment(segment, total, _state=state):
            if args.verbose:
                log(f"[{_timestamp(segment['start'], '.')} -> "
                    f"{_timestamp(segment['end'], '.')}] {segment['text']}")
            elif args.progress and total:
                done = min(segment["end"], total)
                if done - _state["last"] >= 1.0 or done >= total:
                    _state["last"] = done
                    pct = 100.0 * done / total
                    print(f"\r  {pct:5.1f}%  {format_clock(done)} / {format_clock(total)}",
                          end="", file=sys.stderr, flush=True)

        log(f"transcribing {audio.name} ...")
        started = time.perf_counter()
        try:
            result = engine.transcribe(
                audio,
                language=args.language,
                task=args.task,
                beam_size=args.beam_size,
                vad=args.vad,
                on_segment=on_segment,
            )
        except KeyboardInterrupt:
            print("\ninterrupted", file=sys.stderr)
            return 130
        except Exception as exc:
            if args.progress and not args.verbose:
                print(file=sys.stderr)
            print(f"transcribe.py: error: {audio.name}: {exc}", file=sys.stderr)
            cuda_hint(device, exc)
            failures += 1
            continue
        elapsed = time.perf_counter() - started
        if args.progress and not args.verbose:
            print(file=sys.stderr)

        result["audio"] = str(audio)
        result["model"] = args.model
        result["backend"] = backend
        result["device"] = device

        speed = f", {result['duration'] / elapsed:.1f}x realtime" if elapsed > 0 and result["duration"] else ""
        log(f"done in {elapsed:.1f}s ({format_clock(result['duration'])} of audio{speed}), "
            f"language={result['language']}, {len(result['segments'])} segments")

        text = render(result, args.output_format)
        destination = resolve_output_path(args.output, audio, args.output_format, len(paths))
        if destination is None:
            if len(paths) > 1:
                print(f"===== {audio.name} =====")
            sys.stdout.write(text)
        else:
            destination.write_text(text, encoding="utf-8")
            log(f"wrote {destination}")

    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
