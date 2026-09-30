"""Entry point: transcribe audio on the GPU, index the transcript, then chat about it.

    python intel_audio_logic.py                                  # chat over the existing transcript
    python intel_audio_logic.py lecture.mp3                       # transcribe, then chat
    python intel_audio_logic.py lecture.mp3 --whisper-model large-v3   # better accuracy
    python intel_audio_logic.py lecture.mp3 --retranscribe         # ignore the cached transcript
    python intel_audio_logic.py --transcript Learn_Transformer     # use a transcript you already have

Answers cite the point in the audio the context came from, e.g. [12:34-13:01].

Transcription itself lives in transcribe.py, which picks the GPU when one is
available. Run `python transcribe.py --list-devices` to see what this machine will use.
"""

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from ChunckAndEmbed import chunk_by_sentence as chunker
from ChunckAndEmbed import chunk_segments
from ChunckAndEmbed import generate_embedding as embedder
from HybridSearchImplementation import VectorIndex
from HybridSearchImplementation import BM25Index
from HybridSearchImplementation import Retriever
from ConversationHandler import ConversationHandler, DEFAULT_MODEL
from transcribe import transcribe_audio, format_clock

# Transcript used when no audio file is given.
DEFAULT_TRANSCRIPT = Path("Learn_Transformer")
# Transcripts produced from audio are cached here so a rerun does not re-transcribe.
# Cached as JSON rather than plain text, so segment timestamps survive the round trip.
TRANSCRIPT_DIR = Path("transcripts")


def get_transcript(args) -> dict:
    """Return ``{"text", "segments"}``, transcribing the audio only when necessary.

    ``segments`` is empty for a plain-text transcript, which simply means answers
    cannot be traced back to a position in the audio.
    """
    if args.transcript:
        source = resolve_transcript(args.transcript)
        print(f"Using transcript {source}")
        return load_transcript_file(source)

    if not args.audio:
        if not DEFAULT_TRANSCRIPT.is_file():
            raise SystemExit(
                f"error: no audio file given and {DEFAULT_TRANSCRIPT} does not exist.\n"
                "Pass an audio file: python intel_audio_logic.py your_audio.mp3"
            )
        print(f"Using transcript {DEFAULT_TRANSCRIPT}")
        return load_transcript_file(DEFAULT_TRANSCRIPT)

    audio = Path(args.audio)
    if not audio.is_file():
        raise SystemExit(f"error: audio file not found: {audio}")

    # Reuse a previous transcript unless the audio is newer or --retranscribe was asked for.
    cached = TRANSCRIPT_DIR / f"{audio.stem}.json"
    if (
        cached.is_file()
        and not args.retranscribe
        and cached.stat().st_mtime >= audio.stat().st_mtime
    ):
        print(f"Using cached transcript {cached} (--retranscribe to redo it)")
        return load_transcript_file(cached)

    print(f"Transcribing {audio.name} ...")
    started = time.perf_counter()
    result = transcribe_audio(
        audio,
        model=args.whisper_model,
        device=args.device,
        language=args.language,
        beam_size=args.beam_size,
        batch_size=args.batch_size,
    )
    elapsed = time.perf_counter() - started

    speed = f", {result['duration'] / elapsed:.1f}x realtime" if elapsed > 0 else ""
    print(
        f"Transcribed {format_clock(result['duration'])} of audio in {elapsed:.1f}s"
        f"{speed} on {result['device']}"
    )

    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    # A plain-text copy too, for reading or feeding to anything else.
    cached.with_suffix(".txt").write_text(result["text"], encoding="utf-8")
    print(f"Saved transcript to {cached}")
    return result


def resolve_transcript(value: str) -> Path:
    """Find a transcript from a path, a name in the cache, or a bare stem.

    --transcript is usually pointed at something this script itself wrote into
    transcripts/, so `wrd`, `wrd.json` and `transcripts/wrd.json` all resolve rather
    than failing on a literal path that was never going to exist. A missing suffix
    prefers .json, since that is the form carrying segment timings.
    """
    given = Path(value)
    bare_name = not given.is_absolute() and given.parent == Path(".")

    candidates = [given]
    if bare_name:
        candidates.append(TRANSCRIPT_DIR / given.name)
    if not given.suffix:
        for base in [given] + ([TRANSCRIPT_DIR / given.name] if bare_name else []):
            candidates += [base.with_suffix(".json"), base.with_suffix(".txt")]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise SystemExit(transcript_not_found_message(value))


def transcript_not_found_message(value: str) -> str:
    """Explain what was not found, and show what is actually in the cache."""
    lines = [f"error: transcript not found: {value}"]

    cached = (
        sorted(path for path in TRANSCRIPT_DIR.iterdir() if path.is_file())
        if TRANSCRIPT_DIR.is_dir()
        else []
    )
    if cached:
        lines.append(f"\nAvailable in {TRANSCRIPT_DIR}{os.sep}:")
        for path in cached:
            note = "  (timestamped)" if path.suffix.lower() == ".json" else ""
            lines.append(f"  {path.name}{note}")
        lines.append(
            f"\nPass one as: --transcript {TRANSCRIPT_DIR}{os.sep}<name>"
            "   (or just the name, or the name without its extension)"
        )
    else:
        lines.append(
            f"\nNothing cached in {TRANSCRIPT_DIR}{os.sep} yet. Transcribe some audio first:"
            "\n  python intel_audio_logic.py your_audio.mp3"
        )
    return "\n".join(lines)


def load_transcript_file(path: Path) -> dict:
    """Load a cached JSON transcript, or a plain-text one with no timestamps."""
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return {"text": data.get("text", ""), "segments": data.get("segments", [])}
    return {"text": path.read_text(encoding="utf-8"), "segments": []}


def build_retriever(transcript: dict) -> Retriever:
    """Chunk the transcript and index it for hybrid (vector + keyword) search."""
    segments = transcript.get("segments")
    if segments:
        # Timestamped chunks, so a retrieved chunk can be cited back to the audio.
        documents = chunk_segments(segments)
        print(f"Indexing {len(documents)} timestamped chunks ...")
    else:
        documents = [{"content": chunk} for chunk in chunker(transcript["text"])]
        print(f"Indexing {len(documents)} chunks (no timestamps in this transcript) ...")

    # Generate a vector embedding of the chunks using VoyageAI;
    # see HybridSearchImplementation for more details.
    vector_index = VectorIndex(embedding_fn=embedder)
    # Instantiate the BestMatch 25 class for keyword search
    bm25_index = BM25Index()

    # The class that does the hybrid search; see HybridSearchImplementation for details
    retriever = Retriever(bm25_index, vector_index)

    # Add all chunks to the retriever, which internally passes them along to both indexes
    retriever.add_documents(documents)
    return retriever


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="intel_audio_logic.py",
        description="Transcribe audio (GPU-accelerated) and chat about it with Claude, "
                    "with answers pointing back to where in the audio they came from.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n\n")[1],
    )
    p.add_argument("audio", nargs="?", help="audio/video file to transcribe and chat about")
    p.add_argument(
        "-t", "--transcript", metavar="PATH",
        help=f"chat over an existing transcript instead of transcribing "
             f"(default: {DEFAULT_TRANSCRIPT} when no audio is given)",
    )
    p.add_argument(
        "--retranscribe", action="store_true",
        help="re-transcribe even if a cached transcript exists",
    )

    transcription = p.add_argument_group("transcription (see transcribe.py for more)")
    transcription.add_argument(
        "-m", "--whisper-model", default="base", metavar="NAME",
        help="whisper model size (default: base; large-v3 is more accurate)",
    )
    transcription.add_argument(
        "-d", "--device", default="auto", choices=["auto", "cuda", "cpu", "mps"],
        help="compute device (default: auto -- uses the GPU when present)",
    )
    transcription.add_argument("-l", "--language", help="spoken language code, e.g. en")
    transcription.add_argument(
        "--beam-size", type=int, default=5, metavar="N",
        help="beam width; 1 is greedy and fastest (default: 5)",
    )
    transcription.add_argument(
        "--batch-size", type=int, default=0, metavar="N",
        help="batched GPU decoding, a large speedup, but coarser timestamps "
             "(try 8 or 16; default: off)",
    )

    p.add_argument(
        "--llm-model", default=DEFAULT_MODEL, metavar="NAME",
        help=f"Claude model for answering questions (default: {DEFAULT_MODEL})",
    )
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    transcript = get_transcript(args)
    retriever = build_retriever(transcript)

    # Instantiate the ConversationHandler class with the index it should search
    convo = ConversationHandler(retriever, model=args.llm_model)

    # Start the conversation with the LLM
    try:
        convo.converse_with_LLM()
    except (EOFError, KeyboardInterrupt):
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
