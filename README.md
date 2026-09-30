# Intelligent Audio

A Retrieval-Augmented Generation (RAG) chatbot that processes audio content and enables intelligent, context-aware conversations about it. The system transcribes audio, builds a hybrid search index over the transcript, and uses Claude (Anthropic) to answer questions grounded in the audio content.

## How It Works

```
Audio File
    ↓
Whisper Transcription → Text Transcript
    ↓
Sentence-based Chunking (with overlap)
    ↓
    ┌──────────────────────┬──────────────────────┐
    │   Vector Index       │    BM25 Index         │
    │   (Voyage AI         │    (Keyword /         │
    │    Embeddings)       │     TF-IDF style)     │
    └──────────────────────┴──────────────────────┘
                    ↓  Hybrid Search (RRF)
              User Question
                    ↓
         Context-Augmented Prompt
                    ↓
            Claude API Response
                    ↓
         Conversation History Loop
```

The system combines **semantic vector search** and **BM25 keyword search**, merging their results via **Reciprocal Rank Fusion (RRF)** for more robust retrieval before sending context to Claude.

## Project Structure

| File | Purpose |
|------|---------|
| `intel_audio_logic.py` | Main entry point — orchestrates the full pipeline |
| `transcribe.py` | Standalone GPU-accelerated transcription CLI (audio → text/SRT/VTT/JSON) |
| `ConversationHandler.py` | Manages the conversation loop, context retrieval, and LLM calls |
| `HybridSearchImplementation.py` | `VectorIndex`, `BM25Index`, and `Retriever` classes |
| `ChunckAndEmbed.py` | Text chunking and Voyage AI embedding utilities |
| `RAGChat.ipynb` | Jupyter notebook for development and experimentation |
| `Learn_Transformer` | Sample transcript (about Transformer models in ML) |

## Setup

### Prerequisites

- Python 3.9+
- API keys for Anthropic (Claude) and Voyage AI
- Optional: an NVIDIA GPU (CUDA) or Apple Silicon GPU for fast local transcription — no API key needed, Whisper runs on-device

### Installation

```bash
pip install anthropic voyageai python-dotenv
pip install faster-whisper                       # transcription (GPU + CPU)
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12  # only for NVIDIA GPUs
```

`openai-whisper` also works as a fallback backend (`pip install openai-whisper`) and is
the only option for Apple Silicon GPUs (MPS).

### Environment Variables

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=your_anthropic_api_key
VOYAGE_API_KEY=your_voyage_api_key
```

## Usage

Point it at an audio file and start asking questions:

```bash
python intel_audio_logic.py lecture.mp3
```

```
Transcribing lecture.mp3 ...
Transcribed 5:50 of audio in 8.7s, 40.5x realtime on cuda
Saved transcript to transcripts\lecture.json
Indexing 21 timestamped chunks ...

Ask about the audio (or 'exit' to quit): what was said about attention?

Attention is the mechanism that lets the model weigh every word in the
sequence at once rather than reading strictly in order. This was said at
2:31-2:58 in the audio.

  [audio 2:31-2:58]
```

Transcription runs on the GPU when one is available (see
[`transcribe.py`](#transcription-transcribepy)), and the transcript is cached under
`transcripts/`, so asking more questions later does not re-transcribe the audio.

```bash
# chat over the cached transcript (no re-transcription)
python intel_audio_logic.py lecture.mp3

# force a fresh transcription, e.g. after switching model
python intel_audio_logic.py lecture.mp3 --retranscribe --whisper-model large-v3

# faster transcription with batched GPU decoding
python intel_audio_logic.py lecture.mp3 --batch-size 16

# chat over a transcript you already have
python intel_audio_logic.py --transcript Learn_Transformer

# a cached transcript by name -- these are all equivalent
python intel_audio_logic.py --transcript wrd
python intel_audio_logic.py --transcript wrd.json
python intel_audio_logic.py --transcript transcripts/wrd.json

# no arguments: uses the bundled Learn_Transformer transcript
python intel_audio_logic.py
```

| Flag | Purpose |
|------|---------|
| `-t, --transcript` | Chat over an existing transcript instead of transcribing. Accepts a path, or a name from `transcripts/` with or without its extension; a bare name prefers the timestamped `.json` |
| `--retranscribe` | Re-transcribe even if a cached transcript exists |
| `-m, --whisper-model` | Whisper model size (default `base`) |
| `-d, --device` | `auto` (default), `cuda`, `mps`, `cpu` |
| `-l, --language` | Language code, e.g. `en` |
| `--beam-size` / `--batch-size` | Transcription speed/accuracy knobs |
| `--llm-model` | Claude model used to answer (default `claude-haiku-4-5`) |

Each run does the following:

1. Transcribe the audio on the GPU, or reuse the cached transcript
2. Chunk it into overlapping sentence windows
3. Build vector and BM25 search indexes over the chunks
4. Start an interactive conversation loop — type `exit` to quit

**Alternatively**, run `RAGChat.ipynb` in Jupyter for step-by-step exploration.

### Finding when something was said

Answers point back to where in the audio the supporting context came from. Every
chunk carries a start and end time, which is handed to Claude alongside the text and
printed under each answer as `[audio 2:31-2:58]` — so you can jump straight to that
point in the recording. Asking "when did they mention X?" works directly.

Timestamps come from Whisper's segments, which `chunk_segments()` narrows down: a
segment can span 30 seconds and several sentences, so sentence times are interpolated
across it by character position. Two things affect how tight the citations are:

- **`--batch-size`** makes transcription much faster but produces coarser (~30 s)
  segments. Leave it off when you care more about pinpointing a quote than speed.
- **Larger models** segment more accurately, so `--whisper-model large-v3` gives
  better boundaries than `base`.

Transcripts are cached as `transcripts/<name>.json` (segments and all) plus a
`.txt` copy for reading. `--transcript` looks in that directory too, so
`--transcript wrd` finds `transcripts/wrd.json`; if nothing matches, it lists what
is cached. A plain-text transcript passed via `--transcript` has no
timing, so answers from it simply come without a citation.

For subtitles rather than chat, `transcribe.py -f srt` or `-f vtt` writes standard
timestamped subtitle files, and `-f json` gives every segment with its times.

## Transcription (`transcribe.py`)

A standalone CLI that turns audio into a transcript, using the GPU when one is
available. It picks its backend and device automatically, so the same command works
on a CUDA laptop, an Apple Silicon Mac, or a CPU-only server.

Check what the current machine will use:

```bash
python transcribe.py --list-devices
```

```
Backends:
  faster-whisper  installed
  openai-whisper  not installed
Devices:
  CUDA GPUs       1 -> NVIDIA GeForce RTX 4050 Laptop GPU (6141 MiB)
  Apple MPS       not available
  CPU threads     20

Selected: backend=faster-whisper device=cuda compute_type=float16
```

### Usage

```bash
# transcript to stdout
python transcribe.py audio.mp3

# feed the RAG pipeline: write the transcript where intel_audio_logic.py reads it
python transcribe.py audio.mp3 --model large-v3 -o Learn_Transformer

# subtitles, with timestamps
python transcribe.py lecture.m4a --output-format srt -o lecture.srt

# batch a folder into a directory of transcripts
python transcribe.py recordings/*.mp3 -o transcripts/

# maximum GPU throughput (batched decoding, greedy search)
python transcribe.py audio.mp3 --model large-v3 --batch-size 16 --beam-size 1

# force CPU, e.g. on a box with no GPU or a busy one
python transcribe.py audio.mp3 --device cpu --model small
```

### Options

| Flag | Purpose |
|------|---------|
| `-m, --model` | Model size: `tiny` … `base` (default) … `large-v3`, `turbo`, `distil-large-v3` |
| `-o, --output` | Output file, or a directory (implied with several inputs). Omit for stdout |
| `-f, --output-format` | `txt` (default), `json` (with timestamps), `srt`, `vtt` |
| `-d, --device` | `auto` (default), `cuda`, `mps`, `cpu` |
| `--backend` | `auto` (default, prefers faster-whisper), `faster-whisper`, `openai-whisper` |
| `--compute-type` | Quantization: `auto` (float16 on GPU, int8 on CPU), `float16`, `int8`, … |
| `-l, --language` | Language code, e.g. `en`. Auto-detected if omitted |
| `--task` | `transcribe` (default) or `translate` (to English) |
| `--beam-size` | Beam width; `1` is greedy and fastest (default `5`) |
| `--batch-size` | faster-whisper batched decoding — a large GPU speedup (try `8`–`16`) |
| `--no-vad` | Disable voice-activity filtering of silence |
| `-v, --verbose` | Print each segment as it is decoded |
| `-q, --quiet` | Suppress progress output |

Progress, timings and a realtime-factor summary go to **stderr**, so `stdout` stays a
clean transcript and can be piped.

### Measured speedup

Same 5:50 clip, `base` model, RTX 4050 Laptop (6 GB) vs. a 20-thread CPU:

| Command | Time | Speed |
|---------|------|-------|
| `--device cpu` (int8) | 83.4 s | 4.2x realtime |
| GPU default (float16, beam 5) | 25.1 s | 13.9x realtime |
| `--batch-size 16 --beam-size 1` | **8.0 s** | **43.8x realtime** |

`large-v3` also fits in 6 GB: 23.5 s at `--batch-size 8` (14.9x realtime), with
noticeably better punctuation and fewer stutters than `base`.

### Picking a model for your GPU

`faster-whisper` with `float16` needs roughly:

| Model | VRAM | Notes |
|-------|------|-------|
| `base` | ~0.5 GB | Fast, fine for clean speech |
| `small` | ~1 GB | Good default balance |
| `medium` | ~2.5 GB | |
| `large-v3` | ~3.5 GB | Best accuracy; fits in 6 GB |
| `distil-large-v3` | ~1.5 GB | Near-`large-v3` quality, much faster |

If a model does not fit, the script reports the load failure and suggests
`--device cpu`. You can also trade accuracy for memory with `--compute-type int8_float16`.

### Troubleshooting

- **`Library cublas64_12.dll is not found` / `libcublas.so.12: cannot open shared object file`** —
  the CUDA runtime is missing. `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`.
  The script locates those wheels inside `site-packages` and preloads them itself, so
  no `PATH`/`LD_LIBRARY_PATH` setup is needed.
- **`ConnectError` / connection reset while downloading a model** — the Hugging Face
  transfer backend can choke on the multi-GB `large-*` models. Retry with
  `HF_HUB_DISABLE_XET=1`. Models are cached after the first successful download.
- **GPU detected but `--device cpu` is selected** — with the `openai-whisper` backend
  this means PyTorch is a CPU-only build. Check with
  `python -c "import torch; print(torch.version.cuda)"`; if it prints `None`, either
  use `--backend faster-whisper` or reinstall PyTorch with CUDA support.
- **Out of memory** — the script suggests a smaller `--model`, a smaller
  `--batch-size`, or `--compute-type int8_float16`.
- **`open() got an unexpected keyword argument 'metadata_errors'`** — PyAV 19 broke
  compatibility with `faster-whisper` 1.2. Pin it: `pip install "av<19"`.
- **`voyageai.error.RateLimitError`** — Voyage's free tier allows only 3 requests
  and 10K tokens per minute, which a long transcript exceeds. `generate_embedding()`
  splits the work into ~8K-token requests and backs off when the limit is hit, so
  indexing a two-hour recording still completes, just slowly. Adding a payment
  method to the Voyage account removes the wait; pass a larger
  `max_tokens_per_request` to take advantage of it.
- **One chunk is over the request budget** — printed as a warning and sent on its
  own. Voyage truncates anything past the model's context window, so an unusually
  long chunk may be embedded only in part.

### Using it from Python

```python
from transcribe import transcribe_audio

result = transcribe_audio("audio.mp3", model="large-v3")   # device auto-detected
print(result["text"], result["language"], result["duration"])
```

## Key Components

### `ChunckAndEmbed.py`

- **`chunk_by_sentence(text, max_sentences_per_chunk=5, overlap_sentences=1)`** — Splits text into overlapping sentence chunks using regex.
- **`generate_embedding(chunks, model="voyage-3-large", input_type="query", max_tokens_per_request=8000)`**
  — Embeds a string or list of strings. Counts tokens with Voyage's own tokenizer,
  splits the work into requests that fit `max_tokens_per_request`, and retries with
  backoff on rate limits and transient server errors. Raise the budget on a paid
  account: a 32K-token transcript goes out as 1 request at `40000` instead of 5 at
  the default.
- **`chunk_segments(segments, ...)`** — Chunks timestamped transcript segments,
  interpolating sentence times across each segment so a chunk can be cited back to
  a position in the audio.

### `HybridSearchImplementation.py`

- **`VectorIndex`** — Stores document embeddings and searches by cosine similarity.
- **`BM25Index`** — Tokenizes documents and ranks them using the BM25 algorithm.
- **`Retriever`** — Wraps one or more indexes, queries all of them, and merges results via Reciprocal Rank Fusion.

### `ConversationHandler.py`

- **`ConversationHandler(retriever, client=None, model=...)`** — Takes the search index and Anthropic client as arguments, so the class does not depend on the entry point's globals.
- **`converse_with_LLM()`** — Main loop: reads user input, retrieves relevant chunks, augments the prompt, calls Claude, and stores the response in conversation history. Runs until you type `exit`.

## Models Used

| Purpose | Model |
|---------|-------|
| LLM responses | Claude (via Anthropic API) |
| Text embeddings | `voyage-3-large` (Voyage AI) |
| Audio transcription | Whisper via `faster-whisper` (CTranslate2), GPU-accelerated |

## Notes

- `intel_audio_logic.py` transcribes audio itself via `transcribe.py`, caching results in `transcripts/`. Use `transcribe.py` directly when you only want a transcript (or subtitles) without the chat loop.
- The `.gitignore` excludes `.env` to protect API keys — never commit that file.
