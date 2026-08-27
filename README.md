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
| `ConversationHandler.py` | Manages the conversation loop, context retrieval, and LLM calls |
| `HybridSearchImplementation.py` | `VectorIndex`, `BM25Index`, and `Retriever` classes |
| `ChunckAndEmbed.py` | Text chunking and Voyage AI embedding utilities |
| `RAGChat.ipynb` | Jupyter notebook for development and experimentation |
| `Learn_Transformer` | Sample transcript (about Transformer models in ML) |

## Setup

### Prerequisites

- Python 3.9+
- API keys for Anthropic (Claude), Voyage AI, and optionally OpenAI (for Whisper transcription)

### Installation

```bash
pip install anthropic voyageai python-dotenv openai openai-whisper
```

### Environment Variables

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=your_anthropic_api_key
VOYAGE_API_KEY=your_voyage_api_key
OPENAI_API_KEY=your_openai_api_key   # Optional — for Whisper transcription
```

## Usage

```bash
python intel_audio_logic.py
```

This will:
1. Load the transcript from `Learn_Transformer`
2. Chunk it into overlapping sentence windows
3. Build vector and BM25 search indexes over the chunks
4. Start an interactive conversation loop

Type your question at the prompt. Type `exit` to quit.

**Alternatively**, run `RAGChat.ipynb` in Jupyter for step-by-step exploration.

## Key Components

### `ChunckAndEmbed.py`

- **`chunk_by_sentence(text, max_sentences_per_chunk=5, overlap_sentences=1)`** — Splits text into overlapping sentence chunks using regex.
- **`generate_embedding(chunks, model="voyage-3-large")`** — Calls Voyage AI to embed a string or list of strings.

### `HybridSearchImplementation.py`

- **`VectorIndex`** — Stores document embeddings and searches by cosine similarity.
- **`BM25Index`** — Tokenizes documents and ranks them using the BM25 algorithm.
- **`Retriever`** — Wraps one or more indexes, queries all of them, and merges results via Reciprocal Rank Fusion.

### `ConversationHandler.py`

- **`converse_with_LLM()`** — Main loop: reads user input, retrieves relevant chunks, augments the prompt, calls Claude, and stores the response in conversation history.

## Models Used

| Purpose | Model |
|---------|-------|
| LLM responses | Claude (via Anthropic API) |
| Text embeddings | `voyage-3-large` (Voyage AI) |
| Audio transcription | `whisper` (OpenAI) |

## Notes

- Audio transcription via Whisper is currently commented out in `intel_audio_logic.py`; the pipeline reads from a pre-existing transcript file. To add your own audio, transcribe it first and save the output as a plain text file, then point the script at it.
- The `.gitignore` excludes `.env` to protect API keys — never commit that file.
