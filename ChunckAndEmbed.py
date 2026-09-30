
import re
import time

import voyageai
import voyageai.error
vog_client = voyageai.Client()

def chunk_by_sentence(text, max_sentences_per_chunk=5, overlap_sentences=1):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    
    chunks = []
    start_idx = 0
    
    while start_idx < len(sentences):
        end_idx = min(start_idx + max_sentences_per_chunk, len(sentences))
        current_chunk = sentences[start_idx:end_idx]
        chunks.append(" ".join(current_chunk))
        
        start_idx += max_sentences_per_chunk - overlap_sentences
        
        if start_idx < 0:
            start_idx = 0
    
    return chunks


# Voyage rejects a request whose token count exceeds the per-minute budget, so long
# transcripts have to be embedded in pieces. 8k leaves headroom under the 10k/min
# free-tier limit while still filling a request on a paid one.
MAX_TOKENS_PER_REQUEST = 8000


def _estimate_tokens(text: str) -> int:
    """Rough token count -- about 4 characters per token is close enough to batch on."""
    return max(1, len(text) // 4)


def _batch_by_tokens(texts, max_tokens=MAX_TOKENS_PER_REQUEST):
    """Split texts into groups that should each fit in one Voyage request."""
    batch, budget = [], 0
    for text in texts:
        tokens = _estimate_tokens(text)
        if batch and budget + tokens > max_tokens:
            yield batch
            batch, budget = [], 0
        batch.append(text)
        budget += tokens
    if batch:
        yield batch


def _embed_batch(batch, model, input_type, max_retries=6):
    """Embed one batch, backing off when Voyage reports a rate limit.

    Retrying rather than pre-emptively sleeping keeps paid accounts running at full
    speed, and slows down only as much as a restricted account actually requires.
    """
    delay = 20.0  # the limit is per minute, so short retries are pointless
    for attempt in range(max_retries):
        try:
            return vog_client.embed(batch, model=model, input_type=input_type).embeddings
        except voyageai.error.RateLimitError:
            if attempt == max_retries - 1:
                raise
            print(
                f"  Voyage rate limit hit; waiting {delay:.0f}s "
                f"(attempt {attempt + 1}/{max_retries - 1})"
            )
            time.sleep(delay)
            delay = min(delay * 1.5, 90.0)
    raise AssertionError("unreachable")


def generate_embedding(chunks, model="voyage-3-large", input_type="query"):
    is_list = isinstance(chunks, list)
    input = chunks if is_list else [chunks]

    batches = list(_batch_by_tokens(input))
    embeddings = []
    for i, batch in enumerate(batches, start=1):
        if len(batches) > 1:
            print(f"  Embedding batch {i}/{len(batches)} ({len(batch)} chunks)")
        embeddings.extend(_embed_batch(batch, model, input_type))

    return embeddings if is_list else embeddings[0]


def chunk_segments(segments, max_sentences_per_chunk=5, overlap_sentences=1):
    """Chunk timestamped transcript segments, keeping when each chunk was spoken.

    Takes the ``segments`` list produced by transcribe.py and returns a list of
    ``{"content", "start", "end"}`` dicts, so a retrieved chunk can be traced back
    to a position in the audio.

    A Whisper segment often spans several sentences (30 seconds or so when batched
    decoding is on), which is too coarse to cite. Sentence times are therefore
    interpolated across the segment by character position -- approximate, but far
    tighter than attributing every sentence to the whole segment.
    """
    sentences = []
    for segment in segments:
        text = segment["text"].strip()
        if not text:
            continue
        parts = [part for part in re.split(r"(?<=[.!?])\s+", text) if part]
        span = max(segment["end"] - segment["start"], 0.0)
        chars = sum(len(part) for part in parts) or 1

        consumed = 0
        for part in parts:
            start = segment["start"] + span * (consumed / chars)
            consumed += len(part)
            end = segment["start"] + span * (consumed / chars)
            sentences.append({"text": part, "start": start, "end": end})

    # Same windowing as chunk_by_sentence, but carrying the timings along.
    step = max(max_sentences_per_chunk - overlap_sentences, 1)
    chunks = []
    for start_idx in range(0, len(sentences), step):
        window = sentences[start_idx:start_idx + max_sentences_per_chunk]
        if not window:
            break
        chunks.append({
            "content": " ".join(s["text"] for s in window),
            "start": window[0]["start"],
            "end": window[-1]["end"],
        })
        if start_idx + max_sentences_per_chunk >= len(sentences):
            break

    return chunks
