
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


# How much goes in one request is decided by rate limits, not by the model: Voyage's
# free tier allows 10k tokens/min, so 8k leaves headroom while still filling a
# request on a paid account. Override per call with max_tokens_per_request.
DEFAULT_MAX_TOKENS_PER_REQUEST = 8000

# Retried with backoff; everything else (auth, malformed request) fails immediately
# because waiting cannot help.
TRANSIENT_ERRORS = (
    voyageai.error.RateLimitError,
    voyageai.error.ServerError,
    voyageai.error.ServiceUnavailableError,
    voyageai.error.APIConnectionError,
)


def _estimate_tokens(text: str) -> int:
    """Fallback token count at roughly 4 characters per token.

    Only used when the real tokenizer is unavailable. It runs about 12% over on
    English prose but around 45% *under* on CJK and code, and underestimating is
    the direction that gets a request rejected.
    """
    return max(1, len(text) // 4)


def _count_tokens(texts, model):
    """Per-text token counts from Voyage's own tokenizer, falling back to the estimate.

    The tokenizer is fetched from the Hugging Face Hub the first time it is used, so
    this falls back to _estimate_tokens when it cannot be loaded (offline, say).
    Counting is local and cheap once cached: a few hundred chunks take well under a
    tenth of a second.
    """
    try:
        return [vog_client.count_tokens([text], model=model) for text in texts]
    except Exception:
        return [_estimate_tokens(text) for text in texts]


def _batch_by_tokens(texts, counts, max_tokens):
    """Group texts into requests that each stay within max_tokens.

    A text larger than the entire budget cannot be made to fit, so it is sent on its
    own with a warning, rather than silently dropped or folded into a request that
    would be rejected. Voyage truncates anything past the model's context window.
    """
    batch, budget = [], 0
    for text, tokens in zip(texts, counts):
        if tokens > max_tokens:
            if batch:
                yield batch
                batch, budget = [], 0
            print(
                f"  Warning: one chunk is {tokens} tokens, over the "
                f"{max_tokens}-token request budget; sending it on its own"
            )
            yield [text]
            continue

        if batch and budget + tokens > max_tokens:
            yield batch
            batch, budget = [], 0
        batch.append(text)
        budget += tokens

    if batch:
        yield batch


def _embed_batch(batch, model, input_type, max_retries=6):
    """Embed one batch, backing off on rate limits and transient server errors.

    Retrying rather than pre-emptively sleeping keeps paid accounts running at full
    speed, and slows down only as much as a restricted account actually requires.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return vog_client.embed(batch, model=model, input_type=input_type).embeddings
        except TRANSIENT_ERRORS as exc:
            if attempt == max_retries:
                raise
            # Rate limits are measured per minute, so short retries are pointless;
            # a dropped connection is worth retrying straight away.
            base = 20.0 if isinstance(exc, voyageai.error.RateLimitError) else 2.0
            delay = min(base * (1.5 ** (attempt - 1)), 90.0)
            print(
                f"  Voyage {type(exc).__name__}; waiting {delay:.0f}s "
                f"(attempt {attempt}/{max_retries - 1})"
            )
            time.sleep(delay)

    raise AssertionError("unreachable")


def generate_embedding(
    chunks,
    model="voyage-3-large",
    input_type="query",
    max_tokens_per_request=DEFAULT_MAX_TOKENS_PER_REQUEST,
):
    is_list = isinstance(chunks, list)
    input = chunks if is_list else [chunks]

    counts = _count_tokens(input, model)
    batches = list(_batch_by_tokens(input, counts, max_tokens_per_request))

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
