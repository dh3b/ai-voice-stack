"""
The hybrid chunker emits an early first chunk (clause boundary or first_chunk_max_words,
whichever lands first) then reverts to sentence granularity. We assert the opener
contract and a no-word-lost invariant across all four modes.
"""

import asyncio
import re

import pytest

from config import TTSClientConfig


def _words(text):
    return re.findall(r"[0-9a-z]+", text.lower())


def _chunk_words(chunks):
    return [w for chunk in chunks for w in _words(chunk)]


async def _collect(make_iter, tokens):
    """Push tokens (then the None sentinel) onto a queue and drain the chunker."""
    queue: asyncio.Queue = asyncio.Queue()
    for token in tokens:
        queue.put_nowait(token)
    queue.put_nowait(None)
    chunks = []
    async for chunk in make_iter(queue):
        chunks.append(chunk)
    return chunks


@pytest.fixture
def tts(make_tts_client):
    return make_tts_client(TTSClientConfig())


async def test_hybrid_first_chunk_stops_at_clause_boundary(tts):
    # Streamed token-by-token; the first clause boundary (",") lands within the word cap.
    tokens = ["I ", "think", ", ", "therefore ", "I ", "am", ". ", "So ", "it ", "goes", "."]
    chunks = await _collect(tts._iter_hybrid, tokens)

    assert chunks[0] == "I think,"
    assert len(chunks[0].split()) <= tts._config.first_chunk_max_words
    assert chunks[0][-1] in ",;:.?!"  # ends at a clause boundary
    assert _chunk_words(chunks) == _words("".join(tokens))  # nothing dropped


async def test_hybrid_first_chunk_falls_back_to_word_cap(tts):
    tokens = ["one", " two", " three", " four", " five", " six", " seven"]
    chunks = await _collect(tts._iter_hybrid, tokens)

    assert chunks[0] == "one two three four five"
    assert len(chunks[0].split()) == tts._config.first_chunk_max_words
    assert _chunk_words(chunks) == _words("".join(tokens))


@pytest.mark.parametrize(
    "tokens", [["hello"],["..."],["the quick brown"]]
)
async def test_hybrid_edge_cases_preserve_words(tts, tokens):
    chunks = await _collect(tts._iter_hybrid, tokens)
    assert _chunk_words(chunks) == _words("".join(tokens))
    assert all(c.strip() for c in chunks)  # never emits an empty/whitespace chunk


async def test_chars_mode_is_byte_exact(make_tts_client):
    client = make_tts_client(TTSClientConfig(chunk_mode="chars", chunk_size=3))
    text = "Hello, world! How are you?"
    tokens = list(text)  # stream char by char
    chunks = await _collect(client._iter_chars, tokens)
    assert all(len(c) <= 3 for c in chunks[:-1])
    assert "".join(chunks) == text


async def test_words_mode_preserves_word_sequence(make_tts_client):
    client = make_tts_client(TTSClientConfig(chunk_mode="words", chunk_size=3))
    text = "alpha beta gamma delta epsilon zeta"
    # Stream the way real tokens arrive: each later word carries its own leading space.
    tokens = ["alpha", " beta", " gamma", " delta", " epsilon", " zeta"]
    chunks = await _collect(client._iter_words, tokens)
    assert " ".join(chunks).split() == text.split()
    assert all(len(c.split()) <= 3 for c in chunks)


async def test_sentence_mode_splits_and_preserves_words(make_tts_client):
    client = make_tts_client(TTSClientConfig(chunk_mode="sentence"))
    tokens = ["First ", "sentence. ", "Second ", "one! ", "Third?"]
    chunks = await _collect(client._iter_sentences, tokens)
    assert chunks == ["First sentence", "Second one", "Third"]
    assert _chunk_words(chunks) == _words("".join(tokens))
