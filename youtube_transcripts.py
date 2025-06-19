
"""
youtube_transcripts.py

CLI utility to clean raw YouTube transcript text (no punctuation/capitalization) into readable
sentences using the OpenAI Chat Completions API.

Example
-------
$ python youtube_transcripts.py raw.txt               # writes raw_clean.txt
$ python youtube_transcripts.py raw.txt --out tidy.txt
$ python youtube_transcripts.py raw.txt --model gpt-4o-mini --temp 0.3

Environment
-----------
OPENAI_API_KEY must be set in environment variables or a `.env` file at project root.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

DEFAULT_MODEL = "gpt-4o-mini"
MAX_MODEL_TOKENS = 8192  # hard‑coded upper bound for gpt‑4o‑mini
CHUNK_MARGIN = 1024      # leave room for prompt + response

load_dotenv()
client = OpenAI()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def estimate_tokens(text: str) -> int:
    """
    Very rough token estimate: average 4 characters per token.
    Good enough to avoid hitting context limits.
    """
    return max(1, len(text) // 4)


def chunk_text(text: str, max_tokens: int) -> List[str]:
    """
    Naively split a long transcript into chunks that fit under max_tokens.
    It respects existing line breaks and tries to split on blank lines first.
    """
    paragraphs: List[str] = text.splitlines(keepends=False)
    chunks: List[str] = []
    current: List[str] = []
    curr_tokens = 0

    for para in paragraphs:
        para_tokens = estimate_tokens(para)
        if curr_tokens + para_tokens > max_tokens:
            chunks.append("\n".join(current))
            current = [para]
            curr_tokens = para_tokens
        else:
            current.append(para)
            curr_tokens += para_tokens

    if current:
        chunks.append("\n".join(current))

    return chunks


def clean_chunk(raw_chunk: str, model: str, temperature: float) -> str:
    """
    Send a single chunk to the Chat API and return cleaned text.
    """
    system_msg = (
        "You are a helpful assistant that restores proper punctuation, capitalisation, "
        "and sentence boundaries to verbatim transcribed speech. "
        "Keep the original wording, do NOT add or remove content, merely fix the grammar."
        "However, please remove the timestamps."
    )

    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": raw_chunk},
        ],
    )
    return resp.choices[0].message.content.strip()


def clean_transcript(raw: str, model: str, temperature: float) -> str:
    """Clean an entire transcript, chunking under model limits."""
    max_tokens_per_chunk = MAX_MODEL_TOKENS - CHUNK_MARGIN
    chunks = chunk_text(raw, max_tokens_per_chunk)
    cleaned_parts: List[str] = []
    for idx, chunk in enumerate(chunks, 1):
        print(f"[{idx}/{len(chunks)}] Processing chunk…", file=sys.stderr, end=" ", flush=True)
        cleaned = clean_chunk(chunk, model, temperature)
        cleaned_parts.append(cleaned)
        print("done.", file=sys.stderr)
    return "\n".join(cleaned_parts)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Clean a YouTube transcript via OpenAI API.")
    parser.add_argument("infile", type=Path, help="Path to raw transcript .txt")
    parser.add_argument("--out", "-o", type=Path, help="Output path (default: infile stem + '_clean.txt')")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model (default: {DEFAULT_MODEL})")
    parser.add_argument("--temp", type=float, default=0.2, help="Sampling temperature (default: 0.2)")
    args = parser.parse_args()

    if not args.infile.exists():
        sys.exit(f"Input file not found: {args.infile}")

    out_path = args.out or args.infile.with_name(f"{args.infile.stem}_clean.txt")

    with args.infile.open("r", encoding="utf-8") as f:
        raw_text = f.read()

    cleaned = clean_transcript(raw_text, args.model, args.temp)

    with out_path.open("w", encoding="utf-8") as f:
        f.write(cleaned)

    print(f"✨ Cleaned transcript written to {out_path}")


if __name__ == "__main__":
    main()
