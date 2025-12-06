"""
Analyze a PDF with the OpenAI API.

Usage:
  python analyze_pdf.py path/to/file.pdf "Give me relevant cases for healthcare mergers"
"""
import os, argparse, itertools, textwrap, sys
from typing import List
import pdfplumber        # pdfplumber==0.10.*
import tiktoken          # tiktoken==0.6.*
from dotenv import load_dotenv
from openai import OpenAI

# ---------- configuration ----------
MODEL          = "gpt-4o-mini"           # or gpt-4o, gpt-4o-128k, etc.
MAX_TOKENS_IN  = 4096                    # tokens allowed in *prompt* for this model
TOKENS_RESERVE = 1024                    # leave room for the *answer*
CHUNK_TOKENS   = MAX_TOKENS_IN - TOKENS_RESERVE
OVERLAP        = 200                     # overlapping context to keep coherence
SYSTEM_MSG     = (
    "You are an expert antitrust analyst. "
    "Given the following excerpt from a Federal Trade Commission annual report, "
    "answer the user's query in clear bullet-point form. "
    "Cite page numbers when possible."
)
# ------------------------------------

def extract_text(pdf_path: str) -> List[str]:
    """Return a list with one string per PDF page."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return pages

def chunks_by_tokens(pages: List[str]) -> List[str]:
    enc = tiktoken.encoding_for_model(MODEL)
    buffer, buf_tokens, start_page = [], 0, 1
    for i, page in enumerate(pages, 1):
        tokens = len(enc.encode(page))
        # flush if adding this page would overflow
        if buf_tokens + tokens > CHUNK_TOKENS and buffer:
            yield f"[pages {start_page}-{i-1}]\n" + "\n".join(buffer)
            buffer, buf_tokens, start_page = [], 0, i
        buffer.append(f"[p.{i}]  " + page)
        buf_tokens += tokens
    if buffer:
        yield f"[pages {start_page}-{len(pages)}]\n" + "\n".join(buffer)

def ask_openai(client: OpenAI, query: str, context: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",   "content": f"{query}\n\n---\n{context}"}
    ]
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2
    )
    return resp.choices[0].message.content.strip()

def main():
    load_dotenv(override=True)                 # pulls OPENAI_API_KEY from .env
    client = OpenAI()

    parser = argparse.ArgumentParser(description="Summarize or query a PDF with GPT")
    parser.add_argument("pdf",   help="Path to PDF file")
    parser.add_argument("query", help="Question or prompt")
    args = parser.parse_args()

    print("⏳ Extracting text…")
    pages = extract_text(args.pdf)
    if not any(pages):
        sys.exit("❌ No extractable text found in the PDF.")

    print(f"✅ Got {len(pages)} pages. Chunking & querying GPT…\n")
    aggregated_answers = []
    for idx, chunk in enumerate(chunks_by_tokens(pages), 1):
        print(f"  ▶️  chunk {idx} …", end="", flush=True)
        answer = ask_openai(client, args.query, chunk)
        aggregated_answers.append(answer)
        print("done.")

    # Simple merge-and-shrink pass so you don't get 10 near-duplicate bullets
    merged = "\n".join(aggregated_answers)
    dedup = "\n".join(dict.fromkeys(  # preserves order
        line.strip() for line in merged.splitlines() if line.strip()
    ))
    print("\n================ FINAL ANSWER ================\n")
    print(textwrap.fill(dedup, width=100, replace_whitespace=False))

if __name__ == "__main__":
    main()
