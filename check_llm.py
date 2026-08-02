"""
check_llm.py — Standalone diagnostic for whatever LLM endpoint is configured
in config.py (LLM_BASE_URL / LLM_MODEL / GROQ_API_KEY).

Run: python check_llm.py

Isolates where things break: basic reachability, a non-streaming call, and a
streaming call (timing to first token) — independent of the FastAPI app, so
you can tell whether a hang is network/endpoint-side vs. something in api.py.
"""

import time

import requests
from openai import OpenAI

from config import GROQ_API_KEY, LLM_BASE_URL, LLM_MODEL

TIMEOUT_S = 20


def check_reachability():
    print(f"[1/3] GET {LLM_BASE_URL}/models ...")
    start = time.perf_counter()
    try:
        resp = requests.get(
            f"{LLM_BASE_URL}/models",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            timeout=TIMEOUT_S,
        )
        elapsed = time.perf_counter() - start
        print(f"      -> {resp.status_code} in {elapsed:.2f}s")
    except Exception as exc:
        elapsed = time.perf_counter() - start
        print(f"      -> FAILED after {elapsed:.2f}s: {exc!r}")


def check_non_streaming(client: OpenAI):
    print("[2/3] Non-streaming chat completion ...")
    start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "Say hi in one word."}],
            max_tokens=20,
            stream=False,
        )
        elapsed = time.perf_counter() - start
        print(f"      -> got reply in {elapsed:.2f}s: {response.choices[0].message.content!r}")
    except Exception as exc:
        elapsed = time.perf_counter() - start
        print(f"      -> FAILED after {elapsed:.2f}s: {exc!r}")


def check_streaming(client: OpenAI):
    print("[3/3] Streaming chat completion ...")
    start = time.perf_counter()
    first_token_at = None
    chunks = 0
    try:
        stream = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "Say hi in one word."}],
            max_tokens=20,
            stream=True,
        )
        for chunk in stream:
            if first_token_at is None:
                first_token_at = time.perf_counter() - start
                print(f"      -> first token at {first_token_at:.2f}s")
            chunks += 1
        total = time.perf_counter() - start
        print(f"      -> stream finished in {total:.2f}s ({chunks} chunks)")
    except Exception as exc:
        elapsed = time.perf_counter() - start
        print(f"      -> FAILED after {elapsed:.2f}s: {exc!r}")


if __name__ == "__main__":
    if not GROQ_API_KEY:
        raise SystemExit("GROQ_API_KEY not set — check .env_local")

    check_reachability()
    client = OpenAI(base_url=LLM_BASE_URL, api_key=GROQ_API_KEY, timeout=TIMEOUT_S)
    check_non_streaming(client)
    check_streaming(client)
