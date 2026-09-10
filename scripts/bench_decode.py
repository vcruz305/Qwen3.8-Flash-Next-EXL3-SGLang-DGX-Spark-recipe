#!/usr/bin/env python3
"""Decode tok/s benchmark against a running SGLang OpenAI-compatible server.

Usage:
    python bench_decode.py [--base-url http://127.0.0.1:30000] [--model Qwen3.8-Flash-Next]
"""
import argparse
import json
import time
import urllib.request

PROMPT = "The capital of France is"


def run(base_url: str, model: str, max_tokens: int, temperature: float = 0.0) -> dict:
    payload = {
        "model": model,
        "prompt": PROMPT,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as r:
        body = json.loads(r.read().decode())
    dt = time.perf_counter() - t0
    usage = body.get("usage", {})
    ct = usage.get("completion_tokens") or max_tokens
    text = body["choices"][0]["text"]
    return {"max_tokens": max_tokens, "wall_s": dt, "tok_s": ct / dt, "text": text}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--model", default="Qwen3.8-Flash-Next")
    ap.add_argument("--max-tokens", type=int, nargs="+", default=[32, 64, 128, 256])
    args = ap.parse_args()

    for n in args.max_tokens:
        r = run(args.base_url, args.model, n)
        print(
            f"n={r['max_tokens']:<4} wall={r['wall_s']:.3f}s "
            f"tok/s={r['tok_s']:.2f}  text={r['text'][:70]!r}"
        )


if __name__ == "__main__":
    main()
