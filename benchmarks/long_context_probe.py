#!/usr/bin/env python3
"""Bounded long-context retrieval probe for an OpenAI-compatible local server.

The probe creates deterministic, varied archive records, inserts five unique
needles through the prompt, requests all values, and records token counts and
timings as JSON. It intentionally generates only one request at a time.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.error
import urllib.request


WORDS = (
    "amber birch cedar delta ember frost granite harbor indigo juniper kinetic "
    "lantern maple nickel orchid pebble quartz river saffron timber umber violet "
    "willow xenon yellow zephyr atlas beacon copper drift elm fjord galaxy helix "
    "island jasper kelp lunar meadow north opal prairie quiet ridge solar tundra"
).split()

NEEDLES = [
    ("ALPHA", "K7-VIOLET-314159"),
    ("BRAVO", "M2-CEDAR-271828"),
    ("CHARLIE", "R9-QUARTZ-161803"),
    ("DELTA", "T4-HARBOR-141421"),
    ("ECHO", "W8-JUNIPER-173205"),
]

# These probes target an explicitly supplied local/Tailscale server. Do not let
# workstation HTTP proxy variables redirect multi-megabyte prompts elsewhere.
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post_json(base: str, path: str, body: dict, timeout: float) -> dict:
    data = json.dumps(body, separators=(",", ":")).encode()
    req = urllib.request.Request(
        base.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with OPENER.open(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:4000]}") from exc


def record(i: int) -> str:
    # Deterministic but sufficiently varied to avoid an unrealistically tiny PLE
    # hot set from one repeated sentence.
    x = (i * 1103515245 + 12345) & 0x7FFFFFFF
    chosen = [WORDS[(x >> (j % 19) ^ i * (j + 3)) % len(WORDS)] for j in range(18)]
    return f"Archive record {i:07d}: " + " ".join(chosen) + ".\n"


def make_prompt(target_tokens: int, chars_per_token: float) -> str:
    intro = (
        "Read the archive carefully. Five IMPORTANT records contain secret values. "
        "Ignore ordinary records and retain every labelled secret value.\n"
    )
    question = (
        "\nEND OF ARCHIVE. Return the five secret values for ALPHA, BRAVO, CHARLIE, "
        "DELTA, and ECHO in that order. Output only the values separated by commas.\n"
    )
    target_chars = max(10_000, int(target_tokens * chars_per_token))
    needle_at = [int(target_chars * fraction) for fraction in (0.05, 0.25, 0.50, 0.75, 0.95)]
    parts = [intro]
    chars = len(intro)
    inserted = 0
    i = 0
    while chars < target_chars - len(question):
        if inserted < len(NEEDLES) and chars >= needle_at[inserted]:
            label, value = NEEDLES[inserted]
            line = (
                f"IMPORTANT RECORD {label}: The secret value for {label} is {value}. "
                f"Remember {value} exactly.\n"
            )
            parts.append(line)
            chars += len(line)
            inserted += 1
        line = record(i)
        parts.append(line)
        chars += len(line)
        i += 1
    while inserted < len(NEEDLES):
        label, value = NEEDLES[inserted]
        parts.append(f"IMPORTANT RECORD {label}: The secret value for {label} is {value}.\n")
        inserted += 1
    parts.append(question)
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="qwen3.8-flash-next")
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--max-output-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--output")
    args = parser.parse_args()

    # Calibrate text density on representative records without submitting a
    # second ultra-long request merely to count tokens.
    sample = "".join(record(i) for i in range(400))
    calibration = post_json(
        args.base_url,
        "/tokenize",
        {"model": args.model, "prompt": sample},
        min(args.timeout, 120),
    )
    sample_tokens = calibration.get("count") or len(calibration.get("tokens", []))
    if not sample_tokens:
        raise RuntimeError(f"tokenizer response lacks a count: {calibration.keys()}")
    chars_per_token = len(sample) / sample_tokens
    prompt = make_prompt(args.target_tokens, chars_per_token)

    started = time.time()
    response = post_json(
        args.base_url,
        "/v1/chat/completions",
        {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "top_p": 1,
            "max_tokens": args.max_output_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        args.timeout,
    )
    elapsed = time.time() - started
    message = response["choices"][0]["message"]
    content = message.get("content") or ""
    found = {label: value in content for label, value in NEEDLES}
    usage = response.get("usage", {})
    result = {
        "target_tokens": args.target_tokens,
        "prompt_characters": len(prompt),
        "calibrated_characters_per_token": chars_per_token,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "elapsed_seconds": elapsed,
        "estimated_prefill_tokens_per_second": (
            usage.get("prompt_tokens", 0) / elapsed if elapsed else math.nan
        ),
        "needles_found": found,
        "score": f"{sum(found.values())}/{len(found)}",
        "response": content,
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(rendered + "\n")
    if not all(found.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
