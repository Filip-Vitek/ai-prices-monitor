#!/usr/bin/env python3
"""
AI frontier-lab list-price tracker.

Fetches official list prices (USD per 1M tokens, standard pay-as-you-go tier)
for chat/completion models from major frontier labs in the USA, EU and China,
and maintains an append-only time series of price changes.

Primary source: LiteLLM's community-maintained model price database, which
mirrors the official vendor pricing pages and is updated within hours of
vendor changes:
  https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json

Data layout (relative to repo root):
  data/latest.json   - current snapshot of all tracked models
  data/prices.csv    - append-only event log: one row per (first sighting |
                       price change | removal) of a model. This IS the time
                       series: to reconstruct the price of a model at time T,
                       take the last event at or before T.

Run:  python3 fetch_prices.py
Exit code 0 on success (even if no changes), non-zero on fetch/parse failure.
"""

from __future__ import annotations

import csv
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SOURCE_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
LATEST_PATH = DATA_DIR / "latest.json"
CSV_PATH = DATA_DIR / "prices.csv"

# litellm_provider -> (display name, region)
# Only direct first-party APIs (list prices), no resellers/clouds.
PROVIDERS: dict[str, tuple[str, str]] = {
    # --- USA ---
    "openai":    ("OpenAI",            "USA"),
    "anthropic": ("Anthropic",         "USA"),
    "gemini":    ("Google (Gemini)",   "USA"),
    "xai":       ("xAI",               "USA"),
    "meta_llama": ("Meta (Llama API)", "USA"),
    # --- EU ---
    "mistral":   ("Mistral AI",        "EU"),
    # --- China ---
    "deepseek":  ("DeepSeek",          "China"),
    "dashscope": ("Alibaba (Qwen)",    "China"),
    "moonshot":  ("Moonshot (Kimi)",   "China"),
    "zai":       ("Zhipu (GLM / Z.ai)","China"),
    "minimax":   ("MiniMax",           "China"),
}

# Model modes that count as text-in/text-out token pricing.
INCLUDE_MODES = {"chat", "responses", "completion"}

CSV_FIELDS = [
    "timestamp_utc",
    "event",            # new | change | removed
    "region",
    "provider",
    "model",
    "input_usd_per_1m",
    "output_usd_per_1m",
    "prev_input_usd_per_1m",
    "prev_output_usd_per_1m",
]


def fetch_source() -> dict:
    req = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "ai-price-tracker/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def normalize(raw: dict) -> dict[str, dict]:
    """Return {model_key: record} for tracked providers, prices in USD / 1M tokens."""
    out: dict[str, dict] = {}
    for model, spec in raw.items():
        if not isinstance(spec, dict):
            continue
        prov = spec.get("litellm_provider")
        if prov not in PROVIDERS:
            continue
        if spec.get("mode") not in INCLUDE_MODES:
            continue
        inp = spec.get("input_cost_per_token")
        outp = spec.get("output_cost_per_token")
        if inp is None or outp is None:
            continue
        # Strip provider prefix (e.g. "moonshot/kimi-k2" -> "kimi-k2")
        name = model.split("/", 1)[1] if "/" in model else model
        display, region = PROVIDERS[prov]
        key = f"{prov}::{name}"
        out[key] = {
            "provider": display,
            "region": region,
            "model": name,
            "input_usd_per_1m": round(inp * 1_000_000, 6),
            "output_usd_per_1m": round(outp * 1_000_000, 6),
        }
    return out


def load_latest() -> dict[str, dict]:
    if LATEST_PATH.exists():
        return json.loads(LATEST_PATH.read_text())["models"]
    return {}


def append_events(rows: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    new_file = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            w.writeheader()
        w.writerows(rows)


def main() -> int:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        current = normalize(fetch_source())
    except Exception as e:  # noqa: BLE001
        print(f"FETCH FAILED: {e}", file=sys.stderr)
        return 1

    if not current:
        print("FETCH returned no tracked models - refusing to record wipe-out.", file=sys.stderr)
        return 1

    previous = load_latest()
    events: list[dict] = []

    for key, rec in sorted(current.items()):
        prev = previous.get(key)
        base = {
            "timestamp_utc": now,
            "region": rec["region"],
            "provider": rec["provider"],
            "model": rec["model"],
            "input_usd_per_1m": rec["input_usd_per_1m"],
            "output_usd_per_1m": rec["output_usd_per_1m"],
        }
        if prev is None:
            events.append({**base, "event": "new",
                           "prev_input_usd_per_1m": "", "prev_output_usd_per_1m": ""})
        elif (prev["input_usd_per_1m"] != rec["input_usd_per_1m"]
              or prev["output_usd_per_1m"] != rec["output_usd_per_1m"]):
            events.append({**base, "event": "change",
                           "prev_input_usd_per_1m": prev["input_usd_per_1m"],
                           "prev_output_usd_per_1m": prev["output_usd_per_1m"]})

    for key, prev in sorted(previous.items()):
        if key not in current:
            events.append({
                "timestamp_utc": now, "event": "removed",
                "region": prev["region"], "provider": prev["provider"],
                "model": prev["model"],
                "input_usd_per_1m": "", "output_usd_per_1m": "",
                "prev_input_usd_per_1m": prev["input_usd_per_1m"],
                "prev_output_usd_per_1m": prev["output_usd_per_1m"],
            })

    if events:
        append_events(events)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_PATH.write_text(json.dumps(
        {"checked_at_utc": now, "source": SOURCE_URL,
         "model_count": len(current), "models": current},
        indent=2, sort_keys=True) + "\n")

    n_new = sum(e["event"] == "new" for e in events)
    n_chg = sum(e["event"] == "change" for e in events)
    n_rm = sum(e["event"] == "removed" for e in events)
    print(f"{now}  tracked={len(current)}  new={n_new}  changed={n_chg}  removed={n_rm}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
