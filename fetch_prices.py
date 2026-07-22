#!/usr/bin/env python3
"""
AI frontier-lab list-price tracker, v2: two-tier sourcing.

Tier 1 (primary):  crawl each lab's official pricing page (see vendors.py).
Tier 2 (fallback): LiteLLM community price DB, used per vendor whenever the
                   direct scrape fails fetch, extraction, or validation.

Every model in the output carries a `source`: "vendor" or "community".
Vendor prices always win over community prices for the same model.

Data files:
  data/latest.json - current snapshot (with per-model source)
  data/prices.csv  - append-only event log (new/change/removed), incl. source

Env:
  ANTHROPIC_API_KEY - optional; enables LLM extraction for vendors without a
                      deterministic parser. Without it, only vendors with a
                      parser are crawled directly and the rest use fallback.

Run:  python3 fetch_prices.py
"""

from __future__ import annotations

import csv
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import vendors

COMMUNITY_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
LATEST_PATH = DATA_DIR / "latest.json"
CSV_PATH = DATA_DIR / "prices.csv"

INCLUDE_MODES = {"chat", "responses", "completion"}

CSV_FIELDS = [
    "timestamp_utc", "event", "region", "provider", "model",
    "input_usd_per_1m", "output_usd_per_1m",
    "prev_input_usd_per_1m", "prev_output_usd_per_1m", "source",
]


# ------------------------------------------------------- community tier --

def fetch_community() -> dict[str, dict]:
    req = urllib.request.Request(COMMUNITY_URL, headers={"User-Agent": "ai-price-tracker/2.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = json.load(resp)
    out: dict[str, dict] = {}
    for model, spec in raw.items():
        if not isinstance(spec, dict):
            continue
        prov = spec.get("litellm_provider")
        if prov not in vendors.VENDORS or spec.get("mode") not in INCLUDE_MODES:
            continue
        inp, outp = spec.get("input_cost_per_token"), spec.get("output_cost_per_token")
        if inp is None or outp is None:
            continue
        name = (model.split("/", 1)[1] if "/" in model else model).lower()
        display, region = vendors.VENDORS[prov]["name"], vendors.VENDORS[prov]["region"]
        out[f"{prov}::{name}"] = {
            "provider": display, "region": region, "model": name,
            "input_usd_per_1m": round(inp * 1_000_000, 6),
            "output_usd_per_1m": round(outp * 1_000_000, 6),
            "source": "community",
        }
    return out


# ----------------------------------------------------------- vendor tier --

def overlay_vendor(current: dict[str, dict], prov_key: str, rows: list[dict]) -> int:
    """Overlay scraped vendor rows onto the community baseline.
    Match by exact model id, then by unique prefix; unmatched ids are added."""
    cfg = vendors.VENDORS[prov_key]
    prov_models = {k: v for k, v in current.items() if k.startswith(prov_key + "::")}
    matched = 0
    for r in rows:
        mid = r["model"]
        key = f"{prov_key}::{mid}"
        target = key if key in current else None
        if target is None:  # unique prefix match (dated vs undated ids)
            cands = [k for k in prov_models
                     if k.split("::", 1)[1].startswith(mid) or mid.startswith(k.split("::", 1)[1])]
            if len(cands) == 1:
                target = cands[0]
        rec = {"provider": cfg["name"], "region": cfg["region"],
               "model": current[target]["model"] if target else mid,
               "input_usd_per_1m": r["input_usd_per_1m"],
               "output_usd_per_1m": r["output_usd_per_1m"],
               "source": "vendor"}
        current[target or key] = rec
        matched += 1
    return matched


# ------------------------------------------------------------ time series --

def load_latest() -> dict[str, dict]:
    if LATEST_PATH.exists():
        return json.loads(LATEST_PATH.read_text())["models"]
    return {}


def migrate_csv() -> None:
    """Add the `source` column to a v1 prices.csv, once."""
    if not CSV_PATH.exists():
        return
    with CSV_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if rows and "source" in rows[0]:
        return
    with CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            r.setdefault("source", "community")
            w.writerow(r)
    if rows:
        print(f"  migrated prices.csv to v2 schema ({len(rows)} rows)")


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
        current = fetch_community()
    except Exception as e:  # noqa: BLE001
        print(f"COMMUNITY FETCH FAILED: {e}", file=sys.stderr)
        return 1
    if not current:
        print("Community source returned no models - aborting.", file=sys.stderr)
        return 1

    tier: dict[str, str] = {}
    for prov_key in vendors.VENDORS:
        print(f"  crawling {prov_key} …")
        rows = vendors.scrape_vendor(prov_key)
        if rows:
            n = overlay_vendor(current, prov_key, rows)
            tier[prov_key] = f"vendor ({n} models)"
        else:
            tier[prov_key] = "fallback: community"
    for k, v in tier.items():
        print(f"    {k:<10} {v}")

    previous = load_latest()
    migrate_csv()
    events: list[dict] = []

    for key, rec in sorted(current.items()):
        prev = previous.get(key)
        base = {"timestamp_utc": now, "region": rec["region"], "provider": rec["provider"],
                "model": rec["model"], "input_usd_per_1m": rec["input_usd_per_1m"],
                "output_usd_per_1m": rec["output_usd_per_1m"], "source": rec["source"]}
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
                "region": prev["region"], "provider": prev["provider"], "model": prev["model"],
                "input_usd_per_1m": "", "output_usd_per_1m": "",
                "prev_input_usd_per_1m": prev["input_usd_per_1m"],
                "prev_output_usd_per_1m": prev["output_usd_per_1m"],
                "source": prev.get("source", "community")})

    if events:
        append_events(events)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_PATH.write_text(json.dumps(
        {"checked_at_utc": now, "sources": tier, "model_count": len(current),
         "models": current}, indent=2, sort_keys=True) + "\n")

    n_new = sum(e["event"] == "new" for e in events)
    n_chg = sum(e["event"] == "change" for e in events)
    n_rm = sum(e["event"] == "removed" for e in events)
    n_vendor = sum(m["source"] == "vendor" for m in current.values())
    print(f"{now}  tracked={len(current)} (vendor-priced={n_vendor})  "
          f"new={n_new}  changed={n_chg}  removed={n_rm}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
