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
import re
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

def fetch_community() -> tuple[dict[str, dict], set[str]]:
    """Community prices, plus the keys of listings left off the board on purpose."""
    req = urllib.request.Request(COMMUNITY_URL, headers={"User-Agent": "ai-price-tracker/2.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = json.load(resp)
    out: dict[str, dict] = {}
    skipped: set[str] = set()
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
        outputs = spec.get("supported_output_modalities")
        # resale of another lab's model, fine-tuning rates, music (Lyria) / speech (TTS)
        if vendors.excluded(prov, name) or (outputs and "text" not in outputs):
            skipped.add(f"{prov}::{name}")
            continue
        display, region = vendors.VENDORS[prov]["name"], vendors.VENDORS[prov]["region"]
        out[f"{prov}::{name}"] = {
            "provider": display, "region": region, "model": name,
            "input_usd_per_1m": round(inp * 1_000_000, 6),
            "output_usd_per_1m": round(outp * 1_000_000, 6),
            "source": "community",
        }
    return out, skipped


# --------------------------------------------------------- release dates --
# models.dev is a public model catalogue with a release date per model. The
# board uses it to order the "latest models per lab" boxes. Without it the
# page can only guess from version numbers and from when this tracker first
# saw an id, which puts late-added aliases above newer models.

MODELS_DEV_URL = "https://models.dev/api.json"
MODELS_DEV_IDS = {  # our provider key -> models.dev provider ids
    "openai": ["openai"], "anthropic": ["anthropic"], "gemini": ["google"],
    "xai": ["xai"], "mistral": ["mistral"], "deepseek": ["deepseek"],
    "dashscope": ["alibaba"], "moonshot": ["moonshotai"], "zai": ["zai", "zhipuai"],
    "minimax": ["minimax"],
}
_STAMP = re.compile(r"-(\d{4}-\d{2}-\d{2}|\d{8}|\d{2}-\d{4}|\d{4})$")
_VARIANT = re.compile(r"-(latest|preview|beta|experimental|exp|reasoning|non-reasoning|gv2|chat)$")


def fetch_release_dates() -> dict[str, dict[str, str]]:
    """{provider key: {model id: 'YYYY-MM-DD'}}; empty when models.dev is down."""
    try:
        req = urllib.request.Request(MODELS_DEV_URL, headers={"User-Agent": "ai-price-tracker/2.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = json.load(resp)
    except Exception as e:  # noqa: BLE001
        print(f"  models.dev unavailable, keeping known release dates: {e}", file=sys.stderr)
        return {}
    out: dict[str, dict[str, str]] = {}
    for prov, ids in MODELS_DEV_IDS.items():
        dates = out.setdefault(prov, {})
        for pid in ids:
            for mid, m in (raw.get(pid) or {}).get("models", {}).items():
                d = str(m.get("release_date") or "")[:10]
                if re.fullmatch(r"\d{4}-\d{2}-\d{2}", d):
                    dates.setdefault(mid.lower(), d)
    return out


def release_date(dates: dict[str, str], model: str) -> str | None:
    """Exact id first, then the id without date stamp / variant suffixes, then
    the earliest catalogue id that extends it (grok-4.20 -> grok-4.20-0309-…),
    then the base model of a variant (minimax-m2.1-lightning -> minimax-m2.1).
    A shortened id must keep its version number, so deepseek-chat never
    matches every deepseek-… model."""
    if model in dates:
        return dates[model]
    bases = [model]
    while True:
        b = _VARIANT.sub("", _STAMP.sub("", bases[-1]))
        if b == bases[-1]:
            break
        bases.append(b)
    bases = [b for b in bases if re.search(r"\d", b)]
    for b in bases:
        if b in dates:
            return dates[b]
    for b in bases:
        kids = [d for k, d in dates.items() if k.startswith(b + "-")]
        if kids:
            return min(kids)
    b = bases[-1] if bases else ""
    while "-" in b:
        b = b.rsplit("-", 1)[0]
        if not re.search(r"\d", b):
            break
        if b in dates:
            return dates[b]
    return None


# ----------------------------------------------------------- vendor tier --

def overlay_vendor(current: dict[str, dict], prov_key: str, rows: list[dict]) -> int:
    """Overlay scraped vendor rows onto the community baseline.
    Match by exact model id, then by unique prefix; unmatched ids are added."""
    cfg = vendors.VENDORS[prov_key]
    prov_models = {k: v for k, v in current.items() if k.startswith(prov_key + "::")}
    matched = 0
    for r in rows:
        mid = r["model"]
        if vendors.excluded(prov_key, mid):   # e.g. DeepSeek listed on Alibaba's page
            continue
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


def prune_csv(skipped: set[str]) -> None:
    """Drop the history of listings that are now left off the board on purpose
    (resale, fine-tuning, non-text), so the event feed does not show them."""
    if not CSV_PATH.exists() or not skipped:
        return
    by_name = {v["name"]: k for k, v in vendors.VENDORS.items()}
    with CSV_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))
    keep = [r for r in rows
            if f"{by_name.get(r['provider'], r['provider'])}::{r['model']}" not in skipped]
    if len(keep) == len(rows):
        return
    with CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(keep)
    print(f"  pruned {len(rows) - len(keep)} history rows of excluded listings")


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
        current, skipped = fetch_community()
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
    # Excluded listings disappear quietly instead of as "removed" events.
    skipped |= {k for k in previous
                if k not in current and vendors.excluded(*k.split("::", 1))}
    for k in skipped:
        previous.pop(k, None)
    prune_csv(skipped)

    dates = fetch_release_dates()
    for key, rec in current.items():
        d = release_date(dates.get(key.split("::", 1)[0], {}), rec["model"])
        d = d or previous.get(key, {}).get("released")
        if d:
            rec["released"] = d
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
