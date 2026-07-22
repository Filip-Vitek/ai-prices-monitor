"""
Vendor price crawlers: fetch list prices directly from each lab's official
pricing page. Extraction strategy per vendor:

  1. deterministic parser, where we know the page structure (fast, free), else
  2. LLM extraction via the Anthropic API (robust to page redesigns; needs
     ANTHROPIC_API_KEY in the environment), else
  3. the caller falls back to the community database for that vendor.

Every scrape passes validation before it is trusted. A vendor failing for any
reason degrades gracefully to fallback; it never breaks the run.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from html.parser import HTMLParser

UA = {"User-Agent": "Mozilla/5.0 (compatible; ai-price-tracker/2.0; +https://github.com)"}
LLM_MODEL = "claude-haiku-4-5"
MAX_PAGE_CHARS = 120_000

# provider_key (must match keys used for community source) -> config
VENDORS: dict[str, dict] = {
    "openai":    {"name": "OpenAI",             "region": "USA",
                  "urls": ["https://platform.openai.com/docs/pricing"]},
    "anthropic": {"name": "Anthropic",          "region": "USA",
                  "urls": ["https://docs.claude.com/en/docs/about-claude/pricing"]},
    "gemini":    {"name": "Google (Gemini)",    "region": "USA",
                  "urls": ["https://ai.google.dev/gemini-api/docs/pricing"]},
    "xai":       {"name": "xAI",                "region": "USA",
                  "urls": ["https://docs.x.ai/docs/models"]},
    "mistral":   {"name": "Mistral AI",         "region": "EU",
                  "urls": ["https://mistral.ai/pricing#api-pricing"]},
    "deepseek":  {"name": "DeepSeek",           "region": "China",
                  "urls": ["https://api-docs.deepseek.com/quick_start/pricing"],
                  "parser": "parse_deepseek"},
    "dashscope": {"name": "Alibaba (Qwen)",     "region": "China",
                  "urls": ["https://www.alibabacloud.com/help/en/model-studio/models"]},
    "moonshot":  {"name": "Moonshot (Kimi)",    "region": "China",
                  "urls": ["https://platform.moonshot.ai/docs/pricing/chat"]},
    "zai":       {"name": "Zhipu (GLM / Z.ai)", "region": "China",
                  "urls": ["https://docs.z.ai/guides/overview/pricing"]},
    "minimax":   {"name": "MiniMax",            "region": "China",
                  "urls": ["https://platform.minimax.io/document/Price"]},
}


# ---------------------------------------------------------------- fetching --

class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        elif tag in {"tr", "br", "p", "div", "li", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")
        elif tag in {"td", "th"}:
            self.parts.append(" | ")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip and data.strip():
            self.parts.append(data.strip() + " ")


def html_to_text(html: str) -> str:
    p = _TextExtractor()
    try:
        p.feed(html)
    except Exception:  # noqa: BLE001
        pass
    text = "".join(p.parts)
    return re.sub(r"\n{3,}", "\n\n", text)[:MAX_PAGE_CHARS]


def fetch_page_text(url: str) -> str | None:
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=45) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        return html_to_text(raw)
    except Exception as e:  # noqa: BLE001
        print(f"    fetch failed {url}: {e}", file=sys.stderr)
        return None


# --------------------------------------------------- deterministic parsers --

def parse_deepseek(text: str) -> list[dict]:
    """DeepSeek docs render a transposed table: a 'MODEL …' line listing ids
    (before 'BASE URL'), then 'CACHE MISS' and '1M OUTPUT TOKENS' lines
    listing one $ value per model, in the same order."""
    seg = re.search(r"MODEL\s+(.*?)BASE URL", text, re.S)
    if not seg:
        return []
    models = re.findall(r"deepseek-[a-z0-9.\-]+", seg.group(1).lower())
    seen, ids = set(), []
    for m in models:
        m = m.rstrip(".-")
        if m not in seen and not m.endswith(("chat", "reasoner")):  # legacy aliases
            seen.add(m); ids.append(m)
    def dollars(after: str) -> list[float]:
        m = re.search(re.escape(after) + r"(.{0,200})", text, re.I | re.S)
        return [float(x) for x in re.findall(r"\$([0-9.]+)", m.group(1))] if m else []
    inp = dollars("CACHE MISS")
    out = dollars("1M OUTPUT TOKENS")
    n = min(len(ids), len(inp), len(out))
    return [{"model": ids[i], "input_usd_per_1m": inp[i], "output_usd_per_1m": out[i]}
            for i in range(n)]


PARSERS = {"parse_deepseek": parse_deepseek}


# --------------------------------------------------------- LLM extraction --

EXTRACT_PROMPT = """From the pricing page text below, extract the official pay-as-you-go LIST PRICE for every TEXT generation model (chat/completion/reasoning), in USD per 1 million tokens, standard tier.

Rules:
- "model" must be the API model identifier (lowercase, hyphenated, as used in API calls), not the marketing name. Skip models where no API id is discernible.
- input_usd_per_1m = standard input price (use the cache-MISS / non-cached rate if cached rates are shown).
- output_usd_per_1m = standard output price.
- If pricing is tiered by context length, use the lowest/base context tier.
- Skip: embeddings, image/audio/video/speech models, fine-tuning rates, batch discounts, cached rates, per-request fees, deprecated-marked models.
- If prices on the page are not in USD, return [].
- Respond with ONLY a JSON array, no markdown fences, no commentary:
[{"model": "...", "input_usd_per_1m": 0.0, "output_usd_per_1m": 0.0}]

PAGE TEXT:
"""


def extract_with_llm(text: str, api_key: str) -> list[dict]:
    body = json.dumps({
        "model": LLM_MODEL,
        "max_tokens": 8000,
        "messages": [{"role": "user", "content": EXTRACT_PROMPT + text}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=body, method="POST",
        headers={"Content-Type": "application/json",
                 "x-api-key": api_key, "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.load(resp)
    reply = "".join(b.get("text", "") for b in data.get("content", []))
    reply = re.sub(r"^```(json)?|```$", "", reply.strip(), flags=re.M).strip()
    return json.loads(reply)


# -------------------------------------------------------------- validation --

def validate(rows: list[dict]) -> list[dict]:
    ok = []
    for r in rows:
        try:
            m = str(r["model"]).strip().lower()
            i, o = float(r["input_usd_per_1m"]), float(r["output_usd_per_1m"])
        except (KeyError, TypeError, ValueError):
            continue
        if not re.fullmatch(r"[a-z0-9][a-z0-9.\-_/:]{1,80}", m):
            continue
        if not (0 < i <= 1000 and 0 < o <= 2000):
            continue
        ok.append({"model": m, "input_usd_per_1m": round(i, 6),
                   "output_usd_per_1m": round(o, 6)})
    return ok


# ------------------------------------------------------------ orchestrator --

def scrape_vendor(key: str) -> list[dict] | None:
    """Return validated [{model,in,out}] for a vendor, or None to trigger fallback."""
    cfg = VENDORS[key]
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    for url in cfg["urls"]:
        text = fetch_page_text(url)
        if not text or len(text) < 300:
            continue
        rows: list[dict] = []
        if "parser" in cfg:
            try:
                rows = validate(PARSERS[cfg["parser"]](text))
            except Exception as e:  # noqa: BLE001
                print(f"    parser error {key}: {e}", file=sys.stderr)
        if not rows and api_key:
            try:
                rows = validate(extract_with_llm(text, api_key))
            except Exception as e:  # noqa: BLE001
                print(f"    llm extraction failed {key}: {e}", file=sys.stderr)
        if rows:
            return rows
    return None
