# AI Frontier-Lab Token Price Tracker

Tracks official **list prices** (USD per 1M tokens, standard pay-as-you-go tier) for
text models from major frontier labs, twice a day, as an append-only time series —
using the "git scraper" pattern: GitHub Actions runs the fetch on a cron schedule
and commits any changes back to this repo. No servers, no database, full history in git.

## Coverage

| Region | Labs |
|--------|------|
| USA    | OpenAI, Anthropic, Google (Gemini), xAI |
| EU     | Mistral AI |
| China  | DeepSeek, Alibaba (Qwen/DashScope), Moonshot (Kimi), Zhipu (GLM/Z.ai), MiniMax |

~330 chat/completion models. Only first-party APIs (no Bedrock/Azure/OpenRouter resale rates).

## Data sources (two tiers)

**Tier 1 — vendor crawlers (primary).** `vendors.py` fetches each lab's official
pricing page and extracts model / input / output prices:
- deterministic parser where the page structure is known (currently DeepSeek),
- otherwise LLM extraction via the Anthropic API (`claude-haiku-4-5`), which is
  robust to page redesigns. Costs roughly a cent per run.

**Tier 2 — community fallback.** [LiteLLM's price database](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json)
is used *per vendor* whenever the direct crawl fails fetch, extraction, or sanity
validation — so one broken page never breaks the run. Every model carries a
`source` field (`vendor` / `community`) in `latest.json` and `prices.csv`, and the
per-vendor outcome of each run is recorded in `latest.json` under `sources`.

**Enable the vendor tier:** add a repo secret (Settings → Secrets and variables →
Actions → New repository secret) named `ANTHROPIC_API_KEY` with a key from
console.anthropic.com. Without it, only deterministic-parser vendors are crawled
directly and everything else uses the fallback.

Vendor pricing URLs live at the top of `vendors.py` — if a lab moves its pricing
page, update the URL there; validation + fallback covers the gap meanwhile.

## Data layout

- `data/latest.json` — current snapshot (all tracked models, checked timestamp)
- `data/prices.csv` — the time series, as an **event log**:
  - `event=new` — model first seen (row carries its launch list price)
  - `event=change` — price changed (row carries new + previous price)
  - `event=removed` — model disappeared from the source
  
  The price of model M at time T = the last `new`/`change` row for M at or before T.
  This keeps the file small (a row only when something actually changes) while the
  git history of `latest.json` additionally gives you a full twice-daily audit trail.

## Setup (one time, ~2 minutes)

```bash
gh repo create ai-price-tracker --private --source . --push
```

or manually: create a repo, push these files. That's it — the workflow in
`.github/workflows/track-prices.yml` runs at **06:17 and 18:17 UTC** daily and
commits whenever prices change. You can also trigger it manually from the
Actions tab (`workflow_dispatch`).

> Note: GitHub disables scheduled workflows in repos with no activity for 60 days;
> the bot's own commits count as activity, but if prices are ever static for
> 60+ days you may need to re-enable the workflow once.

## Querying the series

```python
import pandas as pd
df = pd.read_csv("data/prices.csv", parse_dates=["timestamp_utc"])

# All price changes, most recent first
changes = df[df.event == "change"].sort_values("timestamp_utc", ascending=False)

# Reconstruct price of one model over time
m = df[(df.model == "deepseek-chat") & (df.event != "removed")]
print(m[["timestamp_utc", "input_usd_per_1m", "output_usd_per_1m"]])
```

## Run locally

```bash
python3 fetch_prices.py   # stdlib only, no dependencies
```

## Web app

`index.html` is a self-contained dashboard ("Token Tariff") that reads
`data/latest.json` and `data/prices.csv` from the same repo:

- region summary cards (USA / EU / China: model counts, median / cheapest / priciest output price)
- sortable, filterable price board with per-model sparklines
- click rows to plot up to 6 models on a step-line price-history chart (log/linear toggle)
- price-event feed (cuts, hikes, launches, delistings)

**Publish it:** repo Settings -> Pages -> "Deploy from a branch" -> `main` / root.
Every cron commit then updates the live site automatically. Nothing else to deploy.

**Preview locally:** `python3 -m http.server` in the repo root, open http://localhost:8000
(open the file via a server, not file://, so the fetch of the data files works).
