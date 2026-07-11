# Build Prompt: Upgrade `fbs-ai-trader` into "AERO TRADE"

Paste this entire document to your coding agent (e.g. Claude Code) with the existing
`fbs-ai-trader` project open in the workspace. Do this task in the phases listed below,
in order, and do not skip Phase 0. Ask the user before starting Phase 8 (real-money trading)
and before deleting/overwriting `config/credentials.py`.

---

## 0. CONTEXT — READ BEFORE CHANGING ANYTHING

This is an existing Python trading bot (currently named `fbs-ai-trader`) that trades one
broker (FBS) via a MetaTrader 5 (MT5) bridge. It combines a rule-based "sniper" strategy
(`core/strategies/sniper_logic.py`) with scikit-learn models (`ai_engine/models/predictor.py`)
through a signal-fusion layer (`decision_maker/signal_fusion.py`). It currently only runs in
a terminal against one demo MT5 account.

**Rename the product to "AERO TRADE" everywhere**: the Python package/directory name, the
logger name (`TradingLogger("HybridBot")` → `TradingLogger("AeroTrade")`), all `BROKER_CONFIG`
default names, README/docs, and any UI you build. Do not change trading logic just because
you're renaming things — treat the rename as its own small, careful pass.

### Known bugs to fix immediately (Phase 1)

1. **`main.py`, method `execute_trade()`**: the symbol is hardcoded:
   ```python
   # Current (WRONG):
   symbol = 'EURUSD'
   order_type = 'BUY' if decision.action == 'BUY' else 'SELL'
   ```
   This ignores whichever symbol actually generated the signal, so trades can be placed on
   the wrong instrument. Trace back where `execute_trade()` is called from and pass the real
   symbol through, e.g.:
   ```python
   async def execute_trade(self, decision, symbol: str, market_data_df, market_analysis=None):
       ...
       order_type = 'BUY' if decision.action == 'BUY' else 'SELL'
       # use the passed-in `symbol`, never a literal
   ```
   Update every call site accordingly and add a unit test that fails if `symbol` is ever
   hardcoded again (e.g. assert the executed order's symbol matches the signal's symbol for
   at least 3 different symbols in a test run).

2. **`config/credentials.py`**: `FBS_REAL` references an undefined variable `login`:
   ```python
   FBS_REAL = {
       'login': login,   # NameError waiting to happen
       ...
   }
   ```
   This file also stores a plaintext password. Delete this file's approach entirely — see
   Phase 2 (credential storage) below. Do not simply fix the variable name; the whole
   plaintext-credentials-in-source-code pattern must go away.

3. **`main.py`, method `monitor_positions()`**: when a position disappears from the broker's
   open orders, the code tries to look up its outcome in order history, but if it can't find
   it, the comment says "assume neutral" — in practice this silently loses track of losses.
   Replace this entire mechanism with the trade ledger described in Phase 6; every order must
   be resolved to a definite win/loss/breakeven outcome and written to persistent storage
   before it's removed from `active_positions`. Never allow a closed trade to vanish without
   a recorded P&L.

4. **`data/market_analysis.py`**: `get_forex_news()`, `get_market_sentiment()`, and
   `get_economic_calendar()` all return hardcoded placeholder data (e.g. sentiment is always
   `{'bullish': 0.5, 'bearish': 0.5}` and news is always the same fake headline). This entire
   file needs to be replaced with real integrations — see Phase 4.

5. **`train_models.py` / `data/market_data.py`**: models are currently trained only on
   `_generate_mock_data()` — a synthetic random walk — never on real market history. This is
   why "accuracy" numbers from training mean nothing. Fix in Phase 3.

---

## 1. Rebranding checklist

- [ ] Rename the root project folder/package to `aero-trade` (or `aero_trade` for the Python
      import path — Python identifiers can't contain hyphens).
- [ ] Update every `logger.info("... Bot ...")`/docstring/README reference from FBS-specific
      or "Hybrid Trading Bot" wording to "AERO TRADE".
- [ ] `BROKER_CONFIG["name"]` should no longer default to `"FBS"` — it should be set per the
      broker the user actually selects (see Phase 2).
- [ ] Add an `APP_NAME = "AERO TRADE"` constant in `config/settings.py` and reference it from
      logs, the API responses, and any UI title bar.
- [ ] Do not touch the underlying strategy/math logic during this pass — pure rename only.

---

## 2. Multi-broker support (any broker offering MT5 login)

The user wants to support **any broker that offers an MT5 login** (Exness, IC Markets, XM,
Pepperstone, HFM, and FBS itself), selected from a list in the UI, with the user entering
their own account number, password, and server name, which then get verified live against
that broker before being saved.

### 2.1 Broker abstraction

Right now `FBSClient` and `MT5Bridge` are one-and-the-same thing pretending to be
FBS-specific. Refactor so there's a single generic MT5-based client, and "broker" becomes
just a named server preset plus a display name/logo, not a different code path.

```python
# core/broker/broker_registry.py
from dataclasses import dataclass
from typing import List

@dataclass
class BrokerPreset:
    id: str            # e.g. "exness"
    display_name: str  # e.g. "Exness"
    demo_server: str   # e.g. "Exness-MT5Trial8"
    real_server: str   # e.g. "Exness-MT5Real8"
    website: str = ""
    logo_path: str = ""

# Seed with the most common MT5 brokers; let the user add a custom one too
# (broker MT5 server names change over time -- the "Custom" option lets it stay usable
# even if a preset goes stale).
SUPPORTED_BROKERS: List[BrokerPreset] = [
    BrokerPreset("fbs", "FBS", "FBS-Demo", "FBS-Real", "https://fbs.com"),
    BrokerPreset("exness", "Exness", "Exness-MT5Trial8", "Exness-MT5Real8", "https://exness.com"),
    BrokerPreset("icmarkets", "IC Markets", "ICMarketsSC-Demo", "ICMarketsSC-Live01", "https://icmarkets.com"),
    BrokerPreset("xm", "XM", "XMGlobal-Demo 3", "XMGlobal-Real 3", "https://xm.com"),
    BrokerPreset("pepperstone", "Pepperstone", "Pepperstone-Demo", "Pepperstone-Live01", "https://pepperstone.com"),
    BrokerPreset("hfm", "HFM (HotForex)", "HFMarketsGlobal-Demo", "HFMarketsGlobal-Live", "https://hfm.com"),
    BrokerPreset("custom", "Other (enter server manually)", "", "", ""),
]

def get_broker(broker_id: str) -> BrokerPreset:
    match = next((b for b in SUPPORTED_BROKERS if b.id == broker_id), None)
    if not match:
        raise ValueError(f"Unknown broker: {broker_id}")
    return match
```

Rename `FBSClient` -> `MT5BrokerClient` (or similar) and have it accept a `BrokerPreset` plus
user-entered `login`/`password`/`is_demo`, instead of importing a hardcoded FBS server name.
Keep `MT5Bridge` largely as-is (it's already broker-agnostic -- it just calls the
`MetaTrader5` Python package, which talks to whatever MT5 terminal/server you log into). The
only broker-specific piece was ever the server string and branding -- confirm this while
refactoring and flag to the user if you find anything else FBS-specific baked in.

Important limitation to flag to the user in your final report: **the `MetaTrader5` Python
package only works on Windows** (it talks to a locally-installed MT5 terminal via DLL calls).
If the backend needs to run on Linux/macOS (e.g. a cloud server), you'll need either (a) a
Windows VM/host for the trading engine, or (b) a broker that exposes its own REST/FIX API
instead of MT5. Note this limitation clearly in the README and ask the user where they plan
to host the bot before assuming a deployment target.

### 2.2 Credential entry, verification, and secure storage flow

Build this flow (used by both the "add a broker account" API endpoint and, later, the UI):

1. User picks a broker from `SUPPORTED_BROKERS` (or enters a custom server name).
2. User enters `login` (account number), `password`, and whether it's a Demo or Real account.
3. Backend attempts `MT5Bridge().connect(login, password, server, demo=...)` **synchronously,
   with a short timeout (e.g. 15s)**, purely to verify the credentials work, then immediately
   calls `get_account_info()` to confirm a real balance/currency/leverage comes back.
4. If verification fails, return a clear error to the UI ("Login failed -- check your account
   number, password, and server name") and **do not save anything**.
5. If it succeeds, encrypt and store the credentials (see below), and return the account
   summary (balance, equity, currency, leverage, server, account number -- never the password)
   for the UI to display as confirmation.

**Never store credentials in a plaintext `.py` or `.json` file again.** Use one of:
- OS-native secure storage via the `keyring` package (works on Windows Credential Manager,
  macOS Keychain, and Linux Secret Service) for a desktop app, or
- An encrypted-at-rest database column using `cryptography.fernet.Fernet` with the encryption
  key itself pulled from an environment variable or OS keychain -- appropriate if the backend
  is a server the desktop/mobile apps talk to over the network.

Example using `keyring` (adjust if backend is server-hosted instead of local-desktop):

```python
import keyring
import json

SERVICE_NAME = "AeroTrade"

def save_broker_credentials(account_id: str, login: int, password: str, server: str, demo: bool):
    payload = json.dumps({"login": login, "password": password, "server": server, "demo": demo})
    keyring.set_password(SERVICE_NAME, account_id, payload)

def load_broker_credentials(account_id: str) -> dict:
    raw = keyring.get_password(SERVICE_NAME, account_id)
    if raw is None:
        raise KeyError(f"No stored credentials for {account_id}")
    return json.loads(raw)
```

Whichever mechanism you pick, add `config/credentials.py` (and any `.env` holding secrets) to
`.gitignore` if they aren't already, and grep the whole repo for any other hardcoded
login/password strings before finishing this phase -- the current demo credentials
(`FBS_DEMO`) are already exposed in the codebase you were given and must be removed/rotated.

### 2.3 Multiple accounts

Support the user adding more than one broker account (e.g. demo on Exness *and* real on
FBS) and switching between them, since the whole point is not being locked to one broker.
Store each as its own `account_id` (e.g. a UUID) with its own encrypted credentials and its
own trading config (risk settings, enabled pairs) so accounts don't share state accidentally.

---

## 3. Real historical data (replacing the mock data pipeline)

The user's data file for extra training data was empty, so **pull historical data directly
from the connected broker via MT5** instead -- `MT5Bridge.get_historical_data()` already
exists and works, it's just never been used for anything but small live lookbacks (200 bars).

### 3.1 Bulk historical download

```python
# data/history_downloader.py
import pandas as pd
from datetime import datetime, timedelta
from core.broker.mt5_bridge import MT5Bridge
from config.settings import DATA_DIR

def download_full_history(bridge: MT5Bridge, symbol: str, timeframe: str,
                           years: int = 5) -> pd.DataFrame:
    """
    MT5's copy_rates_from_pos is capped per call, so pull in chunks going backwards
    until we hit `years` of history or the broker runs out of data.
    """
    all_chunks = []
    chunk_size = 5000
    cutoff = datetime.now() - timedelta(days=365 * years)
    before_dt = datetime.now()

    while True:
        # NOTE: get_historical_data() as currently written only supports "most recent N
        # bars" via copy_rates_from_pos(symbol, tf, 0, count). To page backwards in time
        # add a new method to MT5Bridge, e.g.:
        #   def get_historical_data_before(self, symbol, timeframe, before_dt, count):
        #       ... wraps mt5.copy_rates_from(symbol, mt5_timeframe, before_dt, count)
        df_chunk = bridge.get_historical_data_before(symbol, timeframe, before_dt, chunk_size)
        if df_chunk is None or len(df_chunk) == 0:
            break
        all_chunks.append(df_chunk)
        oldest_time = df_chunk.index.min()
        if oldest_time <= cutoff:
            break
        before_dt = oldest_time

    if not all_chunks:
        return pd.DataFrame()

    full_df = pd.concat(all_chunks).sort_index()
    full_df = full_df[~full_df.index.duplicated(keep="first")]
    full_df = full_df[full_df.index >= cutoff]

    out_path = DATA_DIR / f"{symbol}_{timeframe}_history.parquet"
    full_df.to_parquet(out_path, compression="gzip")
    return full_df
```

Add a script `download_all_history.py` that loops over every enabled trading pair
(see Phase 7 for the new pairs) and every timeframe, calling `download_full_history()`,
so `train_models.py` has years of real bars to work with instead of a random walk. Run this
once after connecting a broker, then incrementally top it up (today's new candles only) on a
schedule -- don't re-download years of history every time.

### 3.2 Update `train_models.py`

Point `generate_training_data()` at the downloaded parquet files (via `market_data`'s existing
`_load_historical_data()`, which already looks for exactly this file pattern) instead of
`_generate_mock_data()`. Keep the mock generator only as an explicit fallback for first-run
demos where no broker is connected yet, and log a loud warning whenever it's used, so nobody
mistakes a mock-trained model for a real one.

---

## 4. Real news, sentiment, and economic-calendar signals

Replace every stub in `data/market_analysis.py` with a real integration. Pick **one** approach
and implement it fully rather than half-wiring several:

- **Option A -- dedicated financial news/calendar API** (recommended first choice): a service
  such as Finnhub, Alpha Vantage News, or Trading Economics' calendar API. These return
  structured `{headline, sentiment_score, timestamp, related_symbols}` data you can use
  directly, and most have a free tier suitable for testing.
- **Option B -- web search + LLM summarization**: search recent headlines for the traded
  symbol/currency (e.g. "EUR interest rate decision this week"), then use an LLM call to
  extract a bullish/bearish/neutral score and a short justification. More flexible, but adds
  latency and cost per symbol per cycle -- cache aggressively (e.g. refresh every 15-30 min,
  not every trading loop iteration).

Whichever you choose, the important architectural change is this: **the result must flow into
`SignalFusion` as a real `SignalSource.SENTIMENT` signal**, not stay a separate, unused method.
`decision_maker/signal_fusion.py` already has `SignalSource.SENTIMENT` defined with a 5% weight
and never populated -- wire it up:

```python
# somewhere in the trading loop, once per symbol per cycle (respecting your cache interval)
market_bias = await market_analysis.analyze_market_conditions(symbol)  # now real, not mocked
sentiment_signal = Signal(
    source=SignalSource.SENTIMENT,
    direction=1 if market_bias["bias"] == "bullish" else (-1 if market_bias["bias"] == "bearish" else 0),
    confidence=market_bias["sentiment"]["confidence"],
    strength=market_bias["sentiment"]["bullish"] - market_bias["sentiment"]["bearish"],
    timestamp=datetime.now().timestamp(),
    metadata={"news_count": market_bias["news_count"], "symbol": symbol}
)
await signal_fusion.add_signal(sentiment_signal)
```

Also raise the sentiment source weight from FBS's original 5% now that it's a real signal
(the user explicitly wants the AI's accuracy improved by comparing news to price data) --
0.15-0.20 is a reasonable starting point, taken proportionally from the AI/technical weights,
but make it a config value (`SIGNAL_FUSION_CONFIG["weights"]["sentiment"]`) so it can be
tuned without a code change, and keep `validate_settings()`'s check that weights sum to 1.0.

### 4.1 Economic calendar -> risk gating

High-impact events (e.g. interest rate decisions, NFP) are a common reason automated bots blow
up -- a trade can be perfectly signaled and still get stopped out by a scheduled news spike.
Use the real economic calendar data to **pause new entries** on a symbol for a configurable
window (e.g. 15 minutes before/after) around high-impact events affecting its currencies, not
just to feed sentiment. Add this as an explicit check in `risk_manager.validate_trade()`.

---

## 5. Dynamic position sizing driven by account size + combined confidence

The current `risk_manager.calculate_position_size()` does per-trade % risk with an aggressive
compounding multiplier, but doesn't yet factor in the *combined* AI+technical+sentiment
confidence from `SignalFusion`, nor does it distinguish "small demo account" caution from
"large real account" caution. Extend it so lot size responds to both account size and the
fused confidence/news alignment:

```python
def calculate_dynamic_position_size(self, account_balance: float, stop_loss_pips: float,
                                     fused_confidence: float, sentiment_agrees: bool,
                                     symbol: str) -> float:
    # 1. Base risk scales down as account size grows past certain tiers, since
    #    a flat 10% risk on a $10 demo account behaves very differently from
    #    10% on a $10,000 real account.
    if account_balance < 100:
        base_risk_pct = 0.02   # very small accounts: cap risk tightly, capital is scarce
    elif account_balance < 1000:
        base_risk_pct = 0.03
    elif account_balance < 10_000:
        base_risk_pct = 0.02
    else:
        base_risk_pct = 0.01   # larger accounts: smaller % risk, absolute risk still grows

    # 2. Reward higher combined confidence, but modestly -- this is not the place
    #    for the existing 3x "aggressive compounding" multiplier on a real account.
    confidence_multiplier = 0.75 + (fused_confidence * 0.5)  # ranges ~0.75x - 1.25x

    # 3. If technical/AI signal and news sentiment disagree, cut size -- don't take
    #    a trade the news is actively fighting against.
    if not sentiment_agrees:
        confidence_multiplier *= 0.5

    risk_amount = account_balance * base_risk_pct * confidence_multiplier
    pip_value = 10  # adjust per-symbol in section 7 (gold/BTC differ from FX pip value)
    position_size = risk_amount / max(stop_loss_pips, 1) / pip_value
    return round(max(position_size, 0.01), 2)
```

Keep the *existing* aggressive-compounding function available behind an explicit
`aggressive_mode` config flag that **defaults to off** and is only usable on demo accounts --
gate it in code, not just in documentation (e.g. raise if `aggressive_mode and not is_demo`).

---

## 6. Machine learning that actually learns from its own trade outcomes

This is the core of the user's request and the part `ai_engine/models/predictor.py` doesn't
really do yet -- `update_with_new_data()` exists but only logs a message. Build a real
feedback loop with three parts: **(a)** a persistent trade ledger, **(b)** labeling closed
trades as training examples, **(c)** periodic retraining/online updates from that ledger.

### 6.1 Trade ledger (also fixes the "doesn't show wins and losses" complaint)

```python
# core/ledger/trade_ledger.py
import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path("data/aero_trade_ledger.db")

def init_ledger():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            order_id TEXT PRIMARY KEY,
            account_id TEXT,
            symbol TEXT,
            direction TEXT,
            entry_price REAL,
            exit_price REAL,
            volume REAL,
            open_time TEXT,
            close_time TEXT,
            pnl_usd REAL,
            outcome TEXT,             -- 'win' | 'loss' | 'breakeven'
            confidence REAL,          -- fused confidence at entry
            features_json TEXT,       -- snapshot of the feature vector used at entry
            sentiment_agreed INTEGER, -- 1/0, did news agree with the trade direction
            model_used TEXT
        )
    """)
    conn.commit()
    conn.close()

def record_open_trade(order_id, account_id, symbol, direction, entry_price, volume,
                       confidence, features_json, sentiment_agreed, model_used):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO trades (order_id, account_id, symbol, direction, entry_price, volume,
           open_time, confidence, features_json, sentiment_agreed, model_used)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (order_id, account_id, symbol, direction, entry_price, volume,
         datetime.now().isoformat(), confidence, features_json, int(sentiment_agreed), model_used)
    )
    conn.commit()
    conn.close()

def record_close_trade(order_id: str, exit_price: float, pnl_usd: float):
    outcome = "win" if pnl_usd > 0 else ("loss" if pnl_usd < 0 else "breakeven")
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """UPDATE trades SET exit_price = ?, close_time = ?, pnl_usd = ?, outcome = ?
           WHERE order_id = ?""",
        (exit_price, datetime.now().isoformat(), pnl_usd, outcome, order_id)
    )
    conn.commit()
    conn.close()
```

Wire this in at exactly two points: right after a successful `place_order()` call (insert),
and inside `monitor_positions()`/`close_position()` once a real `pnl_usd` is known from the
broker (update) -- **every** closed trade must resolve to `win`/`loss`/`breakeven`; never leave
a trade's outcome undetermined the way the current code does when it can't find it in history.
This same table is what your dashboard/API queries to show the user their wins and losses --
see Phase 9's `/account/trades` endpoint.

### 6.2 Learning from mistakes: retraining loop

Two complementary mechanisms -- do both:

1. **Scheduled batch retraining** (simplest, most robust): a script/cron job that, once
   enough new closed trades have accumulated (e.g. every 50-100 new trades, or nightly),
   pulls all rows from `trades` plus fresh broker history, rebuilds the training set the same
   way `train_models.py` does for historical data, but **weights recent real trade outcomes
   more heavily than historical price data alone**, then retrains and versions the model
   (don't overwrite the old one -- save as `rf_5min_v{timestamp}.joblib` and only "promote" it
   to the active model if it beats the current one on a held-out validation slice).

2. **Incremental/online update after every closed trade** for a fast-adapting layer: implement
   `update_with_new_data()` for real this time --

```python
async def update_with_new_data(self, df: pd.DataFrame, actual_direction: int, timeframe: str = '5min'):
    """
    Called after a trade closes. `actual_direction` is derived from the *real* outcome
    (1 if price moved in the predicted direction profitably, 0 otherwise) -- this is what
    makes it learn from mistakes rather than just from raw price history.
    """
    features = await self.prepare_features(df)
    if features.size == 0:
        return

    model_key = f"rf_{timeframe}"
    if model_key not in self.models:
        return

    model = self.models[model_key]
    # RandomForestClassifier doesn't support true partial_fit, so maintain a rolling
    # buffer of recent (features, outcome) pairs and periodically re-fit on buffer + a
    # sample of historical data, rather than claiming true online learning where sklearn
    # can't actually provide it. Simpler alternative: switch this model to
    # sklearn.linear_model.SGDClassifier (which *does* support partial_fit) specifically
    # for the "fast online adjustment" role, keeping RandomForest/GBM for the periodic
    # batch retrain in 6.1. Recommended split:
    #   - SGDClassifier: updated via partial_fit() after every single closed trade
    #   - RandomForest/GBM: only touched by the scheduled batch retrain job
    if hasattr(model, "partial_fit"):
        X = self.scalers[timeframe].transform(features)
        model.partial_fit(X, [actual_direction])
        logger.info(f"Online-updated {model_key} with real trade outcome={actual_direction}")
    else:
        self._pending_retrain_buffer.append((features, actual_direction))
```

Add a `mistake_log` alongside the ledger: whenever a closed trade is a loss **and** the model's
pre-trade confidence was above the trading threshold (i.e. the model was confident and still
wrong), log the full feature snapshot into a separate `mistakes` table/CSV. Use this set
specifically to evaluate whether retrained models are actually improving on the bot's worst
past decisions, not just on aggregate accuracy -- print a before/after "accuracy on known past
mistakes" number every time you promote a new model version.

---

## 7. New trading pairs: Gold and BTC

- **Gold (XAUUSD)** is already in `TRADING_CONFIG["pairs"]` and MT5 handles it as a normal CFD
  symbol, but check that pip-value math (`pip_value = 10` hardcoded all over
  `risk_manager.py`/`main.py`) is correct for gold specifically -- gold's pip value per lot is
  **not** the same as a standard FX pair. Add a per-symbol pip-value/point-value lookup
  instead of the current hardcoded `10`, and prefer pulling it live from the broker
  (`mt5.symbol_info(symbol).trade_tick_value`) over a hardcoded table, since it varies by
  broker and contract size.
- **BTC**: add `BTCUSD` (or whatever the broker names it -- `BTCUSD.`, `BTCUSDm`, etc. vary by
  broker; look it up via `MT5Bridge.get_all_symbols()` after connecting rather than assuming
  the exact string) to the tradeable pairs. Crypto CFDs move in whole-dollar increments, not
  pips -- don't reuse the `* 10000` / `* 0.0001` pip math anywhere for BTC. Instead, compute
  stop-loss/take-profit as a direct price distance (e.g. `sl_price = entry_price - sl_dollars`)
  and size positions off `mt5.symbol_info(symbol).trade_tick_value` /
  `trade_contract_size`, which MT5 reports correctly per-symbol. Also expect materially wider
  spreads and higher volatility on crypto CFDs than on major FX pairs -- make per-symbol
  stop-loss distance configurable rather than reusing the FX defaults (100 pips / 50 pips
  scalping) verbatim.
- Update `sniper_logic.py`'s ATR-based TP/SL calculations similarly -- anywhere pips are
  assumed (`* 10000`), branch on whether the symbol is FX vs. metal vs. crypto.
- Add both to the default `TRADING_CONFIG["pairs"]` list and to the "major pairs" filter in
  `main.py`'s `initialize()` (currently it only keeps symbols starting with
  `EUR/GBP/USD/JPY/AUD/NZD/CAD/CHF/XAU` -- add a `BTC` prefix so it isn't filtered out).

---

## 8. Real-account safety gate (demo already works; real money is new)

The user has only used demo so far and now wants real-account support. Treat this as a
distinct, higher-stakes feature, not a toggle:

- Add an explicit **"Live Trading" mode** that is off by default per account. Turning it on
  requires: (a) re-entering the account password to confirm intent, (b) a UI confirmation
  dialog stating in plain language that real money is at risk, and (c) the aggressive
  compounding / high-risk-per-trade defaults being force-reset to the conservative values from
  Phase 5 (`aggressive_mode` must be explicitly re-enabled per account and must raise if
  attempted on a non-demo account, per section 5).
- Enforce `RISK_CONFIG["max_daily_loss_pct"]` and `max_drawdown_pct` as a hard kill-switch on
  real accounts: if breached, immediately stop opening new trades and alert the user (push
  notification/UI banner) -- don't just log a warning like the current code does.
- Keep demo and real credentials/ledgers/models **fully separate per account_id** -- never let
  a model trained partly on demo-account trade outcomes silently influence real-account sizing
  without the user being told which data went into it.
- This is not financial advice, and implementing this doesn't verify the trading strategy's
  profitability -- flag clearly to the user in the README that automated leveraged trading,
  especially with aggressive compounding, carries real risk of loss, and that they should only
  enable Live Trading with money they can afford to lose.

---

## 9. Backend API layer (wraps the existing Python engine -- don't rewrite it)

Wrap the bot in a **FastAPI** service so a mobile/desktop UI can talk to it over HTTP/WebSocket
without rewriting any trading logic -- the engine keeps running as Python, the API just
exposes it.

Minimum endpoint set:

- `GET  /brokers` -> list of `SUPPORTED_BROKERS` for the picker UI.
- `POST /accounts` -> body: `{broker_id, login, password, is_demo}` -> verifies via section 2.2,
  stores credentials securely, returns `{account_id, balance, equity, currency, leverage}`
  (never echoes the password back).
- `GET  /accounts/{account_id}` -> current account snapshot (balance, equity, margin, open
  positions count).
- `GET  /accounts/{account_id}/positions` -> open positions with live floating P&L per position.
- `GET  /accounts/{account_id}/trades?limit=100` -> closed trade history from the ledger
  (Phase 6.1), including `outcome`, `pnl_usd`, `symbol`, `open_time`, `close_time` -- **this is
  what fixes "doesn't show both losses and profits"**: return every closed trade regardless of
  outcome, plus aggregate `{total_pnl, win_rate, wins, losses, profit_factor}` computed from
  the same table so the UI can show a summary card and a full history list.
- `POST /accounts/{account_id}/start` / `POST /accounts/{account_id}/stop` -> start/stop the
  trading loop for that account.
- `POST /accounts/{account_id}/live-mode` -> body: `{enable: bool, password}` -> the Phase 8
  confirmation gate.
- `GET  /accounts/{account_id}/settings` / `PUT .../settings` -> risk-per-trade, enabled pairs
  (including new Gold/BTC toggles), aggressive-mode flag (demo-only), max daily loss, etc.
- `WS   /accounts/{account_id}/stream` -> push live account/position updates and log lines to
  the UI instead of the UI polling constantly.

Run the actual `HybridTradingBot`/per-account trading loop as an `asyncio` background task per
connected account inside this FastAPI process (or a small task queue if you expect many
concurrent accounts), rather than the current single-process terminal script.

---

## 10. Mobile + desktop UI

The user has left the exact framework choice to you -- pick based on what best serves "one
broker-selection screen, one credential-entry-and-verify screen, and one account view", shipped
to both mobile and desktop with the least duplicated work. Two reasonable directions; decide
based on the team's familiarity and how "native" the apps need to feel, and state your choice
and reasoning at the start of implementation rather than silently picking one:

- **Flutter**: one codebase compiles to iOS, Android, Windows, macOS, and Linux natively -- the
  strongest fit if "mobile and desktop apps" (not just a responsive website) is a hard
  requirement, at the cost of Dart being a new language for whoever maintains this.
- **A responsive web dashboard** (e.g. React or Next.js) that's installable as a PWA on mobile
  and works as a normal desktop web app -- faster to build and iterate on, single JS/TS
  codebase, but it's technically a website wrapped to feel app-like rather than a native app.

Whichever you choose, build these screens against the Phase 9 API, not against the Python
engine directly:

1. **Broker selection** -- grid/list from `GET /brokers`, "Other" option for manual server entry.
2. **Credential entry + verification** -- login/password/demo-or-real fields -> calls
   `POST /accounts` -> shows a spinner while verifying -> shows the returned account summary on
   success or a clear error inline on failure (never silently fail).
3. **Account dashboard** -- balance/equity/margin header, open positions list with live floating
   P&L, a start/stop control, and a prominent, clearly-labeled Demo/Live badge.
4. **Trade history** -- full list from `GET /accounts/{id}/trades`, each row showing symbol,
   direction, size, entry/exit, and **P&L in green for wins / red for losses** (this is the
   direct fix for "doesn't display both losses and profits" -- make sure losses are shown just
   as prominently as wins, not hidden or rounded away).
5. **Settings** -- risk per trade, enabled pairs (with Gold/BTC included), the Live Trading
   toggle (routes through the Phase 8 confirmation), and aggressive-mode (demo-only, disabled
   and explained as such on real accounts).

---

## 11. Backtesting

Update `backtest.py` to run against the real historical parquet files from Phase 3 instead of
whatever it currently uses, and to route trades through the *same* dynamic position-sizing
and signal-fusion code paths as live trading (no separate, drifting copy of the logic) so
backtest results actually predict live behavior. Add a walk-forward mode: train on data up to
date X, test on X+1 to Y, roll forward -- a single train/test split on historical price data
overstates performance for anything meant to keep learning over time.

---

## 12. Suggested build order (phases map to sections above)

1. Fix the 3 bugs in section "Known bugs to fix immediately."
2. Rebrand to AERO TRADE (section 1).
3. Broker abstraction + secure credential storage + verification flow (section 2).
4. Real historical data download + retrain pipeline pointed at it (section 3).
5. Real news/sentiment/calendar integration wired into SignalFusion (section 4).
6. Dynamic, account-size-aware position sizing (section 5).
7. Trade ledger + online/batch retraining loop (section 6).
8. Gold/BTC symbol support with correct pip/tick math (section 7).
9. Real-account safety gate (section 8).
10. FastAPI backend wrapping everything above (section 9).
11. Mobile/desktop UI against that API (section 10).
12. Backtest overhaul (section 11).

Confirm with the user before starting step 9 (real-money support) that they understand the
risk, and before step 3 that their existing exposed demo password should be rotated/changed
on the broker's side, since it has been sitting in plaintext in this codebase.
