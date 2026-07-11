# AERO TRADE

AI-powered multi-broker trading platform. Connect any MT5 broker, trade forex, gold, and crypto with fused AI + technical + sentiment signals.

## Features

- **Multi-broker MT5 support** — FBS, Exness, IC Markets, XM, Pepperstone, HFM, or custom server
- **Secure credential storage** — OS keyring or encrypted-at-rest (never plaintext in source)
- **Real market data** — Finnhub (news/sentiment), Alpha Vantage (macro sentiment), FMP (economic calendar)
- **Signal fusion** — AI + technical + sniper price action + live sentiment (25% weight)
- **Trade ledger** — Every closed trade recorded as win/loss/breakeven with full P&L history
- **Online learning** — Models adapt from real trade outcomes via SGD partial_fit
- **Gold & BTC** — Per-symbol pip/tick math from broker
- **Live trading safety gate** — Password confirmation, hard kill-switch on drawdown
- **FastAPI backend + React PWA UI** — Works on desktop and mobile

## Requirements

- **Windows** (MetaTrader5 Python package requires a local MT5 terminal)
- Python 3.10+
- Node.js 18+ (for UI build)
- MetaTrader 5 terminal installed

## Quick Start

```bash
# Install Python dependencies
pip install -r requirements.txt

# Configure API keys (already in .env if set up)
cp .env.example .env
# Edit .env with your FINNHUB_API_KEY, ALPHA_VANTAGE_API_KEY, FMP_API_KEY

# Start API server
python -m api.server

# Build UI (separate terminal)
cd ui && npm install && npm run build

# Or run UI in dev mode
cd ui && npm run dev
```

Open http://localhost:8000 (production) or http://localhost:3000 (dev).

## API Keys

Set in `.env`:

```
FINNHUB_API_KEY=your_key
ALPHA_VANTAGE_API_KEY=your_key
FMP_API_KEY=your_key
```

Rate limits are respected via in-memory TTL caching (15–30 min refresh).

## Download Historical Data

After connecting a broker account via the UI:

```bash
python download_all_history.py
python train_models.py
```

## Run Trading Bot (CLI)

```bash
python main.py
```

## Run Tests

```bash
pytest tests/ -v
```

## Important Notes

- **MT5 only works on Windows** — For cloud/Linux deployment, use a Windows VM or a broker REST API
- **Not financial advice** — Automated leveraged trading carries real risk of loss
- **Rotate exposed credentials** — If demo passwords were previously in source code, change them on the broker side
- **Logo resolution** — Source logo is 1408×768; sufficient for UI header and mobile icons

## Project Structure

```
aero-trade/
├── api/              # FastAPI backend
├── core/
│   ├── broker/       # MT5 client, broker registry, credential store
│   ├── engine/       # AeroTradeBot trading loop
│   ├── ledger/       # SQLite trade ledger
│   └── symbols.py    # Pip/tick utilities
├── data/
│   ├── market_analysis.py  # Finnhub + Alpha Vantage + FMP
│   └── history_downloader.py
├── ui/               # React PWA dashboard
├── assets/branding/  # Logo and favicons
└── tests/
```

## License

Proprietary — AERO TRADE
