# 📊 Monitoring & Settings Guide

## 🔍 1. Monitor the Bot

### Real-Time Monitoring
Run the monitor to see what the bot is doing:

```bash
python monitor_bot.py
```

**What you'll see:**
- 💰 **Account Status** - Balance, equity, profit, margin
- 📊 **Open Positions** - All active trades
- 📈 **Signal Statistics** - AI predictions and confidence levels
- 🤖 **AI Model Status** - Whether models are trained
- 💹 **Market Data** - Current prices and spreads
- 📊 **Trading Statistics** - Win rate, total P&L, profit factor

**Updates every 10 seconds** - Press `Ctrl+C` to stop

### Quick Status Check
```bash
python test_mt5_connection.py
```

Shows:
- Connection status
- Account balance
- Current market prices

---

## ⚙️ 2. Adjust Trading Settings

### Interactive Settings Adjuster
```bash
python adjust_settings.py
```

**You can adjust:**
1. **Risk Per Trade** (1-5% of balance)
2. **Min AI Confidence** (50-95% - higher = more conservative)
3. **Stop Loss** (10-100 pips)
4. **Take Profit** (20-200 pips)
5. **Max Positions** (1-10 simultaneous trades)
6. **Max Daily Trades** (1-50 trades per day)

### Manual Settings Edit
For permanent changes, edit `config/settings.py`:

#### Risk Management
```python
RISK_CONFIG = {
    "stop_loss_pips": 20,        # Stop loss in pips
    "take_profit_pips": 40,       # Take profit in pips
    "max_drawdown_pct": 10.0,     # Max drawdown before stopping
    "max_daily_trades": 10,       # Max trades per day
    "max_consecutive_losses": 3,  # Stop after X losses
}
```

#### AI Confidence
```python
AI_CONFIG = {
    "enabled": True,
    "min_confidence": 0.65,  # 65% - only trade when very confident
    "update_interval": 30,   # Check every 30 seconds
}
```

#### Trading Risk
```python
TRADING_CONFIG = {
    "risk_per_trade": 0.02,      # 2% risk per trade
    "max_daily_loss": 0.05,       # Stop if lose 5% in a day
    "max_positions": 3,           # Max 3 open positions
    "max_position_size": 0.05,    # Max 5% of balance per position
}
```

---

## 📈 Recommended Settings

### Conservative (Safer)
- Risk per trade: **1%** (0.01)
- Min confidence: **75%** (0.75)
- Stop loss: **30 pips**
- Take profit: **60 pips**
- Max positions: **2**

### Moderate (Balanced) - Current
- Risk per trade: **2%** (0.02)
- Min confidence: **65%** (0.65)
- Stop loss: **20 pips**
- Take profit: **40 pips**
- Max positions: **3**

### Aggressive (Higher Risk)
- Risk per trade: **3%** (0.03)
- Min confidence: **60%** (0.60)
- Stop loss: **15 pips**
- Take profit: **30 pips**
- Max positions: **5**

---

## 🎯 Quick Tips

### To See More Trades:
- Lower `min_confidence` (e.g., 0.60 = 60%)
- Increase `max_daily_trades`
- Increase `max_positions`

### To Be More Conservative:
- Raise `min_confidence` (e.g., 0.75 = 75%)
- Lower `risk_per_trade` (e.g., 0.01 = 1%)
- Reduce `max_positions`

### To Improve Predictions:
```bash
python train_models.py
```
Trains AI models on historical data for better predictions.

---

## 📱 Monitor on Your Phone

**Check your MT5 app:**
1. Open MT5 mobile app
2. Log into your FBS account
3. Go to "Trade" tab
4. Look for positions with comment "AI Trading Bot"
5. Refresh to see new trades

**All trades will show:**
- Entry price
- Current P&L
- Stop loss and take profit levels
- Open time

---

## 🔄 Restart Bot After Settings Change

After changing settings in `config/settings.py`:
1. Stop the bot (`Ctrl+C`)
2. Restart: `python run_bot_real.py`

Settings from `adjust_settings.py` are temporary (until restart).

---

## 📊 Understanding the Monitor

### Account Status
- **Balance**: Total account balance
- **Equity**: Balance + unrealized P&L
- **Profit**: Total profit/loss
- **Free Margin**: Available margin for new trades
- **Margin Level**: Safety indicator (should be > 100%)

### Signal Statistics
- **Total Signals**: All signals generated
- **Recent Signals**: Last hour
- **Avg Confidence**: Average prediction confidence
- **By Source**: Breakdown by AI, technical, etc.

### AI Model Status
- **Trained**: Number of trained models
- If 0/3: Models need training (`python train_models.py`)

---

**Need help?** Check the logs or run `python test_imports.py` to verify everything is working.
