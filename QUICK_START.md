# Quick Start Guide - FBS AI Trading Bot

## 🚀 Running the Bot

### Option 1: Simple Run (Recommended for first time)
```bash
python run_bot_simple.py
```
This will:
- Initialize the bot in DEMO mode (safe)
- Connect to demo account ($10,000 balance)
- Start trading loop
- Press `Ctrl+C` to stop

### Option 2: Using the batch file (Windows)
```bash
run_bot.bat
```

### Option 3: Direct Python
```bash
python main.py
```

## 📊 Training AI Models (Optional but Recommended)

Before the bot can make good predictions, you should train the AI models:

```bash
python train_models.py
```

This will:
- Generate training data from market history
- Train the AI models
- Save models to `ai_engine/models/saved/`

**Note:** The bot will work without trained models, but predictions will be neutral (HOLD signals).

## 🧪 Testing

Test if everything is set up correctly:
```bash
python test_imports.py
```

Quick bot test:
```bash
python test_bot.py
```

## ⚙️ Configuration

### Demo Mode (Default - Safe)
The bot runs in demo mode by default. No real money is at risk.

### Real Account (Advanced)
1. Edit `config/credentials.py`
2. Set `USE_DEMO = False`
3. Add your real FBS credentials
4. **WARNING:** Only do this after thorough testing!

## 📝 Current Status

✅ Bot initializes successfully
✅ All components working
✅ Demo mode active
⚠️ AI models need training for better predictions

## 🛑 Stopping the Bot

Press `Ctrl+C` in the terminal to stop the bot gracefully.

## 📁 Project Structure

- `main.py` - Main bot orchestrator
- `config/` - Configuration files
- `ai_engine/` - AI prediction models
- `core/` - Core trading components (broker, risk management)
- `data/` - Market data handling
- `decision_maker/` - Signal fusion engine

## 🆘 Troubleshooting

**Import errors?**
```bash
pip install -r requirements.txt
```

**Bot not making trades?**
- Models may need training: `python train_models.py`
- Check confidence thresholds in `config/settings.py`

**Want to see more details?**
- Check `logs/trading_bot.log` (if logging is enabled)
