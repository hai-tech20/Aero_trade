"""
Interactive Settings Adjuster
Easily adjust trading bot settings
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from config.settings import (
    TRADING_CONFIG, AI_CONFIG, SIGNAL_FUSION_CONFIG, 
    RISK_CONFIG, BROKER_CONFIG
)

def show_current_settings():
    """Display current settings"""
    print("=" * 70)
    print("  CURRENT TRADING BOT SETTINGS")
    print("=" * 70)
    
    print("\n[TRADING] TRADING CONFIGURATION:")
    print(f"   Demo Mode: {TRADING_CONFIG.get('demo_mode', True)}")
    print(f"   Trading Pairs: {', '.join(TRADING_CONFIG.get('pairs', []))}")
    print(f"   Timeframes: {', '.join(TRADING_CONFIG.get('timeframes', []))}")
    print(f"   Risk Per Trade: {TRADING_CONFIG.get('risk_per_trade', 0.02)*100:.1f}%")
    print(f"   Max Daily Loss: {TRADING_CONFIG.get('max_daily_loss', 0.05)*100:.1f}%")
    print(f"   Max Positions: {TRADING_CONFIG.get('max_positions', 3)}")
    print(f"   Max Position Size: {TRADING_CONFIG.get('max_position_size', 0.05)*100:.1f}%")
    print(f"   Default Lot Size: {TRADING_CONFIG.get('default_lot_size', 0.01)}")
    
    print("\n[AI] AI CONFIGURATION:")
    print(f"   AI Enabled: {AI_CONFIG.get('enabled', True)}")
    print(f"   Min Confidence: {AI_CONFIG.get('min_confidence', 0.65)*100:.0f}%")
    print(f"   Update Interval: {AI_CONFIG.get('update_interval', 30)} seconds")
    
    print("\n[RISK] RISK MANAGEMENT:")
    print(f"   Stop Loss: {RISK_CONFIG.get('stop_loss_pips', 20)} pips")
    print(f"   Take Profit: {RISK_CONFIG.get('take_profit_pips', 40)} pips")
    print(f"   Max Drawdown: {RISK_CONFIG.get('max_drawdown_pct', 10.0):.1f}%")
    print(f"   Max Daily Trades: {RISK_CONFIG.get('max_daily_trades', 10)}")
    print(f"   Max Consecutive Losses: {RISK_CONFIG.get('max_consecutive_losses', 3)}")
    print(f"   Max Position Size: {RISK_CONFIG.get('max_position_pct', 5.0):.1f}% of balance")
    print(f"   Max Daily Loss: {RISK_CONFIG.get('max_daily_loss_pct', 5.0):.1f}% of balance")
    
    print("\n[SIGNAL] SIGNAL FUSION:")
    weights = SIGNAL_FUSION_CONFIG.get('weights', {})
    print(f"   AI Weight: {weights.get('ai', 0.4)*100:.0f}%")
    print(f"   Technical Weight: {weights.get('technical', 0.25)*100:.0f}%")
    print(f"   Price Action Weight: {weights.get('price_action', 0.2)*100:.0f}%")
    min_conf = SIGNAL_FUSION_CONFIG.get('min_confidence', {})
    print(f"   Min AI Confidence: {min_conf.get('ai', 0.65)*100:.0f}%")
    print(f"   Min Combined Confidence: {min_conf.get('combined', 0.7)*100:.0f}%")
    
    print("\n" + "=" * 70)

def adjust_setting():
    """Interactive setting adjustment"""
    print("\nWhich setting would you like to adjust?")
    print("\n1. Risk Per Trade (currently {:.1f}%)".format(TRADING_CONFIG.get('risk_per_trade', 0.02)*100))
    print("2. Min AI Confidence (currently {:.0f}%)".format(AI_CONFIG.get('min_confidence', 0.65)*100))
    print("3. Stop Loss (currently {} pips)".format(RISK_CONFIG.get('stop_loss_pips', 20)))
    print("4. Take Profit (currently {} pips)".format(RISK_CONFIG.get('take_profit_pips', 40)))
    print("5. Max Positions (currently {})".format(TRADING_CONFIG.get('max_positions', 3)))
    print("6. Max Daily Trades (currently {})".format(RISK_CONFIG.get('max_daily_trades', 10)))
    print("7. Show all settings")
    print("8. Exit")
    
    choice = input("\nEnter choice (1-8): ").strip()
    
    if choice == '1':
        new_value = input(f"Enter new risk per trade (0.01-0.05, current: {TRADING_CONFIG.get('risk_per_trade', 0.02)}): ")
        try:
            value = float(new_value)
            if 0.01 <= value <= 0.05:
                TRADING_CONFIG['risk_per_trade'] = value
                print(f"[OK] Risk per trade set to {value*100:.1f}%")
            else:
                print("[ERROR] Value must be between 0.01 (1%) and 0.05 (5%)")
        except:
            print("[ERROR] Invalid input")
    
    elif choice == '2':
        new_value = input(f"Enter new min confidence (0.50-0.95, current: {AI_CONFIG.get('min_confidence', 0.65)}): ")
        try:
            value = float(new_value)
            if 0.50 <= value <= 0.95:
                AI_CONFIG['min_confidence'] = value
                print(f"[OK] Min confidence set to {value*100:.0f}%")
            else:
                print("[ERROR] Value must be between 0.50 (50%) and 0.95 (95%)")
        except:
            print("[ERROR] Invalid input")
    
    elif choice == '3':
        new_value = input(f"Enter new stop loss in pips (10-100, current: {RISK_CONFIG.get('stop_loss_pips', 20)}): ")
        try:
            value = int(new_value)
            if 10 <= value <= 100:
                RISK_CONFIG['stop_loss_pips'] = value
                print(f"[OK] Stop loss set to {value} pips")
            else:
                print("[ERROR] Value must be between 10 and 100 pips")
        except:
            print("[ERROR] Invalid input")
    
    elif choice == '4':
        new_value = input(f"Enter new take profit in pips (20-200, current: {RISK_CONFIG.get('take_profit_pips', 40)}): ")
        try:
            value = int(new_value)
            if 20 <= value <= 200:
                RISK_CONFIG['take_profit_pips'] = value
                print(f"[OK] Take profit set to {value} pips")
            else:
                print("[ERROR] Value must be between 20 and 200 pips")
        except:
            print("[ERROR] Invalid input")
    
    elif choice == '5':
        new_value = input(f"Enter new max positions (1-10, current: {TRADING_CONFIG.get('max_positions', 3)}): ")
        try:
            value = int(new_value)
            if 1 <= value <= 10:
                TRADING_CONFIG['max_positions'] = value
                print(f"[OK] Max positions set to {value}")
            else:
                print("[ERROR] Value must be between 1 and 10")
        except:
            print("[ERROR] Invalid input")
    
    elif choice == '6':
        new_value = input(f"Enter new max daily trades (1-50, current: {RISK_CONFIG.get('max_daily_trades', 10)}): ")
        try:
            value = int(new_value)
            if 1 <= value <= 50:
                RISK_CONFIG['max_daily_trades'] = value
                print(f"[OK] Max daily trades set to {value}")
            else:
                print("[ERROR] Value must be between 1 and 50")
        except:
            print("[ERROR] Invalid input")
    
    elif choice == '7':
        show_current_settings()
        return
    
    elif choice == '8':
        print("Exiting...")
        return False
    
    else:
        print("✗ Invalid choice")
    
    return True

def save_settings():
    """Save settings to file"""
    print("\n[NOTE] Settings are currently in memory.")
    print("To make changes permanent, edit config/settings.py directly")
    print("Current settings will be used until bot is restarted.")

def main():
    """Main menu"""
    while True:
        show_current_settings()
        if not adjust_setting():
            break
        save_settings()
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
