"""
Real-time Bot Monitor
Shows what the bot is doing in real-time
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime
import time

sys.path.append(str(Path(__file__).parent))

from core.broker.fbs_client import fbs_client
from decision_maker.signal_fusion import signal_fusion
from ai_engine.models.predictor import ai_predictor

async def monitor_loop():
    """Monitor bot activity"""
    print("=" * 70)
    print("  FBS AI TRADING BOT - REAL-TIME MONITOR")
    print("=" * 70)
    print("\nMonitoring bot activity...")
    print("Press Ctrl+C to stop monitoring\n")
    
    # Connect to broker
    await fbs_client.connect()
    
    iteration = 0
    
    try:
        while True:
            iteration += 1
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            print(f"\n[{timestamp}] Iteration #{iteration}")
            print("-" * 70)
            
            # 1. Account Status
            account = await fbs_client.get_account_info()
            if account:
                balance = account.get('balance', 0)
                equity = account.get('equity', 0)
                profit = account.get('profit', 0)
                margin = account.get('margin', 0)
                free_margin = account.get('free_margin', 0)
                
                print(f"[ACCOUNT] Account Status:")
                print(f"   Balance: ${balance:,.2f}")
                print(f"   Equity: ${equity:,.2f}")
                print(f"   Profit: ${profit:+,.2f}")
                print(f"   Free Margin: ${free_margin:,.2f}")
                if margin > 0:
                    margin_level = (equity / margin * 100) if margin > 0 else 0
                    print(f"   Margin Level: {margin_level:.1f}%")
            
            # 2. Open Positions
            positions = await fbs_client.get_open_orders()
            if positions:
                print(f"\n[POSITIONS] Open Positions: {len(positions)}")
                for pos in positions:
                    symbol = pos.get('symbol', 'N/A')
                    volume = pos.get('volume', 0)
                    order_id = pos.get('order_id', 'N/A')
                    print(f"   {symbol} | Volume: {volume} lots | ID: {order_id}")
            else:
                print(f"\n[POSITIONS] Open Positions: 0")
            
            # 3. Signal Statistics
            signal_stats = await signal_fusion.get_signal_statistics()
            if signal_stats:
                print(f"\n[SIGNALS] Signal Statistics:")
                print(f"   Total Signals: {signal_stats.get('total_signals', 0)}")
                print(f"   Recent Signals (last hour): {signal_stats.get('recent_signals', 0)}")
                avg_conf = signal_stats.get('avg_confidence', 0)
                if avg_conf > 0:
                    print(f"   Avg Confidence: {avg_conf:.1%}")
                
                by_source = signal_stats.get('by_source', {})
                if by_source:
                    print(f"   By Source:")
                    for source, stats in by_source.items():
                        count = stats.get('count', 0)
                        conf = stats.get('avg_confidence', 0)
                        print(f"     {source}: {count} signals (avg conf: {conf:.1%})")
            
            # 4. AI Model Status
            model_perf = await ai_predictor.get_model_performance()
            if model_perf:
                trained_models = sum(1 for m in model_perf.values() if m.get('is_trained', False))
                total_models = len(model_perf)
                print(f"\n[AI] AI Models:")
                print(f"   Trained: {trained_models}/{total_models}")
                if trained_models == 0:
                    print(f"   [WARNING] Models not trained - predictions will be neutral")
                    print(f"   [TIP] Run 'python train_models.py' to train models")
            
            # 5. Recent Market Data
            price_info = await fbs_client.get_market_price('EURUSD')
            if price_info:
                print(f"\n[MARKET] Market Data (EURUSD):")
                print(f"   Bid: {price_info.get('bid', 0):.5f}")
                print(f"   Ask: {price_info.get('ask', 0):.5f}")
                print(f"   Spread: {price_info.get('spread', 0):.1f} pips")
            
            # 6. Trading Stats
            stats = await fbs_client.get_trading_stats()
            if stats:
                print(f"\n[STATS] Trading Statistics:")
                print(f"   Total Trades: {stats.get('total_trades', 0)}")
                print(f"   Win Rate: {stats.get('win_rate', 0):.1f}%")
                print(f"   Total P&L: ${stats.get('total_pnl', 0):+,.2f}")
                profit_factor = stats.get('profit_factor', 0)
                if profit_factor != float('inf'):
                    print(f"   Profit Factor: {profit_factor:.2f}")
            
            print("\n" + "=" * 70)
            print("Next update in 10 seconds...")
            
            await asyncio.sleep(10)  # Update every 10 seconds
            
    except KeyboardInterrupt:
        print("\n\n[STOPPED] Monitoring stopped by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        await fbs_client.disconnect()

if __name__ == "__main__":
    asyncio.run(monitor_loop())
