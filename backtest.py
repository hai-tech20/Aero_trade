"""
Backtesting Engine - Test trading strategies on historical data
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.logger import logger
from ai_engine.models.predictor import ai_predictor
from decision_maker.signal_fusion import signal_fusion
from core.risk_management.risk_manager import risk_manager
from data.market_data import market_data

class BacktestEngine:
    """Backtesting engine for strategy evaluation"""
    
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = []
        self.trade_history = []
        self.equity_curve = []
        
        # Performance metrics
        self.metrics = {}
        
        logger.info(f"Backtest Engine initialized with ${initial_balance:,.2f}")
    
    async def run_backtest(self, symbol: str = "EURUSD",
                          timeframe: str = "5min",
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None):
        """
        Run backtest on historical data
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe for data
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        """
        logger.info("=" * 60)
        logger.info("🧪 BACKTEST STARTED")
        logger.info("=" * 60)
        
        try:
            # Load historical data
            df = await self._load_historical_data(symbol, timeframe, start_date, end_date)
            
            if df is None or len(df) < 100:
                logger.error("Insufficient data for backtest")
                return False
            
            logger.info(f"Backtesting {symbol} {timeframe}")
            logger.info(f"Period: {df.index[0].date()} to {df.index[-1].date()}")
            logger.info(f"Data points: {len(df)}")
            
            # Initialize AI predictor
            if not ai_predictor.is_trained:
                logger.warning("AI models not trained. Training with backtest data...")
                await self._train_with_data(df, symbol, timeframe)
            
            # Run simulation
            await self._simulate_trading(df, symbol, timeframe)
            
            # Calculate metrics
            self._calculate_metrics()
            
            # Print results
            self._print_results()
            
            return True
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def _load_historical_data(self, symbol: str, timeframe: str,
                                  start_date: Optional[str], end_date: Optional[str]) -> Optional[pd.DataFrame]:
        """Load historical data — prefers real parquet history."""
        try:
            from config.settings import DATA_DIR
            history_path = DATA_DIR / f"{symbol}_{timeframe}_history.parquet"
            if history_path.exists():
                logger.info(f"Loading real history: {history_path}")
                df = pd.read_parquet(history_path)
            else:
                await market_data.connect()
                df = await market_data.get_latest_data(symbol, timeframe, lookback=10000)
                logger.warning("[MOCK DATA] Backtest using synthetic data — run download_all_history.py first")
            
            if df is None:
                logger.error("No data available")
                return None
            
            # Filter by date if specified
            if start_date:
                start_dt = pd.to_datetime(start_date)
                df = df[df.index >= start_dt]
            
            if end_date:
                end_dt = pd.to_datetime(end_date)
                df = df[df.index <= end_dt]
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            return None
    
    async def _train_with_data(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Train AI models with historical data"""
        try:
            # Prepare training data
            X = []
            y = []
            
            lookback = 20
            horizon = 5
            
            for i in range(lookback, len(df) - horizon):
                window = df.iloc[i-lookback:i]
                features = await ai_predictor.prepare_features(window)
                
                if features.size > 0:
                    current_price = df['close'].iloc[i]
                    future_price = df['close'].iloc[i + horizon]
                    label = 1 if future_price > current_price else 0
                    
                    X.append(features.flatten())
                    y.append(label)
            
            X_train = np.array(X)
            y_train = np.array(y)
            
            if len(X_train) > 100:
                # Train models
                await ai_predictor.train(X_train, y_train, model_name='rf_5min', save=False)
                logger.success(f"Trained model with {len(X_train)} samples")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
    
    async def _simulate_trading(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Simulate trading on historical data"""
        logger.info("Simulating trades...")
        
        lookback = 100
        trade_count = 0
        
        for i in range(lookback, len(df) - 10):  # Leave room for exit
            try:
                # Get data up to current point
                current_data = df.iloc[:i+1]
                current_price = df['close'].iloc[i]
                current_time = df.index[i]
                
                # Update market data cache for AI
                market_data.data_cache[f"{symbol}_{timeframe}"] = current_data
                
                # Get AI prediction
                ai_prediction = await ai_predictor.predict(current_data, timeframe=timeframe)
                
                if ai_prediction.direction == 0:
                    continue  # No signal
                
                # Check if we should enter trade
                if not self.positions:  # No open positions
                    # Calculate position size
                    position_size = risk_manager.calculate_position_size(
                        self.balance,
                        0.02,  # 2% risk
                        20,    # 20 pip SL
                        symbol,
                        ai_prediction.confidence
                    )
                    
                    if position_size <= 0.01:  # Minimum lot
                        continue
                    
                    # Enter trade
                    trade = {
                        'entry_time': current_time,
                        'entry_price': current_price,
                        'symbol': symbol,
                        'direction': 'BUY' if ai_prediction.direction == 1 else 'SELL',
                        'size': position_size,
                        'stop_loss': current_price * (0.98 if ai_prediction.direction == 1 else 1.02),
                        'take_profit': current_price * (1.03 if ai_prediction.direction == 1 else 0.97),
                        'ai_confidence': ai_prediction.confidence
                    }
                    
                    self.positions.append(trade)
                    trade_count += 1
                    
                    logger.debug(f"Trade #{trade_count}: {trade['direction']} {symbol} "
                                f"@{current_price:.4f}")
                
                # Check open positions for exit
                for position in self.positions[:]:  # Copy list
                    # Check stop loss
                    if position['direction'] == 'BUY':
                        if current_price <= position['stop_loss']:
                            self._close_position(position, current_price, current_time, 'SL')
                            self.positions.remove(position)
                        elif current_price >= position['take_profit']:
                            self._close_position(position, current_price, current_time, 'TP')
                            self.positions.remove(position)
                    else:  # SELL
                        if current_price >= position['stop_loss']:
                            self._close_position(position, current_price, current_time, 'SL')
                            self.positions.remove(position)
                        elif current_price <= position['take_profit']:
                            self._close_position(position, current_price, current_time, 'TP')
                            self.positions.remove(position)
                
                # Record equity
                self.equity_curve.append(self.balance)
                
                # Progress update
                if i % 1000 == 0:
                    progress = (i / len(df)) * 100
                    logger.info(f"Progress: {progress:.1f}% | Balance: ${self.balance:,.2f} | Trades: {trade_count}")
            
            except Exception as e:
                logger.error(f"Error at step {i}: {e}")
                continue
        
        # Close any remaining positions
        for position in self.positions:
            self._close_position(position, df['close'].iloc[-1], df.index[-1], 'END')
        
        logger.info(f"Backtest completed: {trade_count} trades executed")
    
    def _close_position(self, position: Dict, exit_price: float, exit_time, reason: str):
        """Close a position and calculate P&L"""
        # Calculate P&L
        if position['direction'] == 'BUY':
            pnl_pips = (exit_price - position['entry_price']) * 10000
        else:  # SELL
            pnl_pips = (position['entry_price'] - exit_price) * 10000
        
        # Simplified: 1 pip = $10 per lot
        pnl_usd = pnl_pips * 10 * position['size']
        
        # Update balance
        self.balance += pnl_usd
        
        # Record trade
        trade = {
            'entry_time': position['entry_time'],
            'exit_time': exit_time,
            'symbol': position['symbol'],
            'direction': position['direction'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'size': position['size'],
            'pnl_pips': pnl_pips,
            'pnl_usd': pnl_usd,
            'reason': reason,
            'ai_confidence': position.get('ai_confidence', 0)
        }
        
        self.trade_history.append(trade)
        risk_manager.record_trade(trade)
    
    def _calculate_metrics(self):
        """Calculate backtest performance metrics"""
        if not self.trade_history:
            self.metrics = {'error': 'No trades executed'}
            return
        
        # Extract P&L data
        pnls = [t['pnl_usd'] for t in self.trade_history]
        
        # Basic metrics
        total_trades = len(self.trade_history)
        winning_trades = sum(1 for pnl in pnls if pnl > 0)
        losing_trades = sum(1 for pnl in pnls if pnl < 0)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl = sum(pnls)
        total_return = (self.balance - self.initial_balance) / self.initial_balance * 100
        
        # Profit factor
        gross_profit = sum(pnl for pnl in pnls if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in pnls if pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average trade
        avg_win = np.mean([pnl for pnl in pnls if pnl > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([abs(pnl) for pnl in pnls if pnl < 0]) if losing_trades > 0 else 0
        
        # Expectancy
        expectancy = (win_rate/100 * avg_win) - ((1 - win_rate/100) * avg_loss)
        
        # Max drawdown
        equity_curve = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max * 100
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
        
        # Sharpe ratio (simplified)
        if len(pnls) > 1 and np.std(pnls) > 0:
            sharpe_ratio = np.mean(pnls) / np.std(pnls) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        self.metrics = {
            'initial_balance': self.initial_balance,
            'final_balance': self.balance,
            'total_return_pct': total_return,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate_pct': win_rate,
            'total_pnl': total_pnl,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'expectancy': expectancy,
            'max_drawdown_pct': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'avg_ai_confidence': np.mean([t.get('ai_confidence', 0) for t in self.trade_history])
        }
    
    def _print_results(self):
        """Print backtest results"""
        logger.info("=" * 60)
        logger.info("📊 BACKTEST RESULTS")
        logger.info("=" * 60)
        
        metrics = self.metrics
        
        if 'error' in metrics:
            logger.error(metrics['error'])
            return
        
        logger.info(f"Initial Balance: ${metrics['initial_balance']:,.2f}")
        logger.info(f"Final Balance: ${metrics['final_balance']:,.2f}")
        logger.info(f"Total Return: {metrics['total_return_pct']:+.2f}%")
        logger.info("")
        logger.info(f"Total Trades: {metrics['total_trades']}")
        logger.info(f"Winning Trades: {metrics['winning_trades']}")
        logger.info(f"Losing Trades: {metrics['losing_trades']}")
        logger.info(f"Win Rate: {metrics['win_rate_pct']:.1f}%")
        logger.info("")
        logger.info(f"Total P&L: ${metrics['total_pnl']:+,.2f}")
        logger.info(f"Profit Factor: {metrics['profit_factor']:.2f}")
        logger.info(f"Expectancy: ${metrics['expectancy']:+.2f}")
        logger.info("")
        logger.info(f"Avg Win: ${metrics['avg_win']:+.2f}")
        logger.info(f"Avg Loss: ${metrics['avg_loss']:+.2f}")
        logger.info(f"Max Drawdown: {metrics['max_drawdown_pct']:.1f}%")
        logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info("")
        logger.info(f"Avg AI Confidence: {metrics['avg_ai_confidence']:.2f}")
        
        if metrics['total_pnl'] > 0:
            logger.success("✅ Backtest PASSED - Strategy is profitable")
        else:
            logger.warning("⚠️ Backtest FAILED - Strategy is not profitable")
        
        logger.info("=" * 60)
    
    def save_results(self, filename: str = "backtest_results.csv"):
        """Save backtest results to CSV"""
        try:
            if self.trade_history:
                df = pd.DataFrame(self.trade_history)
                df.to_csv(filename, index=False)
                logger.success(f"Results saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")

async def main():
    """Main backtest function"""
    engine = BacktestEngine(initial_balance=10000.0)
    
    # Run backtest
    success = await engine.run_backtest(
        symbol="EURUSD",
        timeframe="5min",
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    
    if success:
        # Save results
        engine.save_results()
    
    return success

if __name__ == "__main__":
    asyncio.run(main())