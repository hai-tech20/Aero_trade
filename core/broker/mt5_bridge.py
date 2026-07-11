"""
MetaTrader 5 Bridge — broker-agnostic
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, List

from utils.logger import logger
from config.settings import APP_NAME, BROKER_CONFIG
from core.symbols import get_pip_size, symbol_asset_class, pips_to_price_distance


class MT5Bridge:
    """Connect to any MT5 broker."""

    def __init__(self):
        self.connected = False
        self.account_info = {}

    def connect(self, login: int, password: str, server: str, demo: bool = True) -> bool:
        try:
            if not mt5.initialize():
                logger.error(f"MT5 initialize failed: {mt5.last_error()}")
                return False
            authorized = mt5.login(login=login, password=password, server=server)
            if authorized:
                self.connected = True
                self.account_info = mt5.account_info()._asdict()
                account_type = "DEMO" if demo else "REAL"
                broker_name = BROKER_CONFIG.get("name") or server
                logger.success(f"{APP_NAME}: Connected to {broker_name} {account_type} #{login}")
                return True
            logger.error(f"Login failed: {mt5.last_error()}")
            return False
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False

    def get_account_info(self) -> Dict:
        if not self.connected:
            return {}
        info = mt5.account_info()
        if info:
            self.account_info = info._asdict()
        return self.account_info.copy()

    def get_symbol_info(self, symbol: str) -> Dict:
        try:
            info = mt5.symbol_info(symbol)
            if info is None:
                return {}
            return info._asdict()
        except Exception:
            return {}

    def _sl_tp_prices(self, symbol: str, order_type: str, price: float, sl_pips: float, tp_pips: float):
        """Compute SL/TP price levels per asset class."""
        cls = symbol_asset_class(symbol)
        if cls == "crypto":
            from config.settings import RISK_CONFIG
            sl_dist = sl_pips if sl_pips > 10 else RISK_CONFIG.get("crypto_sl_dollars", 200)
            tp_dist = tp_pips if tp_pips > 10 else RISK_CONFIG.get("crypto_tp_dollars", 150)
            if order_type == "BUY":
                return price - sl_dist, price + tp_dist
            return price + sl_dist, price - tp_dist
        sl_dist = pips_to_price_distance(symbol, sl_pips)
        tp_dist = pips_to_price_distance(symbol, tp_pips)
        if order_type == "BUY":
            return price - sl_dist, price + tp_dist
        return price + sl_dist, price - tp_dist

    def place_order(self, symbol: str, order_type: str, volume: float, **kwargs) -> Dict:
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return {"success": False, "error": f"Symbol {symbol} not found"}
            if not symbol_info.visible:
                mt5.symbol_select(symbol, True)

            tick = mt5.symbol_info_tick(symbol)
            price = tick.ask if order_type == "BUY" else tick.bid

            order = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL,
                "price": price,
                "deviation": kwargs.get("deviation", 20),
                "magic": kwargs.get("magic", 234000),
                "comment": kwargs.get("comment", APP_NAME),
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            sl_pips = kwargs.get("stop_loss", 0)
            tp_pips = kwargs.get("take_profit", 0)
            if sl_pips or tp_pips:
                sl_price, tp_price = self._sl_tp_prices(symbol, order_type, price, sl_pips, tp_pips)
                if sl_pips:
                    order["sl"] = sl_price
                if tp_pips:
                    order["tp"] = tp_price

            result = mt5.order_send(order)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.trade(f"Order: {order_type} {symbol} {volume} lots @ {price:.5f}")
                return {
                    "success": True,
                    "order_id": result.order,
                    "ticket": result.deal,
                    "price": result.price,
                    "volume": result.volume,
                }
            return {"success": False, "error": result.comment, "retcode": result.retcode}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def close_order(self, ticket: int) -> Dict:
        try:
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return {"success": False, "error": "Position not found"}
            position = position[0]
            close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            tick = mt5.symbol_info_tick(position.symbol)
            price = tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask
            close_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": close_type,
                "position": ticket,
                "price": price,
                "deviation": 20,
                "magic": position.magic,
                "comment": f"{APP_NAME} Close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(close_request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                return {"success": True, "ticket": result.deal, "pnl_usd": position.profit}
            return {"success": False, "error": result.comment}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_positions(self, symbol: str = "") -> List[Dict]:
        try:
            positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
            return [p._asdict() for p in positions] if positions else []
        except Exception:
            return []

    def get_deal_history(self, limit: int = 100) -> List[Dict]:
        try:
            from datetime import timedelta
            deals = mt5.history_deals_get(datetime.now() - timedelta(days=30), datetime.now())
            if not deals:
                return []
            result = []
            for d in deals[-limit:]:
                dd = d._asdict()
                result.append({
                    "order_id": dd.get("position_id", dd.get("ticket")),
                    "symbol": dd.get("symbol"),
                    "pnl_usd": dd.get("profit", 0),
                    "profit": dd.get("profit", 0),
                    "volume": dd.get("volume"),
                    "time": dd.get("time"),
                })
            return result
        except Exception:
            return []

    def get_market_price(self, symbol: str) -> Dict:
        try:
            tick = mt5.symbol_info_tick(symbol)
            pip = get_pip_size(symbol)
            return {
                "symbol": symbol,
                "bid": tick.bid,
                "ask": tick.ask,
                "time": tick.time,
                "spread": (tick.ask - tick.bid) / pip if pip else 0,
            }
        except Exception:
            return {}

    def get_all_symbols(self) -> List[str]:
        try:
            symbols = mt5.symbols_get()
            if symbols is None:
                return []
            tradeable = []
            for symbol in symbols:
                sd = symbol._asdict()
                if sd.get("trade_mode", 0) in [0, 2, 4] and sd.get("visible", False):
                    tradeable.append(sd["name"])
            return tradeable
        except Exception as e:
            logger.error(f"Error getting symbols: {e}")
            return []

    def _tf_map(self):
        return {
            "1min": mt5.TIMEFRAME_M1, "5min": mt5.TIMEFRAME_M5,
            "15min": mt5.TIMEFRAME_M15, "30min": mt5.TIMEFRAME_M30,
            "1h": mt5.TIMEFRAME_H1, "4h": mt5.TIMEFRAME_H4, "1d": mt5.TIMEFRAME_D1,
        }

    def get_historical_data(self, symbol: str, timeframe: str, count: int = 1000) -> Optional[pd.DataFrame]:
        try:
            mt5_tf = self._tf_map().get(timeframe, mt5.TIMEFRAME_M5)
            rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, count)
            if rates is None or len(rates) == 0:
                return None
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.set_index("time", inplace=True)
            df.columns = [c.lower() for c in df.columns]
            return df
        except Exception as e:
            logger.error(f"Historical data error {symbol}: {e}")
            return None

    def get_historical_data_before(self, symbol: str, timeframe: str, before_dt: datetime, count: int = 5000) -> Optional[pd.DataFrame]:
        try:
            mt5_tf = self._tf_map().get(timeframe, mt5.TIMEFRAME_M5)
            rates = mt5.copy_rates_from(symbol, mt5_tf, before_dt, count)
            if rates is None or len(rates) == 0:
                return None
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.set_index("time", inplace=True)
            df.columns = [c.lower() for c in df.columns]
            return df
        except Exception as e:
            logger.error(f"Paged history error {symbol}: {e}")
            return None

    def get_tick_value(self, symbol: str) -> float:
        info = self.get_symbol_info(symbol)
        if not info:
            return 10.0
        tick_val = info.get("trade_tick_value", 0)
        tick_size = info.get("trade_tick_size", 0.00001)
        pip = get_pip_size(symbol, info.get("point", 0.00001))
        if tick_val and tick_size:
            return tick_val / tick_size * pip
        return 10.0

    def disconnect(self):
        mt5.shutdown()
        self.connected = False
        logger.info("Disconnected from MT5")


mt5_bridge = MT5Bridge()
