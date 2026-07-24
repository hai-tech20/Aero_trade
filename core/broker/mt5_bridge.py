"""
MetaTrader 5 Bridge — broker-agnostic
"""

import MetaTrader5 as mt5
import pandas as pd
import concurrent.futures
import time
import threading
from datetime import datetime
from typing import Any, Callable, Dict, Optional, List, cast

from utils.logger import logger
from config.settings import APP_NAME, BROKER_CONFIG
from core.symbols import get_pip_size, symbol_asset_class, pips_to_price_distance


mt5_api = cast(Any, mt5)


class MT5Bridge:
    """Connect to any MT5 broker."""

    _mt5_lock = threading.Lock()

    def __init__(self):
        self.connected = False
        self.account_info = {}
        self._last_login: Optional[int] = None
        self._last_password: Optional[str] = None
        self._last_server: Optional[str] = None
        self._last_path: Optional[str] = None

    def _call_mt5_with_timeout(self, mt5_func: Callable[..., Any], *args: Any, timeout: int = 10, **kwargs: Any) -> Any:
        """Run an MT5 call in a thread pool with timeout, serialized with a lock."""
        def wrapped():
            with MT5Bridge._mt5_lock:
                return mt5_func(*args, **kwargs)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(wrapped)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            logger.error(f"MT5 call timed out after {timeout}s")
            executor.shutdown(wait=False)
            raise TimeoutError(f"MT5 call timed out after {timeout}s")

    def connect(self, login: int, password: str, server: str, demo: bool = True, path: Optional[str] = None) -> bool:
        try:
            init_kwargs = {}
            if path:
                init_kwargs["path"] = path

            try:
                def do_init():
                    return mt5_api.initialize(**init_kwargs)
                init_ok = self._call_mt5_with_timeout(do_init, timeout=10)
            except TimeoutError:
                logger.error("MT5 initialize timed out — terminal may be stuck on history sync for a broken symbol")
                logger.error("Try: Right-click symbol in MT5 Market Watch -> Refresh, or clear MT5 history cache")
                return False

            if not init_ok:
                err = mt5_api.last_error()
                logger.error(f"MT5 initialize failed: {err}")
                return False

            try:
                def do_login():
                    return mt5_api.login(login=login, password=password, server=server)
                authorized = self._call_mt5_with_timeout(do_login, timeout=10)
            except TimeoutError:
                logger.error("MT5 login timed out")
                try:
                    mt5_api.shutdown()
                except Exception:
                    pass
                return False

            if authorized:
                self.connected = True
                try:
                    info = mt5_api.account_info()
                    if info:
                        self.account_info = info._asdict()
                except Exception as e:
                    logger.error(f"Failed to get account info after login: {e}")
                    try:
                        mt5_api.shutdown()
                    except Exception:
                        pass
                    return False
                account_type = "DEMO" if demo else "REAL"
                broker_name = BROKER_CONFIG.get("name") or server
                self._last_login = login
                self._last_password = password
                self._last_server = server
                self._last_path = path
                logger.success(f"{APP_NAME}: Connected to {broker_name} {account_type} #{login}")
                return True

            err = mt5_api.last_error()
            logger.error(f"Login failed: {err}")
            try:
                mt5_api.shutdown()
            except Exception:
                pass
            return False
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False

    def reconnect(self) -> bool:
        """Attempt to restore a lost MT5 connection using the last known-good credentials."""
        if not self._last_login or not self._last_password or not self._last_server:
            logger.error("Cannot reconnect — no previous successful connection to restore")
            return False
        logger.warning("Attempting MT5 reconnect after lost IPC connection...")
        self.connected = False
        return self.connect(self._last_login, self._last_password, self._last_server, path=self._last_path)

    def get_account_info(self) -> Dict:
        if not self.connected:
            return {}
        try:
            def get_info():
                return mt5_api.account_info()
            info = self._call_mt5_with_timeout(get_info, timeout=10)
            if info:
                self.account_info = info._asdict()
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
        return self.account_info.copy()

    def get_symbol_info(self, symbol: str) -> Dict:
        try:
            def get_info():
                return mt5_api.symbol_info(symbol)
            info = self._call_mt5_with_timeout(get_info, timeout=10)
            if info is None:
                return {}
            return info._asdict()
        except Exception as e:
            logger.error(f"Error getting symbol info for {symbol}: {e}")
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
            # Get symbol info with timeout
            def get_sym_info():
                return mt5_api.symbol_info(symbol)
            symbol_info = self._call_mt5_with_timeout(get_sym_info, timeout=10)
            if symbol_info is None:
                return {"success": False, "error": f"Symbol {symbol} not found"}

            # Ensure symbol is visible/selected with retries and timeout
            if not symbol_info.visible:
                for attempt in range(3):
                    try:
                        def select_sym():
                            return mt5_api.symbol_select(symbol, True)
                        success = self._call_mt5_with_timeout(select_sym, timeout=5)
                        if success:
                            break
                    except Exception as e:
                        logger.warning(f"Attempt {attempt+1} to select symbol {symbol} failed: {e}")
                    time.sleep(0.2)

            # Get tick data with timeout
            def get_tick():
                return mt5_api.symbol_info_tick(symbol)
            tick = self._call_mt5_with_timeout(get_tick, timeout=10)
            if tick is None:
                return {"success": False, "error": f"Could not get tick data for {symbol}"}

            price = tick.ask if order_type == "BUY" else tick.bid

            # Determine filling mode (default to IOC, try RETURN if IOC fails)
            filling_modes = [
                mt5_api.ORDER_FILLING_IOC,
                mt5_api.ORDER_FILLING_RETURN,
                mt5_api.ORDER_FILLING_FOK
            ]

            order = {
                "action": mt5_api.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5_api.ORDER_TYPE_BUY if order_type == "BUY" else mt5_api.ORDER_TYPE_SELL,
                "price": price,
                "deviation": kwargs.get("deviation", 20),
                "magic": kwargs.get("magic", 234000),
                "comment": kwargs.get("comment", APP_NAME),
                "type_time": mt5_api.ORDER_TIME_GTC,
                "type_filling": None,  # Set per attempt
            }

            sl_pips = kwargs.get("stop_loss", 0)
            tp_pips = kwargs.get("take_profit", 0)
            if sl_pips or tp_pips:
                sl_price, tp_price = self._sl_tp_prices(symbol, order_type, price, sl_pips, tp_pips)
                if sl_pips:
                    order["sl"] = sl_price
                if tp_pips:
                    order["tp"] = tp_price

            # Try each filling mode with timeout
            last_result = None
            for filling_mode in filling_modes:
                order["type_filling"] = filling_mode
                try:
                    def send_order():
                        return mt5_api.order_send(order)
                    result = self._call_mt5_with_timeout(send_order, timeout=15)
                    last_result = result
                    if result.retcode == mt5_api.TRADE_RETCODE_DONE:
                        logger.trade(f"Order: {order_type} {symbol} {volume} lots @ {price:.5f}")
                        return {
                            "success": True,
                            "order_id": result.order,
                            "ticket": result.deal,
                            "price": result.price,
                            "volume": result.volume,
                        }
                except Exception as e:
                    logger.error(f"Order attempt with filling mode {filling_mode} failed: {e}")
                    continue

            # If all filling modes failed, return last error
            if last_result:
                logger.error(f"Order failed (retcode {last_result.retcode}): {last_result.comment}")
                return {"success": False, "error": last_result.comment, "retcode": last_result.retcode}
            return {"success": False, "error": "All order attempts failed"}
        except Exception as e:
            err_msg = f"Order exception: {str(e)}"
            logger.error(err_msg)
            return {"success": False, "error": err_msg}

    def close_order(self, ticket: int) -> Dict:
        try:
            # Get position with timeout
            def get_pos():
                return mt5_api.positions_get(ticket=ticket)
            positions = self._call_mt5_with_timeout(get_pos, timeout=10)
            if not positions:
                return {"success": False, "error": "Position not found"}
            position = positions[0]
            close_type = mt5_api.ORDER_TYPE_SELL if position.type == mt5_api.ORDER_TYPE_BUY else mt5_api.ORDER_TYPE_BUY
            # Get tick with timeout
            def get_tick():
                return mt5_api.symbol_info_tick(position.symbol)
            tick = self._call_mt5_with_timeout(get_tick, timeout=10)
            if not tick:
                return {"success": False, "error": "Could not get tick data"}
            price = tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask
            close_request = {
                "action": mt5_api.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": close_type,
                "position": ticket,
                "price": price,
                "deviation": 20,
                "magic": position.magic,
                "comment": f"{APP_NAME} Close",
                "type_time": mt5_api.ORDER_TIME_GTC,
                "type_filling": mt5_api.ORDER_FILLING_IOC,
            }
            # Send order with timeout
            def send_close():
                return mt5_api.order_send(close_request)
            result = self._call_mt5_with_timeout(send_close, timeout=15)
            if result.retcode == mt5_api.TRADE_RETCODE_DONE:
                return {"success": True, "ticket": result.deal, "pnl_usd": position.profit}
            return {"success": False, "error": result.comment}
        except Exception as e:
            logger.error(f"Close order error: {e}")
            return {"success": False, "error": str(e)}

    def get_positions(self, symbol: str = "") -> List[Dict]:
        try:
            def get_pos():
                return mt5_api.positions_get(symbol=symbol) if symbol else mt5_api.positions_get()
            positions = self._call_mt5_with_timeout(get_pos, timeout=10)
            return [p._asdict() for p in positions] if positions else []
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []

    def get_deal_history(self, limit: int = 100) -> List[Dict]:
        try:
            from datetime import timedelta
            def get_deals():
                return mt5_api.history_deals_get(datetime.now() - timedelta(days=30), datetime.now())
            deals = self._call_mt5_with_timeout(get_deals, timeout=15)
            if not deals:
                return []
            result = []
            for d in deals[-limit:]:
                dd = d._asdict()
                result.append({
                    "order_id": dd.get("position_id", dd.get("ticket")),
                    "ticket": dd.get("ticket"),
                    "position_id": dd.get("position_id"),
                    "symbol": dd.get("symbol"),
                    "pnl_usd": dd.get("profit", 0),
                    "profit": dd.get("profit", 0),
                    "volume": dd.get("volume"),
                    "price": dd.get("price"),
                    "time": dd.get("time"),
                    "reason": dd.get("reason"),
                    "comment": dd.get("comment", ""),
                    "entry": dd.get("entry"),  # DEAL_ENTRY_IN=0, DEAL_ENTRY_OUT=1
                    "type": dd.get("type"),
                })
            return result
        except Exception as e:
            logger.error(f"Error getting deal history: {e}")
            return []

    def get_close_deal_for_position(self, position_id: int) -> Dict:
        """Find the closing (OUT) deal for a position ticket."""
        try:
            from datetime import timedelta
            def get_deals():
                return mt5_api.history_deals_get(datetime.now() - timedelta(days=30), datetime.now())
            deals = self._call_mt5_with_timeout(get_deals, timeout=15)
            if not deals:
                return {}
            # Prefer OUT deals matching position_id
            out_deals = [
                d._asdict() for d in deals
                if d._asdict().get("position_id") == position_id and d._asdict().get("entry") == 1
            ]
            if out_deals:
                dd = out_deals[-1]
                return {
                    "order_id": dd.get("position_id", dd.get("ticket")),
                    "pnl_usd": dd.get("profit", 0),
                    "profit": dd.get("profit", 0),
                    "price": dd.get("price"),
                    "reason": dd.get("reason"),
                    "comment": dd.get("comment", ""),
                }
            # Fallback: any deal with matching position_id
            for d in reversed(deals):
                dd = d._asdict()
                if dd.get("position_id") == position_id:
                    return {
                        "order_id": dd.get("position_id"),
                        "pnl_usd": dd.get("profit", 0),
                        "profit": dd.get("profit", 0),
                        "price": dd.get("price"),
                        "reason": dd.get("reason"),
                        "comment": dd.get("comment", ""),
                    }
            return {}
        except Exception as e:
            logger.error(f"Error finding close deal for {position_id}: {e}")
            return {}

    def get_market_price(self, symbol: str) -> Dict:
        try:
            def get_tick():
                return mt5_api.symbol_info_tick(symbol)
            tick = self._call_mt5_with_timeout(get_tick, timeout=10)
            pip = get_pip_size(symbol)
            return {
                "symbol": symbol,
                "bid": tick.bid,
                "ask": tick.ask,
                "time": tick.time,
                "spread": (tick.ask - tick.bid) / pip if pip else 0,
            }
        except Exception as e:
            logger.error(f"Error getting market price for {symbol}: {e}")
            return {}

    def get_all_symbols(self) -> List[str]:
        try:
            def get_symbols():
                return mt5_api.symbols_get()
            symbols = self._call_mt5_with_timeout(get_symbols, timeout=10)
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
            "1min": mt5_api.TIMEFRAME_M1, "5min": mt5_api.TIMEFRAME_M5,
            "15min": mt5_api.TIMEFRAME_M15, "30min": mt5_api.TIMEFRAME_M30,
            "1h": mt5_api.TIMEFRAME_H1, "4h": mt5_api.TIMEFRAME_H4, "1d": mt5_api.TIMEFRAME_D1,
        }

    def _fmt_sync_error(self, symbol: str, context: str = "history") -> str:
        return (
            f"MT5 couldn't sync price {context} for {symbol} — try refreshing the symbol in MT5 "
            "(File → Open Data Folder → clear the Bases cache for this server, "
            "or right-click the symbol in Market Watch → Refresh) and reconnect."
        )

    def _get_last_error_details(self) -> Dict:
        try:
            err = mt5_api.last_error()
            if isinstance(err, tuple):
                return {
                    "code": err[0] if len(err) > 0 else None,
                    "message": err[1] if len(err) > 1 else "",
                    "raw": err,
                }
        except Exception as e:
            return {"code": None, "message": f"last_error unavailable: {e}", "raw": None}
        return {"code": None, "message": "", "raw": None}

    def ensure_symbol_selected(self, symbol: str) -> bool:
        try:
            def select_symbol() -> Any:
                return mt5_api.symbol_select(symbol, True)
            selected = bool(self._call_mt5_with_timeout(select_symbol, timeout=10))
            info = self.get_symbol_info(symbol)
            visible = bool(info.get("visible", False)) if info else False
            if not selected or not info or not visible:
                err = self._get_last_error_details()
                logger.warning(
                    f"Symbol selection check failed for {symbol}: selected={selected} visible={visible} info_exists={bool(info)} | last_error={err['raw']}"
                )
                if err["raw"] and err["raw"][0] == -10004:
                    if self.reconnect():
                        logger.warning(f"Reconnected — retrying symbol selection for {symbol}")
                        selected = bool(self._call_mt5_with_timeout(select_symbol, timeout=10))
                        info = self.get_symbol_info(symbol)
                        visible = bool(info.get("visible", False)) if info else False
                        return bool(selected and info and visible)
                return False
            return True
        except Exception as e:
            logger.error(f"Error selecting symbol {symbol}: {e}")
            logger.error(f"Symbol selection last_error {symbol}: {self._get_last_error_details()}")
            return False

    def get_historical_data(self, symbol: str, timeframe: str, count: int = 1000) -> Optional[pd.DataFrame]:
        try:
            if not self.ensure_symbol_selected(symbol):
                logger.error(f"Historical fetch aborted — symbol not selected/visible: {symbol}")
                return None
            mt5_tf = self._tf_map().get(timeframe, mt5_api.TIMEFRAME_M5)
            last_err = None
            for attempt in range(3):
                def get_rates():
                    return mt5_api.copy_rates_from_pos(symbol, mt5_tf, 0, count)
                rates = self._call_mt5_with_timeout(get_rates, timeout=15)
                if rates is not None and len(rates) > 0:
                    df = pd.DataFrame(rates)
                    df["time"] = pd.to_datetime(df["time"], unit="s")
                    df.set_index("time", inplace=True)
                    df.columns = [c.lower() for c in df.columns]
                    return df
                last_err = self._get_last_error_details()
                logger.warning(
                    f"Historical fetch failure {symbol} {timeframe} attempt {attempt + 1}/3 | last_error={last_err['raw']}"
                )
                if last_err.get("code") in (1, 2, 5, 6):
                    logger.error(self._fmt_sync_error(symbol))
                    return None
                if attempt < 2:
                    time.sleep(0.5)
            logger.error(f"Historical data sync failed for {symbol} {timeframe}: {last_err}")
            logger.error("Try: Right-click symbol in MT5 Market Watch -> Refresh, or clear MT5 history cache")
            return None
        except Exception as e:
            logger.error(f"Historical data error {symbol} {timeframe}: {e}")
            logger.error(f"Historical data last_error {symbol} {timeframe}: {self._get_last_error_details()}")
            logger.error("Try: Right-click symbol in MT5 Market Watch -> Refresh, or clear MT5 history cache")
            return None

    def get_historical_data_before(self, symbol: str, timeframe: str, before_dt: datetime, count: int = 5000) -> Optional[pd.DataFrame]:
        try:
            if not self.ensure_symbol_selected(symbol):
                logger.error(f"Paged historical fetch aborted — symbol not selected/visible: {symbol}")
                return None
            mt5_tf = self._tf_map().get(timeframe, mt5_api.TIMEFRAME_M5)
            last_err = None
            for attempt in range(3):
                def get_rates():
                    return mt5_api.copy_rates_from(symbol, mt5_tf, before_dt, count)
                rates = self._call_mt5_with_timeout(get_rates, timeout=15)
                if rates is not None and len(rates) > 0:
                    df = pd.DataFrame(rates)
                    df["time"] = pd.to_datetime(df["time"], unit="s")
                    df.set_index("time", inplace=True)
                    df.columns = [c.lower() for c in df.columns]
                    return df
                last_err = self._get_last_error_details()
                logger.warning(
                    f"Paged history fetch failure {symbol} {timeframe} before={before_dt} count={count} attempt {attempt + 1}/3 | last_error={last_err['raw']}"
                )
                if last_err.get("code") in (1, 2, 5, 6):
                    logger.error(self._fmt_sync_error(symbol))
                    return None
                if attempt < 2:
                    time.sleep(0.5)
            logger.error(f"Paged history sync failed for {symbol} {timeframe}: {last_err}")
            logger.error("Try: Right-click symbol in MT5 Market Watch -> Refresh, or clear MT5 history cache")
            return None
        except Exception as e:
            logger.error(f"Paged history error {symbol} {timeframe}: {e}")
            logger.error(f"Paged history last_error {symbol} {timeframe}: {self._get_last_error_details()}")
            logger.error("Try: Right-click symbol in MT5 Market Watch -> Refresh, or clear MT5 history cache")
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
        try:
            mt5_api.shutdown()
        except Exception:
            pass
        self.connected = False
        logger.info("Disconnected from MT5")


mt5_bridge = MT5Bridge()
