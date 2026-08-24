"""
Generic MT5 Broker Client for AERO TRADE.
Works with any broker offering MT5 login.
"""

import asyncio
import concurrent.futures
import time
from typing import Dict, List, Optional
from datetime import datetime

from utils.logger import logger
from config.settings import BROKER_CONFIG, TRADING_CONFIG, APP_NAME
from core.broker.mt5_bridge import MT5Bridge, mt5_bridge
from core.broker.broker_registry import BrokerPreset, get_broker


class MT5BrokerClient:
    """MT5-based broker client — broker-agnostic."""

    def __init__(self, account_id: str = "default"):
        self.account_id = account_id
        self.connected = False
        self.account_info = {}
        self.orders = []
        self.use_mt5 = False
        self.mt5_bridge: Optional[MT5Bridge] = mt5_bridge
        self.broker_preset: Optional[BrokerPreset] = None
        self.is_demo = True
        self.demo_balance = 10000.0
        logger.info(f"{APP_NAME} MT5 Broker Client initialized")

    async def connect(
        self,
        login: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
        demo: bool = True,
        broker_id: str = "custom",
        mt5_path: Optional[str] = None,
    ) -> bool:
        try:
            self.is_demo = demo
            if broker_id != "custom":
                self.broker_preset = get_broker(broker_id)
                BROKER_CONFIG["name"] = self.broker_preset.display_name

            if login is None or password is None or server is None:
                logger.error("MT5 connection failed: missing login credentials")
                return False

            return await self._connect_mt5(login, password, server, demo, mt5_path)
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    async def connect_with_credentials(self, creds: dict) -> bool:
        return await self.connect(
            login=creds["login"],
            password=creds["password"],
            server=creds["server"],
            demo=creds.get("demo", True),
            broker_id=creds.get("broker_id", "custom"),
            mt5_path=creds.get("mt5_path"),
        )

    async def _connect_mt5(self, login: int, password: str, server: str, demo: bool, path: Optional[str] = None) -> bool:
        try:
            loop = asyncio.get_event_loop()
            try:
                success = await asyncio.wait_for(
                    loop.run_in_executor(None, self.mt5_bridge.connect, login, password, server, demo, path),
                    timeout=15,
                )
            except asyncio.TimeoutError:
                logger.error("MT5 connect timed out — terminal may be stuck on history sync for a broken symbol")
                logger.error("Try: Right-click symbol in MT5 Market Watch -> Refresh, or clear MT5 history cache")
                try:
                    self.mt5_bridge.disconnect()
                except Exception:
                    pass
                return False
            if success:
                self.use_mt5 = True
                self.account_info = self.mt5_bridge.get_account_info()
                self.connected = True
                account_type = "DEMO" if demo else "REAL"
                broker_name = BROKER_CONFIG.get("name") or server
                logger.success(f"Connected to {broker_name} {account_type} Account #{login}")
                return True
            self.connected = False
            logger.error(f"MT5 connection failed for login #{login} on {server} — bot will NOT start with fake/mock data")
            return False
        except ImportError:
            self.connected = False
            logger.error("MetaTrader5 package is not installed — cannot connect to a real broker, refusing to start")
            return False
        except Exception as e:
            self.connected = False
            logger.error(f"MT5 connection error: {e}")
            return False

    @staticmethod
    def verify_credentials(login: int, password: str, server: str, demo: bool = True, timeout: int = 15, path: Optional[str] = None) -> tuple:
        """Synchronous credential verification with timeout."""
        bridge = MT5Bridge()
        try:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(bridge.connect, login, password, server, demo, path)
                success = future.result(timeout=timeout)
            if not success:
                return False, "Login failed — check account number, password, and server name", {}
            info = bridge.get_account_info()
            bridge.disconnect()
            return True, "Connected", info
        except concurrent.futures.TimeoutError:
            bridge.disconnect()
            return False, "Connection timed out — MT5 terminal may be stuck syncing history for a broken symbol. Try restarting MT5 and reconnecting.", {}
        except Exception as e:
            bridge.disconnect()
            return False, str(e), {}

    async def get_account_info(self) -> Dict:
        if not self.connected or not self.mt5_bridge:
            logger.error("Cannot get account info: no active MT5 connection")
            return {}
        self.account_info = self.mt5_bridge.get_account_info()
        return self.account_info.copy()

    async def place_order(self, symbol: str, order_type: str, volume: float, **kwargs) -> Dict:
        try:
            if not self.connected or not self.mt5_bridge:
                return {"success": False, "error": "No live MT5 connection", "order_id": None}
            if volume <= 0:
                return {"success": False, "error": "Volume must be positive", "order_id": None}

            result = self.mt5_bridge.place_order(symbol=symbol, order_type=order_type, volume=volume, **kwargs)
            if result.get("success"):
                result["symbol"] = symbol
                result["direction"] = order_type
            return result
        except Exception as e:
            return {"success": False, "error": str(e), "order_id": None}

    async def close_order(self, order_id: int, **kwargs) -> Dict:
        if not self.connected or not self.mt5_bridge:
            return {"success": False, "error": "No live MT5 connection"}
        result = self.mt5_bridge.close_order(order_id)
        if result.get("success"):
            self.mt5_bridge.get_positions()
            result["pnl_usd"] = kwargs.get("pnl_usd", 0)
        return result

    async def get_open_orders(self) -> Optional[List[Dict]]:
        """Returns None when the broker query itself failed — callers must
        treat that as "unknown", not as "confirmed zero positions"."""
        if not self.connected or not self.mt5_bridge:
            logger.error("Cannot get open orders: no live MT5 connection")
            return None
        positions = self.mt5_bridge.get_positions()
        if positions is None:
            return None
        return [{
            "order_id": p.get("ticket", 0),
            "symbol": p.get("symbol", ""),
            "type": "BUY" if p.get("type", 0) == 0 else "SELL",
            "volume": p.get("volume", 0),
            "entry_price": p.get("price_open", 0),
            "profit": p.get("profit", 0),
            "pnl_usd": p.get("profit", 0),
            "stop_loss": p.get("sl", 0),
            "take_profit": p.get("tp", 0),
            "status": "open",
        } for p in positions]

    async def get_close_deal(self, position_id: int) -> Dict:
        if not self.connected or not self.mt5_bridge:
            logger.error("Cannot get close deal: no live MT5 connection")
            return {}
        return self.mt5_bridge.get_close_deal_for_position(position_id)

    async def get_order_history(self, limit: int = 100) -> List[Dict]:
        if not self.connected or not self.mt5_bridge:
            logger.error("Cannot get order history: no live MT5 connection")
            return []
        return self.mt5_bridge.get_deal_history(limit=limit)

    async def get_market_price(self, symbol: str) -> Dict:
        if not self.connected or not self.mt5_bridge:
            logger.error("Cannot get market price: no live MT5 connection")
            return {}
        return self.mt5_bridge.get_market_price(symbol)

    async def modify_position(self, order_id: int, stop_loss: float = 0.0, take_profit: float = 0.0, **kwargs) -> Dict:
        if not self.connected or not self.mt5_bridge:
            return {"success": False, "error": "No live MT5 connection"}
        sl = kwargs.get("sl", stop_loss)
        tp = kwargs.get("tp", take_profit)
        return self.mt5_bridge.modify_position_sl_tp(int(order_id), sl=float(sl or 0.0), tp=float(tp or 0.0))

    async def close_partial(self, order_id: int, volume: float, **kwargs) -> Dict:
        if not self.connected or not self.mt5_bridge:
            return {"success": False, "error": "No live MT5 connection"}
        if volume <= 0:
            return {"success": False, "error": "Volume must be positive for partial close"}
        return self.mt5_bridge.close_partial_position(int(order_id), float(volume))

    async def disconnect(self):
        if self.use_mt5 and self.mt5_bridge:
            self.mt5_bridge.disconnect()
        self.connected = False
        logger.info(f"Disconnected from {APP_NAME} broker")


# Backward-compatible alias
FBSClient = MT5BrokerClient
fbs_client = MT5BrokerClient()
