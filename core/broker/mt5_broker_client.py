"""
Generic MT5 Broker Client for AERO TRADE.
Works with any broker offering MT5 login.
"""

import asyncio
import time
import random
from typing import Dict, List, Optional
from datetime import datetime

from utils.logger import logger
from config.settings import BROKER_CONFIG, TRADING_CONFIG, APP_NAME
from core.broker.mt5_bridge import MT5Bridge
from core.broker.broker_registry import BrokerPreset, get_broker


class MT5BrokerClient:
    """MT5-based broker client — broker-agnostic."""

    def __init__(self, account_id: str = "default"):
        self.account_id = account_id
        self.connected = False
        self.account_info = {}
        self.orders = []
        self.use_mt5 = False
        self.mt5_bridge: Optional[MT5Bridge] = None
        self.broker_preset: Optional[BrokerPreset] = None
        self.is_demo = True
        self.demo_balance = 10000.0
        logger.info(f"{APP_NAME} MT5 Broker Client initialized")

    async def connect(
        self,
        login: int = None,
        password: str = None,
        server: str = None,
        demo: bool = True,
        broker_id: str = "custom",
    ) -> bool:
        try:
            self.is_demo = demo
            if broker_id != "custom":
                self.broker_preset = get_broker(broker_id)
                BROKER_CONFIG["name"] = self.broker_preset.display_name

            if login and password and server:
                return await self._connect_mt5(login, password, server, demo)
            return await self._connect_mock(demo)
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
        )

    async def _connect_mt5(self, login: int, password: str, server: str, demo: bool) -> bool:
        try:
            import MetaTrader5  # noqa: F401
            self.mt5_bridge = MT5Bridge()
            success = self.mt5_bridge.connect(login=login, password=password, server=server, demo=demo)
            if success:
                self.use_mt5 = True
                self.account_info = self.mt5_bridge.get_account_info()
                self.connected = True
                account_type = "DEMO" if demo else "REAL"
                broker_name = BROKER_CONFIG.get("name") or server
                logger.success(f"Connected to {broker_name} {account_type} Account #{login}")
                return True
            return await self._connect_mock(demo)
        except ImportError:
            logger.warning("MetaTrader5 not installed — using mock mode")
            return await self._connect_mock(demo)
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            return await self._connect_mock(demo)

    async def _connect_mock(self, demo: bool) -> bool:
        await asyncio.sleep(0.5)
        self.account_info = {
            "login": 0,
            "balance": self.demo_balance,
            "equity": self.demo_balance,
            "margin": 0.0,
            "free_margin": self.demo_balance,
            "leverage": BROKER_CONFIG.get("leverage", 100),
            "currency": TRADING_CONFIG.get("account_currency", "USD"),
            "name": "Demo Account (Mock)",
            "server": "Mock-Server",
            "profit": 0.0,
        }
        self.connected = True
        self.is_demo = demo
        logger.warning(f"Using MOCK connection: Balance ${self.demo_balance:,.2f}")
        return True

    @staticmethod
    def verify_credentials(login: int, password: str, server: str, demo: bool = True, timeout: int = 15) -> tuple:
        """Synchronous credential verification with timeout."""
        bridge = MT5Bridge()
        try:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(bridge.connect, login, password, server, demo)
                success = future.result(timeout=timeout)
            if not success:
                return False, "Login failed — check account number, password, and server name", {}
            info = bridge.get_account_info()
            bridge.disconnect()
            return True, "Connected", info
        except concurrent.futures.TimeoutError:
            bridge.disconnect()
            return False, "Connection timed out — is MT5 terminal running?", {}
        except Exception as e:
            bridge.disconnect()
            return False, str(e), {}

    async def get_account_info(self) -> Dict:
        if not self.connected:
            await self.connect(demo=True)
        if self.use_mt5 and self.mt5_bridge:
            self.account_info = self.mt5_bridge.get_account_info()
        return self.account_info.copy()

    async def place_order(self, symbol: str, order_type: str, volume: float, **kwargs) -> Dict:
        try:
            if not self.connected:
                await self.connect(demo=True)
            if volume <= 0:
                return {"success": False, "error": "Volume must be positive", "order_id": None}

            if self.use_mt5 and self.mt5_bridge:
                result = self.mt5_bridge.place_order(
                    symbol=symbol, order_type=order_type, volume=volume, **kwargs
                )
                if result.get("success"):
                    result["symbol"] = symbol
                    result["direction"] = order_type
                return result

            order_id = int(time.time() * 1000)
            order = {
                "order_id": order_id, "symbol": symbol, "type": order_type,
                "volume": volume, "entry_price": 1.0, "status": "open",
                "time": datetime.now().isoformat(),
                "comment": kwargs.get("comment", APP_NAME),
            }
            self.orders.append(order)
            return {"success": True, "order_id": order_id, "price": 1.0, "volume": volume, "symbol": symbol}
        except Exception as e:
            return {"success": False, "error": str(e), "order_id": None}

    async def close_order(self, order_id: int, **kwargs) -> Dict:
        if self.use_mt5 and self.mt5_bridge:
            result = self.mt5_bridge.close_order(order_id)
            if result.get("success"):
                positions = self.mt5_bridge.get_positions()
                # P&L from last deal — simplified
                result["pnl_usd"] = kwargs.get("pnl_usd", 0)
            return result
        order = next((o for o in self.orders if o["order_id"] == order_id), None)
        if not order:
            return {"success": False, "error": f"Order {order_id} not found"}
        pnl_usd = random.uniform(-50, 50)
        order["status"] = "closed"
        order["pnl_usd"] = pnl_usd
        self.orders = [o for o in self.orders if o["order_id"] != order_id]
        return {"success": True, "order_id": order_id, "pnl_usd": pnl_usd}

    async def get_open_orders(self) -> List[Dict]:
        if self.use_mt5 and self.mt5_bridge:
            positions = self.mt5_bridge.get_positions()
            return [{
                "order_id": p.get("ticket", 0),
                "symbol": p.get("symbol", ""),
                "type": "BUY" if p.get("type", 0) == 0 else "SELL",
                "volume": p.get("volume", 0),
                "entry_price": p.get("price_open", 0),
                "profit": p.get("profit", 0),
                "status": "open",
            } for p in positions]
        return [o for o in self.orders if o.get("status") == "open"]

    async def get_order_history(self, limit: int = 100) -> List[Dict]:
        if self.use_mt5 and self.mt5_bridge:
            return self.mt5_bridge.get_deal_history(limit=limit)
        return self.orders[-limit:]

    async def get_market_price(self, symbol: str) -> Dict:
        if self.use_mt5 and self.mt5_bridge:
            return self.mt5_bridge.get_market_price(symbol)
        return {"symbol": symbol, "bid": 1.1, "ask": 1.10015, "spread": 1.5}

    async def disconnect(self):
        if self.use_mt5 and self.mt5_bridge:
            self.mt5_bridge.disconnect()
        self.connected = False
        logger.info(f"Disconnected from {APP_NAME} broker")


# Backward-compatible alias
FBSClient = MT5BrokerClient
fbs_client = MT5BrokerClient()
