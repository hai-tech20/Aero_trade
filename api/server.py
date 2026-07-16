"""FastAPI backend for AERO TRADE."""

import asyncio
import threading
from typing import Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path

from config.settings import APP_NAME, TRADING_CONFIG, BROKER_CONFIG
from core.broker.broker_registry import SUPPORTED_BROKERS, broker_to_dict, get_broker, is_demo_server
from core.broker.credential_store import (
    create_account, list_accounts, get_account_meta, update_account_meta,
    load_broker_credentials, delete_account,
)
from core.broker.mt5_broker_client import MT5BrokerClient
from core.engine.aero_trade_bot import AeroTradeBot
from core.ledger.trade_ledger import get_closed_trades, get_trade_summary, init_ledger
from ai_engine.training.model_status import get_model_status
from utils.logger import logger

# Active bot instances per account
_bots: Dict[str, AeroTradeBot] = {}
_bot_tasks: Dict[str, asyncio.Task[object]] = {}


class CreateAccountRequest(BaseModel):
    broker_id: str
    login: int
    password: str
    server: str = ""
    custom_server: str = ""
    is_demo_confirmed: Optional[bool] = None
    mt5_path: Optional[str] = None


class LiveModeRequest(BaseModel):
    enable: bool
    password: str


class SettingsUpdate(BaseModel):
    risk_per_trade: Optional[float] = None
    enabled_pairs: Optional[list] = None
    aggressive_mode: Optional[bool] = None
    max_daily_loss_pct: Optional[float] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_ledger()
    yield
    for aid, task in list(_bot_tasks.items()):
        task.cancel()
        if aid in _bots:
            _bots[aid].stop()


app = FastAPI(title=APP_NAME, version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok", "app": APP_NAME}


@app.get("/brokers")
def list_brokers():
    return [broker_to_dict(b) for b in SUPPORTED_BROKERS]


@app.post("/accounts")
def create_account_endpoint(req: CreateAccountRequest):
    broker = get_broker(req.broker_id)
    server = (req.server or req.custom_server).strip()
    if not server:
        raise HTTPException(400, "Server name is required")

    resolved_demo = is_demo_server(server, broker)
    if resolved_demo is None:
        if req.is_demo_confirmed is None:
            raise HTTPException(400, {
                "code": "ambiguous_server_type",
                "message": "Server name does not clearly indicate Demo or Real. Confirm account type before proceeding.",
                "server": server,
            })
        resolved_demo = req.is_demo_confirmed

    ok, msg, info = MT5BrokerClient.verify_credentials(req.login, req.password, server, resolved_demo, path=req.mt5_path or "")
    if not ok:
        raise HTTPException(401, msg)

    account_id = create_account(req.broker_id, req.login, req.password, server, resolved_demo, broker.display_name, req.mt5_path or "")
    BROKER_CONFIG["name"] = broker.display_name
    return {
        "account_id": account_id,
        "balance": info.get("balance", 0),
        "equity": info.get("equity", 0),
        "currency": info.get("currency", "USD"),
        "leverage": info.get("leverage", 100),
        "server": server,
        "is_demo": resolved_demo,
        "broker": broker.display_name,
    }


@app.get("/accounts")
def get_accounts():
    return list_accounts()


@app.get("/accounts/{account_id}")
async def get_account(account_id: str):
    try:
        meta = get_account_meta(account_id)
        
        if account_id in _bots:
            bot = _bots[account_id]
            info = bot.account_info or {}
            positions = []
            if bot.broker and bot.broker.connected:
                positions = await bot.broker.get_open_orders()
            return {
                **info,
                "account_id": account_id,
                "is_demo": meta.get("is_demo", True),
                "live_trading_enabled": meta.get("live_trading_enabled", False),
                "open_positions_count": len(positions),
                "broker": meta.get("broker_id", ""),
            }
        
        creds = load_broker_credentials(account_id)
        client = MT5BrokerClient(account_id)
        try:
            await asyncio.wait_for(client.connect_with_credentials(creds), timeout=10)
        except asyncio.TimeoutError:
            raise HTTPException(504, "MT5 connection timed out — terminal may be stuck on history sync. Try restarting MT5 and reconnecting.")
        if not client.connected:
            raise HTTPException(502, "Failed to connect to MT5 terminal")
        info = await client.get_account_info()
        positions = []
        if client.mt5_bridge and client.connected:
            positions = client.mt5_bridge.get_positions()
        await client.disconnect()
        return {
            **info,
            "account_id": account_id,
            "is_demo": meta.get("is_demo", True),
            "live_trading_enabled": meta.get("live_trading_enabled", False),
            "open_positions_count": len(positions),
            "broker": meta.get("broker_id", ""),
        }
    except KeyError:
        raise HTTPException(404, "Account not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/accounts/{account_id}/positions")
async def get_positions(account_id: str):
    try:
        if account_id in _bots:
            bot = _bots[account_id]
            orders = []
            if bot.broker and bot.broker.connected:
                orders = await bot.broker.get_open_orders()
            return orders
        
        creds = load_broker_credentials(account_id)
        client = MT5BrokerClient(account_id)
        try:
            await asyncio.wait_for(client.connect_with_credentials(creds), timeout=10)
        except asyncio.TimeoutError:
            raise HTTPException(504, "MT5 connection timed out — terminal may be stuck on history sync. Try restarting MT5 and reconnecting.")
        if not client.connected:
            raise HTTPException(502, "Failed to connect to MT5 terminal")
        orders = await client.get_open_orders()
        await client.disconnect()
        return orders
    except KeyError:
        raise HTTPException(404, "Account not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/accounts/{account_id}/trades")
def get_trades(account_id: str, limit: int = 100):
    trades = get_closed_trades(account_id, limit)
    summary = get_trade_summary(account_id)
    return {"trades": trades, "summary": summary}


@app.get("/accounts/{account_id}/activity")
def get_activity(account_id: str, limit: int = 50):
    if account_id not in _bots:
        return {"activity": []}
    return {"activity": _bots[account_id].get_activity(limit=limit)}


@app.get("/models/status")
def model_status():
    return {"models": get_model_status()}


@app.get("/debug/mt5-test")
def debug_mt5_test():
    try:
        from core.broker.credential_store import load_broker_credentials
        creds = load_broker_credentials('f806d8ec-84ea-4185-8657-a0f13f07a09c')
        from core.broker.mt5_broker_client import MT5BrokerClient
        client = MT5BrokerClient('debug')
        import asyncio
        loop = asyncio.get_event_loop()
        loop.run_until_complete(client.connect_with_credentials(creds))
        result: dict[str, object] = {
            "connected": client.connected,
            "use_mt5": client.use_mt5,
            "mt5_bridge": client.mt5_bridge is not None,
        }
        if client.mt5_bridge:
            df = client.mt5_bridge.get_historical_data("EURUSD", "1min", count=10)
            result["historical_data"] = df is not None
            result["bars_count"] = len(df) if df is not None else 0
        return result
    except Exception as e:
        return {"error": str(e)}


_bot_threads: Dict[str, threading.Thread] = {}
_bot_loops: Dict[str, asyncio.AbstractEventLoop] = {}


def _run_bot_in_thread(account_id: str, creds: dict):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _bot_loops[account_id] = loop

    async def _bot_main():
        try:
            bot = AeroTradeBot(account_id)
            await bot.initialize(creds)
            _bots[account_id] = bot
            _bot_tasks[account_id] = asyncio.create_task(bot.run())
        except Exception as e:
            logger.error(f"Failed to start bot for {account_id}: {e}")

    loop.run_until_complete(_bot_main())
    loop.close()


@app.post("/accounts/{account_id}/start")
async def start_trading(account_id: str):
    if account_id in _bot_tasks and not _bot_tasks[account_id].done():
        return {"status": "already_running"}

    if account_id in _bot_threads and _bot_threads[account_id].is_alive():
        return {"status": "already_running"}

    creds = load_broker_credentials(account_id)
    creds["broker_id"] = get_account_meta(account_id).get("broker_id", "custom")

    thread = threading.Thread(
        target=_run_bot_in_thread,
        args=(account_id, creds),
        daemon=True,
    )
    thread.start()
    _bot_threads[account_id] = thread
    return {"status": "started"}


@app.post("/accounts/{account_id}/stop")
async def stop_trading(account_id: str):
    if account_id in _bots:
        _bots[account_id].stop()
    if account_id in _bot_tasks:
        _bot_tasks[account_id].cancel()
        del _bot_tasks[account_id]
    return {"status": "stopped"}


@app.post("/accounts/{account_id}/live-mode")
def enable_live_mode(account_id: str, req: LiveModeRequest):
    try:
        meta = get_account_meta(account_id)
        creds = load_broker_credentials(account_id)
        if creds["password"] != req.password:
            raise HTTPException(401, "Password confirmation failed")
        if meta.get("is_demo"):
            raise HTTPException(400, "Live mode only applies to real accounts")
        if req.enable:
            update_account_meta(account_id, {
                "live_trading_enabled": True,
                "aggressive_mode": False,
            })
            TRADING_CONFIG["aggressive_mode"] = False
            return {"status": "live_trading_enabled", "warning": "Real money is at risk"}
        update_account_meta(account_id, {"live_trading_enabled": False})
        return {"status": "live_trading_disabled"}
    except KeyError:
        raise HTTPException(404, "Account not found")


@app.get("/accounts/{account_id}/settings")
def get_settings(account_id: str):
    try:
        meta = get_account_meta(account_id)
        return {
            "risk_per_trade": TRADING_CONFIG.get("risk_per_trade"),
            "enabled_pairs": meta.get("enabled_pairs") or TRADING_CONFIG.get("pairs"),
            "aggressive_mode": meta.get("aggressive_mode", False),
            "max_daily_loss_pct": meta.get("settings", {}).get("max_daily_loss_pct", 5.0),
            "is_demo": meta.get("is_demo", True),
            "live_trading_enabled": meta.get("live_trading_enabled", False),
        }
    except KeyError:
        raise HTTPException(404, "Account not found")


@app.put("/accounts/{account_id}/settings")
def update_settings(account_id: str, req: SettingsUpdate):
    try:
        meta = get_account_meta(account_id)
        updates = {}
        if req.enabled_pairs is not None:
            updates["enabled_pairs"] = req.enabled_pairs
        if req.aggressive_mode is not None:
            if req.aggressive_mode and not meta.get("is_demo"):
                raise HTTPException(400, "Aggressive mode only available on demo accounts")
            updates["aggressive_mode"] = req.aggressive_mode
            TRADING_CONFIG["aggressive_mode"] = req.aggressive_mode
        if req.risk_per_trade is not None:
            TRADING_CONFIG["risk_per_trade"] = req.risk_per_trade
        if req.max_daily_loss_pct is not None:
            updates.setdefault("settings", {})["max_daily_loss_pct"] = req.max_daily_loss_pct
        update_account_meta(account_id, updates)
        return {"status": "updated"}
    except KeyError:
        raise HTTPException(404, "Account not found")


@app.delete("/accounts/{account_id}")
def remove_account(account_id: str):
    if account_id in _bots:
        _bots[account_id].stop()
    delete_account(account_id)
    return {"status": "deleted"}


@app.websocket("/accounts/{account_id}/stream")
async def websocket_stream(websocket: WebSocket, account_id: str):
    await websocket.accept()
    try:
        while True:
            data = {}
            if account_id in _bots:
                bot = _bots[account_id]
                data = {
                    "balance": bot.account_info.get("balance", 0),
                    "equity": bot.account_info.get("equity", 0),
                    "positions": len(bot.active_positions),
                    "running": bot.running,
                }
            await websocket.send_json(data)
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass


# Mount static files AFTER API routes
ui_dist = Path("ui/dist")
if ui_dist.exists():
    app.mount("/", StaticFiles(directory="ui/dist", html=True), name="ui")


def run_server(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run("api.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run_server()
