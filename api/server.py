"""FastAPI backend for AERO TRADE."""

import asyncio
import json
from typing import Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path

from config.settings import APP_NAME, TRADING_CONFIG, BROKER_CONFIG
from core.broker.broker_registry import SUPPORTED_BROKERS, broker_to_dict, get_broker
from core.broker.credential_store import (
    create_account, list_accounts, get_account_meta, update_account_meta,
    load_broker_credentials, delete_account,
)
from core.broker.mt5_broker_client import MT5BrokerClient
from core.engine.aero_trade_bot import AeroTradeBot
from core.ledger.trade_ledger import get_closed_trades, get_trade_summary, init_ledger

# Active bot instances per account
_bots: Dict[str, AeroTradeBot] = {}
_bot_tasks: Dict[str, asyncio.Task] = {}


class CreateAccountRequest(BaseModel):
    broker_id: str
    login: int
    password: str
    is_demo: bool = True
    custom_server: str = ""


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

assets_path = Path("assets")
if assets_path.exists():
    app.mount("/assets", StaticFiles(directory="assets"), name="assets")

ui_dist = Path("ui/dist")
if ui_dist.exists():
    app.mount("/", StaticFiles(directory="ui/dist", html=True), name="ui")


@app.get("/api/health")
def health():
    return {"status": "ok", "app": APP_NAME}


@app.get("/brokers")
def list_brokers():
    return [broker_to_dict(b) for b in SUPPORTED_BROKERS]


@app.post("/accounts")
def create_account_endpoint(req: CreateAccountRequest):
    broker = get_broker(req.broker_id)
    server = req.custom_server or (broker.demo_server if req.is_demo else broker.real_server)
    if not server:
        raise HTTPException(400, "Server name required for custom broker")

    ok, msg, info = MT5BrokerClient.verify_credentials(req.login, req.password, server, req.is_demo)
    if not ok:
        raise HTTPException(401, msg)

    account_id = create_account(req.broker_id, req.login, req.password, server, req.is_demo, broker.display_name)
    BROKER_CONFIG["name"] = broker.display_name
    return {
        "account_id": account_id,
        "balance": info.get("balance", 0),
        "equity": info.get("equity", 0),
        "currency": info.get("currency", "USD"),
        "leverage": info.get("leverage", 100),
        "server": server,
        "is_demo": req.is_demo,
        "broker": broker.display_name,
    }


@app.get("/accounts")
def get_accounts():
    return list_accounts()


@app.get("/accounts/{account_id}")
async def get_account(account_id: str):
    try:
        creds = load_broker_credentials(account_id)
        meta = get_account_meta(account_id)
        client = MT5BrokerClient(account_id)
        await client.connect_with_credentials(creds)
        info = await client.get_account_info()
        await client.disconnect()
        positions = []
        if client.mt5_bridge:
            positions = client.mt5_bridge.get_positions()
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
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/accounts/{account_id}/positions")
async def get_positions(account_id: str):
    try:
        creds = load_broker_credentials(account_id)
        client = MT5BrokerClient(account_id)
        await client.connect_with_credentials(creds)
        orders = await client.get_open_orders()
        await client.disconnect()
        return orders
    except KeyError:
        raise HTTPException(404, "Account not found")


@app.get("/accounts/{account_id}/trades")
def get_trades(account_id: str, limit: int = 100):
    trades = get_closed_trades(account_id, limit)
    summary = get_trade_summary(account_id)
    return {"trades": trades, "summary": summary}


@app.post("/accounts/{account_id}/start")
async def start_trading(account_id: str):
    if account_id in _bot_tasks and not _bot_tasks[account_id].done():
        return {"status": "already_running"}
    try:
        creds = load_broker_credentials(account_id)
        creds["broker_id"] = get_account_meta(account_id).get("broker_id", "custom")
        bot = AeroTradeBot(account_id)
        await bot.initialize(creds)
        _bots[account_id] = bot
        _bot_tasks[account_id] = asyncio.create_task(bot.run())
        return {"status": "started"}
    except Exception as e:
        raise HTTPException(500, str(e))


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


def run_server(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run("api.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run_server()
