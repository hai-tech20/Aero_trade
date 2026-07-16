"""Secure credential storage for AERO TRADE broker accounts."""

import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from utils.logger import logger

SERVICE_NAME = "AeroTrade"
ACCOUNTS_FILE = Path("data/accounts.json")

try:
    import keyring
    HAS_KEYRING = True
except ImportError:
    keyring = None
    HAS_KEYRING = False

try:
    from cryptography.fernet import Fernet
    HAS_CRYPTO = True
except ImportError:
    Fernet = None
    HAS_CRYPTO = False


def _get_fernet() -> Optional[Any]:
    if not HAS_CRYPTO or Fernet is None:
        return None
    key = os.getenv("AERO_TRADE_ENCRYPTION_KEY", "")
    if not key:
        return None
    fernet_cls = cast(Any, Fernet)
    return fernet_cls(key.encode() if isinstance(key, str) else key)


def _encrypt(payload: str) -> str:
    f = _get_fernet()
    if f:
        return f.encrypt(payload.encode()).decode()
    return payload


def _decrypt(payload: str) -> str:
    f = _get_fernet()
    if f:
        return f.decrypt(payload.encode()).decode()
    return payload


def _load_accounts_meta() -> Dict[str, Any]:
    if ACCOUNTS_FILE.exists():
        return json.loads(ACCOUNTS_FILE.read_text(encoding="utf-8"))
    return {"accounts": {}}


def _save_accounts_meta(data: Dict[str, Any]):
    ACCOUNTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    ACCOUNTS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_broker_credentials(
    account_id: str,
    login: int,
    password: str,
    server: str,
    demo: bool,
    broker_id: str = "custom",
    mt5_path: Optional[str] = None,
) -> None:
    payload = json.dumps({
        "login": login,
        "password": password,
        "server": server,
        "demo": demo,
        "broker_id": broker_id,
        "mt5_path": mt5_path,
    })
    if HAS_KEYRING and keyring is not None:
        keyring.set_password(SERVICE_NAME, account_id, payload)
    else:
        meta = _load_accounts_meta()
        meta["accounts"].setdefault(account_id, {})["encrypted"] = _encrypt(payload)
        _save_accounts_meta(meta)


def load_broker_credentials(account_id: str) -> dict:
    if HAS_KEYRING and keyring is not None:
        raw = keyring.get_password(SERVICE_NAME, account_id)
        if raw is None:
            raise KeyError(f"No stored credentials for {account_id}")
        return json.loads(raw)
    meta = _load_accounts_meta()
    acct = meta.get("accounts", {}).get(account_id)
    if not acct or "encrypted" not in acct:
        raise KeyError(f"No stored credentials for {account_id}")
    return json.loads(_decrypt(acct["encrypted"]))


def create_account(
    broker_id: str,
    login: int,
    password: str,
    server: str,
    is_demo: bool,
    display_name: str = "",
    mt5_path: Optional[str] = None,
) -> str:
    account_id = str(uuid.uuid4())
    save_broker_credentials(account_id, login, password, server, is_demo, broker_id, mt5_path)
    meta = _load_accounts_meta()
    meta["accounts"][account_id] = {
        "broker_id": broker_id,
        "login": login,
        "server": server,
        "is_demo": is_demo,
        "display_name": display_name or f"Account {login}",
        "live_trading_enabled": False,
        "aggressive_mode": False,
        "enabled_pairs": [],
        "settings": {},
    }
    _save_accounts_meta(meta)
    logger.info(f"Created account {account_id} for broker {broker_id}")
    return account_id


def list_accounts() -> List[Dict]:
    meta = _load_accounts_meta()
    result = []
    for account_id, info in meta.get("accounts", {}).items():
        result.append({"account_id": account_id, **{k: v for k, v in info.items() if k != "encrypted"}})
    return result


def get_account_meta(account_id: str) -> Dict:
    meta = _load_accounts_meta()
    acct = meta.get("accounts", {}).get(account_id)
    if not acct:
        raise KeyError(f"Unknown account: {account_id}")
    return acct


def update_account_meta(account_id: str, updates: Dict):
    meta = _load_accounts_meta()
    if account_id not in meta.get("accounts", {}):
        raise KeyError(f"Unknown account: {account_id}")
    meta["accounts"][account_id].update(updates)
    _save_accounts_meta(meta)


def delete_account(account_id: str):
    meta = _load_accounts_meta()
    meta.get("accounts", {}).pop(account_id, None)
    _save_accounts_meta(meta)
    if HAS_KEYRING and keyring is not None:
        try:
            keyring.delete_password(SERVICE_NAME, account_id)
        except Exception:
            pass
