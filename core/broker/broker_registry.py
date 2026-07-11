"""Broker presets for MT5-based brokers."""

from dataclasses import dataclass
from typing import List


@dataclass
class BrokerPreset:
    id: str
    display_name: str
    demo_server: str
    real_server: str
    website: str = ""
    logo_path: str = ""


SUPPORTED_BROKERS: List[BrokerPreset] = [
    BrokerPreset("fbs", "FBS", "FBS-Demo", "FBS-Real", "https://fbs.com"),
    BrokerPreset("exness", "Exness", "Exness-MT5Trial8", "Exness-MT5Real8", "https://exness.com"),
    BrokerPreset("icmarkets", "IC Markets", "ICMarketsSC-Demo", "ICMarketsSC-Live01", "https://icmarkets.com"),
    BrokerPreset("xm", "XM", "XMGlobal-Demo 3", "XMGlobal-Real 3", "https://xm.com"),
    BrokerPreset("pepperstone", "Pepperstone", "Pepperstone-Demo", "Pepperstone-Live01", "https://pepperstone.com"),
    BrokerPreset("hfm", "HFM (HotForex)", "HFMarketsGlobal-Demo", "HFMarketsGlobal-Live", "https://hfm.com"),
    BrokerPreset("custom", "Other (enter server manually)", "", "", ""),
]


def get_broker(broker_id: str) -> BrokerPreset:
    match = next((b for b in SUPPORTED_BROKERS if b.id == broker_id), None)
    if not match:
        raise ValueError(f"Unknown broker: {broker_id}")
    return match


def broker_to_dict(broker: BrokerPreset) -> dict:
    return {
        "id": broker.id,
        "display_name": broker.display_name,
        "demo_server": broker.demo_server,
        "real_server": broker.real_server,
        "website": broker.website,
        "logo_path": broker.logo_path,
    }
