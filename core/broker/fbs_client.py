"""Re-export MT5BrokerClient for backward compatibility."""

from core.broker.mt5_broker_client import MT5BrokerClient, FBSClient, fbs_client

__all__ = ["MT5BrokerClient", "FBSClient", "fbs_client"]
