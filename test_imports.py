# test_imports.py
print("Testing imports...")

try:
    from config.settings import TRADING_CONFIG
    print("[OK] config.settings imported")
    
    from ai_engine.models.predictor import ai_predictor
    print("[OK] ai_engine.models.predictor imported")
    
    from decision_maker.signal_fusion import signal_fusion
    print("[OK] decision_maker.signal_fusion imported")
    
    from data.market_data import market_data
    print("[OK] data.market_data imported")
    
    from core.broker.fbs_client import fbs_client
    print("[OK] core.broker.fbs_client imported")
    
    from core.risk_management.risk_manager import risk_manager
    print("[OK] core.risk_management.risk_manager imported")
    
    from utils.logger import logger
    print("[OK] utils.logger imported")
    
    print("\n[SUCCESS] ALL IMPORTS SUCCESSFUL!")
    
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"[ERROR] Error: {e}")