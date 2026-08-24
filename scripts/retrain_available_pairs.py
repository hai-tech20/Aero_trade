import asyncio
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from ai_engine.models.predictor import HybridAIPredictor
from config.settings import DATA_DIR, TRADING_CONFIG

def build_targets() -> list[tuple[str, str]]:
    targets: list[tuple[str, str]] = []
    for symbol in TRADING_CONFIG.get("pairs", []):
        for timeframe in TRADING_CONFIG.get("timeframes", []):
            path = DATA_DIR / f"{symbol}_{timeframe}_history.parquet"
            if path.exists():
                targets.append((str(symbol), timeframe))
    return targets


async def main() -> None:
    predictor = HybridAIPredictor(autoload_models=True)
    targets = build_targets()
    result = await predictor.run_retraining_now(reason="available_pairs_retrain", targets=targets)
    print(json.dumps({"target_count": len(targets), "targets": targets, "result": result}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
