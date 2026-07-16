import asyncio
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from ai_engine.models.predictor import HybridAIPredictor

TARGETS = [
    ("GBPUSD", "15min"),
    ("USDJPY", "1min"),
    ("XAUUSD", "1min"),
    ("XAUUSD", "5min"),
    ("XAUUSD", "15min"),
    ("BTCUSD", "1min"),
    ("BTCUSD", "5min"),
    ("BTCUSD", "15min"),
]


async def main() -> None:
    predictor = HybridAIPredictor(autoload_models=True)
    result = await predictor.run_retraining_now(reason="per_symbol_fix_batch", targets=TARGETS)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
