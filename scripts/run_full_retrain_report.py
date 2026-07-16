import asyncio
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from ai_engine.models.predictor import HybridAIPredictor
from config.settings import TRADING_CONFIG


async def main():
    predictor = HybridAIPredictor(autoload_models=True)
    result = await predictor.run_retraining_now("full_history_download")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
