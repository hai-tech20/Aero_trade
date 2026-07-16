import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from ai_engine.models.predictor import HybridAIPredictor


async def main():
    predictor = HybridAIPredictor(autoload_models=True)
    await predictor.start_retraining_scheduler()
    service = predictor.retraining_service
    if service is None:
        raise RuntimeError("Retraining service unavailable")
    service.set_test_config(buffer_threshold=3, interval_seconds=3600)

    for i in range(3):
        service.notify_new_trade_sample(
            {
                "order_id": f"test-{i}",
                "symbol": "EURUSD",
                "features": [0.1] * 10,
                "label": i % 2,
                "timeframe": "5min",
                "recorded_at": f"test-{i}",
            }
        )
        await asyncio.sleep(0.2)

    await asyncio.sleep(2)
    print(predictor.get_retraining_status())
    await predictor.stop_retraining_scheduler()


if __name__ == "__main__":
    asyncio.run(main())
