import asyncio
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from ai_engine.models.predictor import HybridAIPredictor


def parse_targets(argv: list[str]) -> list[tuple[str, str]]:
    targets: list[tuple[str, str]] = []
    for item in argv:
        symbol, timeframe = item.split(":", 1)
        targets.append((symbol, timeframe))
    return targets


async def main() -> None:
    targets = parse_targets(sys.argv[1:])
    predictor = HybridAIPredictor(autoload_models=True)
    result = await predictor.run_retraining_now(reason="final_targeted_batch", targets=targets)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
