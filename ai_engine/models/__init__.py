"""AI model package exports."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .predictor import HybridAIPredictor, Prediction
    from .predictor import ai_predictor, get_ai_predictor

__all__ = ["ai_predictor", "HybridAIPredictor", "Prediction", "get_ai_predictor"]


def __getattr__(name: str):
    if name in __all__:
        from .predictor import ai_predictor, HybridAIPredictor, Prediction, get_ai_predictor
        exports = {
            "ai_predictor": ai_predictor,
            "HybridAIPredictor": HybridAIPredictor,
            "Prediction": Prediction,
            "get_ai_predictor": get_ai_predictor,
        }
        return exports[name]
    raise AttributeError(name)