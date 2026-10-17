from .linear import LinearBaseline
from .encoder_only import EncoderOnlyTransformer
from .encoder_decoder import EncoderDecoderTransformer
from .deepar import DeepAR
from .temporal_fusion_transformer import TemporalFusionTransformer

MODEL_REGISTRY = {
    "linear": LinearBaseline,
    "encoder_only": EncoderOnlyTransformer,
    "encoder_decoder": EncoderDecoderTransformer,
    "deepar": DeepAR,
    "tft": TemporalFusionTransformer,
}

__all__ = [
    "LinearBaseline",
    "EncoderOnlyTransformer",
    "EncoderDecoderTransformer",
    "DeepAR",
    "TemporalFusionTransformer",
    "MODEL_REGISTRY",
]
