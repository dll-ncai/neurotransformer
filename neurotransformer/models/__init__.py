from .ablations import (
    NeurotransformerWOConv,
    NeurotransformerWODecoder,
    NeurotransformerWODil,
    NeurotransformerWOEncoder,
    NeurotransformerWORes,
    NeurotransformerWOTrans,
)
from .blocks import MFFMBlock, WaveBlock, WaveLayer
from .neurotransformer import Neurotransformer

MODEL_REGISTRY = {
    "neurotransformer": Neurotransformer,
    "wo_res": NeurotransformerWORes,
    "wo_dil": NeurotransformerWODil,
    "wo_conv": NeurotransformerWOConv,
    "wo_encoder": NeurotransformerWOEncoder,
    "wo_decoder": NeurotransformerWODecoder,
    "wo_trans": NeurotransformerWOTrans,
}

ABLATIONS = {
    "wo_res": "Removed residual convolutions (MFFM)",
    "wo_dil": "Removed dilated convolutions",
    "wo_conv": "Removed convolutions",
    "wo_encoder": "Removed encoder",
    "wo_decoder": "Removed decoder",
    "wo_trans": "Removed transformer",
}


def build_model(name="neurotransformer", n_classes=3):
    """Instantiate a model from :data:`MODEL_REGISTRY` by name."""
    try:
        model_cls = MODEL_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown model '{name}'. Choose from: {', '.join(MODEL_REGISTRY)}") from None
    return model_cls(n_classes=n_classes)


__all__ = [
    "ABLATIONS",
    "MODEL_REGISTRY",
    "MFFMBlock",
    "Neurotransformer",
    "NeurotransformerWOConv",
    "NeurotransformerWODecoder",
    "NeurotransformerWODil",
    "NeurotransformerWOEncoder",
    "NeurotransformerWORes",
    "NeurotransformerWOTrans",
    "WaveBlock",
    "WaveLayer",
    "build_model",
]
