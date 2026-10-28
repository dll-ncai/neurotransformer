from .ablations import (
    NeurotransformerWOConv,
    NeurotransformerWODecoder,
    NeurotransformerWODil,
    NeurotransformerWOEncoder,
    NeurotransformerWORes,
    NeurotransformerWOTrans,
)
from .baselines import BASELINES
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


# Every name build_model accepts: Neurotransformer, its ablations and the baselines
MODEL_NAMES = list(MODEL_REGISTRY) + list(BASELINES)


def build_model(name="neurotransformer", n_classes=3):
    """Instantiate a model by name, from :data:`MODEL_REGISTRY` or :data:`BASELINES`."""
    model_cls = MODEL_REGISTRY.get(name) or BASELINES.get(name)
    if model_cls is None:
        raise ValueError(f"Unknown model '{name}'. Choose from: {', '.join(MODEL_NAMES)}")
    return model_cls(n_classes=n_classes)


__all__ = [
    "ABLATIONS",
    "BASELINES",
    "MODEL_NAMES",
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
