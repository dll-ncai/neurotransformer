"""Neurotransformer: hybrid convolution-transformer classification of single-channel EEG windows."""

from .models import MODEL_REGISTRY, Neurotransformer, build_model

__version__ = "0.1.0"

__all__ = ["MODEL_REGISTRY", "Neurotransformer", "build_model", "__version__"]
