"""Baseline models compared with Neurotransformer in the paper.

EEGNet and EEGConformer need braindecode, MiniRocket needs sktime and numba (``pip install -e ".[baselines]"``);
they are imported only when one of these models is built.
"""

from .braindecode_models import eegconformer, eegnet
from .inception_time import InceptionTime
from .leednet import LEEDNetRaw
from .minirocket import MiniRocket
from .scnet import SCNet
from .wavenet_lstm import WaveNetLSTM

BASELINES = {
    "eegnet": eegnet,
    "eegconformer": eegconformer,
    "scnet": SCNet,
    "wavenet_lstm": WaveNetLSTM,
    "inceptiontime": InceptionTime,
    "minirocket": MiniRocket,
    "leednet_raw": LEEDNetRaw,
}

__all__ = ["BASELINES", "InceptionTime", "LEEDNetRaw", "MiniRocket", "SCNet", "WaveNetLSTM", "eegconformer", "eegnet"]
