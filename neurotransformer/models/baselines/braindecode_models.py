"""EEGNet and EEG Conformer from braindecode (BSD-3-Clause), set up for one channel, 400 samples at 200 Hz.

The paper's runs used braindecode 0.8 (``pip install "braindecode==0.8.*"``); other versions rename models and
layers, so their checkpoints may not load.
"""

import inspect

N_TIMES, SFREQ = 400, 200


def _braindecode_models():
    try:
        from braindecode import models
    except ImportError:
        raise ImportError('EEGNet and EEGConformer need braindecode: pip install "braindecode==0.8.*", '
                          "or pip install -e '.[baselines]'") from None
    return models


def eegnet(n_classes=3):
    """EEGNet-v4 (Lawhern et al., 2018)."""
    return _braindecode_models().EEGNetv4(n_chans=1, n_outputs=n_classes, n_times=N_TIMES, sfreq=SFREQ)


def eegconformer(n_classes=3):
    """EEG Conformer (Song et al., 2023). ``final_fc_length='auto'`` sizes the classifier for 400-sample windows."""
    cls = _braindecode_models().EEGConformer
    kwargs = dict(n_chans=1, n_outputs=n_classes, n_times=N_TIMES, sfreq=SFREQ, final_fc_length="auto")
    if "add_log_softmax" in inspect.signature(cls.__init__).parameters:
        kwargs["add_log_softmax"] = False  # return logits; the predictions are the same either way
    return cls(**kwargs)
