"""MiniRocket (Dempster et al., KDD 2021) with a trainable linear head, for inputs ``(batch, 1, L)``.

The random convolutional transform is sktime's implementation (BSD-3-Clause; ``pip install sktime numba``).
MiniRocket's kernels are fixed and their biases are set from data, so the transform is fitted once, on the first
training batch the model sees (as in the paper's experiments), and only the linear head is trained. The
fitted transform is kept in buffers, so checkpoints restore the complete model.
"""

import numpy as np
import torch
import torch.nn as nn

N_KERNELS = 84  # MiniRocket's fixed kernel set; each kernel gives (num_kernels // 84) features
_PARAM_BUFFERS = ("dilations", "num_features_per_dilation", "biases")


def _sktime_minirocket():
    try:
        from sktime.transformations.rocket import MiniRocket
    except ImportError:
        try:
            from sktime.transformations.panel.rocket import MiniRocket  # sktime < 1.0
        except ImportError:
            raise ImportError("MiniRocket needs sktime and numba: pip install sktime numba, "
                              "or pip install -e '.[baselines]'") from None
    return MiniRocket


class MiniRocket(nn.Module):
    def __init__(self, n_classes=3, num_kernels=10000, max_dilations_per_kernel=32, n_jobs=1):
        super().__init__()
        self.num_kernels = num_kernels
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.n_jobs = n_jobs
        self.n_features = N_KERNELS * (num_kernels // N_KERNELS)  # 9,996 for 10,000
        for name, dtype in zip(_PARAM_BUFFERS, (torch.int32, torch.int32, torch.float32)):
            self.register_buffer(name, torch.zeros(0, dtype=dtype))
        self.head = nn.Linear(self.n_features, n_classes)
        self._transformer = None

    @property
    def is_fitted(self):
        return self.biases.numel() > 0

    def fit_transform_params(self, x, seed=None):
        """Fit the dilations and biases on ``x`` (``(n, 1, L)``). ``seed`` defaults to one drawn from torch's RNG."""
        if seed is None:
            seed = int(torch.randint(0, 2 ** 31 - 1, (1,)).item())
        transformer = _sktime_minirocket()(num_kernels=self.num_kernels, n_jobs=self.n_jobs,
                                           max_dilations_per_kernel=self.max_dilations_per_kernel, random_state=seed)
        transformer.fit(self._to_numpy(x))
        for name, value in zip(_PARAM_BUFFERS, transformer.parameters):
            setattr(self, name, torch.as_tensor(value, device=self.biases.device))
        if self.biases.numel() != self.n_features:
            raise RuntimeError(f"MiniRocket produced {self.biases.numel()} features, expected {self.n_features}")
        self._transformer = transformer
        return self

    def forward(self, x):
        if not self.is_fitted:
            if not self.training:
                raise RuntimeError("MiniRocket's transform is not fitted: train the model, call "
                                   "fit_transform_params(x) or load a checkpoint first")
            self.fit_transform_params(x)
        features = self._transform(x)
        return self.head(torch.as_tensor(features, device=x.device))

    def _transform(self, x):
        x_np = self._to_numpy(x)
        if self._transformer is None:
            # Restore from the buffers (after load_state_dict): sktime needs a fit call to set its own state,
            # then the fitted parameters are replaced by the stored ones.
            transformer = _sktime_minirocket()(num_kernels=self.num_kernels, n_jobs=self.n_jobs,
                                               max_dilations_per_kernel=self.max_dilations_per_kernel, random_state=0)
            transformer.fit(x_np)
            transformer.parameters = tuple(getattr(self, name).cpu().numpy() for name in _PARAM_BUFFERS)
            self._transformer = transformer
        return np.asarray(self._transformer.transform(x_np), dtype=np.float32)

    @staticmethod
    def _to_numpy(x):
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f"MiniRocket expects input of shape (batch, 1, length), got {tuple(x.shape)}")
        return x.detach().float().cpu().numpy()

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # The buffers are empty until fitted; resize them to the checkpoint's shapes before loading.
        for name in _PARAM_BUFFERS:
            if prefix + name in state_dict:
                setattr(self, name, torch.empty_like(state_dict[prefix + name], device=getattr(self, name).device))
        self._transformer = None
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
