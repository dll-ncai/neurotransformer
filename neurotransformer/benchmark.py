"""Model cost: parameter count, FLOPs and single-window inference latency."""

import time

import torch

from .utils import count_params


def count_flops(model, example_input):
    """FLOPs of one forward pass (2 x MACs). Needs the optional ``thop`` package; returns None without it."""
    try:
        from thop import profile
    except ImportError:
        return None
    import copy

    # thop registers buffers on the model it profiles, so profile a copy.
    macs, _ = profile(copy.deepcopy(model).eval(), inputs=(example_input,), verbose=False)
    return float(macs) * 2.0


@torch.no_grad()
def measure_latency_ms(model, example_input, device="cpu", warmup=10, iters=100):
    """Average wall-clock time of one forward pass on ``device``, in milliseconds."""
    device = torch.device(device)
    model = model.to(device).eval()
    x = example_input.to(device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - start) / iters * 1000.0


def model_costs(model, window_samples=400, warmup=10, iters=100):
    """Parameters (M), MFLOPs per window and CPU/GPU latency (ms) for a batch of one window."""
    example = torch.randn(1, 1, window_samples)
    flops = count_flops(model.cpu(), example)
    costs = {
        "params_m": count_params(model) / 1e6,
        "mflops_per_segment": flops / 1e6 if flops is not None else None,
        "cpu_latency_ms": measure_latency_ms(model, example, "cpu", warmup, iters),
        "gpu_latency_ms": None,
    }
    if torch.cuda.is_available():
        costs["gpu_latency_ms"] = measure_latency_ms(model, example, "cuda", warmup, iters)
        model.cpu()
    return costs
