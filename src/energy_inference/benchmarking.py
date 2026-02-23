import time

import torch


@torch.no_grad()
def bench_once(
    model: torch.nn.Module,
    batch: int,
    resolution: int,
    iters: int,
    warmup: int,
    device: torch.device,
) -> tuple[float, float]:
    """
    Benchmark average inference latency and FPS for one configuration.

    Returns:
        (latency_ms, fps)
    """
    x = torch.randn(batch, 3, resolution, resolution, device=device)

    for _ in range(warmup):
        _ = model(x)

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(x)
    t1 = time.perf_counter()

    total_s = t1 - t0
    latency_s = total_s / iters
    fps = batch / latency_s
    return latency_s * 1000.0, fps

