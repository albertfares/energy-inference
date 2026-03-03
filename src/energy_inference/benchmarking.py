import time
from typing import Optional

import torch

from energy_inference.tools.INA3221Sampler import INA3221Sampler


@torch.no_grad()
def bench_once(
    model: torch.nn.Module,
    batch: int,
    resolution: int,
    iters: int,
    warmup: int,
    device: torch.device,
    sampler: Optional[INA3221Sampler] = None,
) -> tuple[float, float, dict[str, float]]:
    """
    Benchmark average inference latency, FPS, and optional energy for one configuration.

    Returns:
        (latency_ms, fps, energy_dict)
    """
    x = torch.randn(batch, 3, resolution, resolution, device=device)

    for _ in range(warmup):
        _ = model(x)
        
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Hardware sampling start
    sample_start_perf = None
    if sampler is not None:
        _, sample_start_perf = sampler.start()
        time.sleep(0.2)  # Give sampler time to stabilize

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(x)
        
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    t1 = time.perf_counter()

    # Hardware sampling stop and integration
    energy_dict = {}
    if sampler is not None:
        sampler.stop(timeout=0.1)
        if sample_start_perf is not None:
            # Note: sampler.load_power_times might raise exception if CSV is completely empty/missing, 
            # we catch that gracefully to avoid crashing the whole experiment run.
            try:
                energy_dict = sampler.get_energy_range(t0, t1, sample_start_perf)
            except Exception as e:
                import logging
                logging.warning(f"Failed to integrate energy: {e}")

    total_s = t1 - t0
    latency_s = total_s / max(iters, 1)
    fps = batch / latency_s if latency_s > 0 else 0.0
    return latency_s * 1000.0, fps, energy_dict
