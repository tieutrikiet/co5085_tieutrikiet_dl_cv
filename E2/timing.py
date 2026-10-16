"""Đo tốc độ: benchmark riêng MSA và latency suy luận của cả mô hình."""
import copy
import time

import torch

from common.utils import set_seed


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()


def bench_attention(cls, N, D=64, h=4, B=128, iters=50, warmup=10, device=torch.device("cpu")):
    """Thời gian trung bình (ms) cho một lượt forward + backward của một module MSA."""
    set_seed(0)
    module = cls(embed_dim=D, num_heads=h).to(device).train()
    x = torch.randn(B, N, D, device=device, requires_grad=True)
    for i in range(warmup + iters):
        if i == warmup:
            sync(device)
            t0 = time.perf_counter()
        module(x).sum().backward()
    sync(device)
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1e3  # Convert to milliseconds


@torch.no_grad()
def measure_latency(model, iters=200, warmup=10, device=torch.device("cpu")):
    """Đo độ trễ của một model với batch=1. Chọn device mặc định là cpu để gần với kịch bản chạy trên thiết bị."""
    model = copy.deepcopy(model).to(device).eval()
    x = torch.randn(1, 1, 28, 28, device=device)

    # warm up
    for _ in range(warmup):
        _ = model(x)

    sync(device)
    t0 = time.perf_counter()

    for _ in range(iters):
        _ = model(x)

    sync(device)
    t1 = time.perf_counter()

    return 1000 * (t1 - t0) / iters  # latency in milliseconds
