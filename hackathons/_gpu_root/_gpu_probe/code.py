import os, sys, subprocess, datetime
print("PROBE_START", datetime.datetime.utcnow().isoformat())
print("python", sys.version.split()[0])
try:
    import torch
    print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print("gpu_name", p.name, "vram_gb", round(p.total_memory / 2**30, 2))
        x = torch.randn(2048, 2048, device="cuda")
        y = (x @ x).sum().item()
        print("matmul_ok", isinstance(y, float), "sample", round(float(y), 3))
        print("gpu_mem_alloc_mb", round(torch.cuda.memory_allocated() / 2**20, 1))
except Exception as e:
    print("torch_error", repr(e))
print("PROBE_END", datetime.datetime.utcnow().isoformat())
