import datetime, sys
print("COLAB_PROBE_START", datetime.datetime.now(datetime.UTC).isoformat())
print("python", sys.version.split()[0])
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu", torch.cuda.get_device_name(0),
              round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2), "GB")
        x = torch.randn(1024, 1024, device="cuda")
        print("matmul_ok", round(float((x @ x).sum()), 3))
except Exception as e:
    print("torch_error", repr(e))
print("COLAB_PROBE_END", datetime.datetime.now(datetime.UTC).isoformat())
