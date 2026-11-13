import os, sys, torch, datetime
print("SANITY", datetime.datetime.now().isoformat())
print("cwd", os.getcwd())
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("files:", sorted(os.listdir("/content"))[:20])
