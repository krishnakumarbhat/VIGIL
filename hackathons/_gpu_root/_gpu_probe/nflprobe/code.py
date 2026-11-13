import os, json
src = "/kaggle/input/competitions/nfl-big-data-bowl-2027"
out = {"exists": os.path.isdir(src), "listing": {}}
if os.path.isdir(src):
    for root, dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        out["listing"][rel] = [(f, os.path.getsize(os.path.join(root,f))) for f in files][:40]
print(json.dumps(out, indent=1)[:4000])
