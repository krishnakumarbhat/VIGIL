import os, shutil, json
src = "/kaggle/input/competitions/nfl-big-data-bowl-2027"
dst = "/kaggle/working/nfl_data"; os.makedirs(dst, exist_ok=True)
found = []
for root, dirs, files in os.walk(src):
    for f in files:
        p = os.path.join(root, f)
        sz = os.path.getsize(p)
        found.append((os.path.relpath(p, src), sz))
print("ALL FILES:", json.dumps(found, indent=1)[:5000])
for rel, sz in found:
    p = os.path.join(src, rel)
    try:
        if sz < 400*1024*1024:
            shutil.copy(p, os.path.join(dst, os.path.basename(rel)))
            print("copied", rel, sz)
    except Exception as e:
        print("skip", rel, repr(e)[:80])
# peek headers
for f in sorted(os.listdir(dst)):
    if f.endswith(".csv"):
        with open(os.path.join(dst, f), "r", errors="ignore") as fh:
            head = [next(fh, "") for _ in range(3)]
        print("HEAD", f, head)
