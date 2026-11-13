import os, shutil, json
src="/kaggle/input/competitions/gemma-4-developer-agent"; dst="/kaggle/working/gemma4_data"; os.makedirs(dst, exist_ok=True)
man={}
for root,dirs,files in os.walk(src):
    rel=os.path.relpath(root,src); man[rel]={"dirs":dirs[:40],"files":[(f,os.path.getsize(os.path.join(root,f))) for f in files[:60]]}
    for f in files:
        p=os.path.join(root,f)
        try:
            if os.path.getsize(p) < 60*1024*1024: shutil.copy(p, os.path.join(dst,f))
        except Exception: pass
json.dump(man, open(os.path.join(dst,"_manifest.json"),"w"), indent=1)
print("copied:", sorted(os.listdir(dst))[:40])
