"""Colab Pro job wrapper: bake config (no positional args allowed by `colab exec`).

Sets environment defaults then executes the uploaded training module.
"""
import os
import sys

CONFIG = {
    "MODEL_ID": "Qwen/Qwen2.5-7B",
    "DATA_PATH": "/content/train_all.parquet",
    "LABELS_PATH": "/content/labels.json",
    "OUT_DIR": "/content/out",
    "JOB_HOURS": "5.2",
    "MAX_LEN": "384",
    "EPOCHS": "2.0",
    "BATCH": "4",
    "ACCUM": "4",
    "USE_4BIT": "1",
}
for _k, _v in CONFIG.items():
    os.environ[_k] = _v
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.argv = ["track05_lora_train.py"]
_SRC = "/content/track05_lora_train.py"
print("WRAPPER config:", CONFIG, flush=True)
exec(compile(open(_SRC).read(), _SRC, "exec"), {"__name__": "__main__"})
