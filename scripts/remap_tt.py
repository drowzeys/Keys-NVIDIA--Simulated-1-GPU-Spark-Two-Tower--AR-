#!/usr/bin/env python3
"""Remap TwoTower context tower -> standard NemotronHForCausalLM checkpoint.
  context_tower.*        -> backbone.*
  context_lm_head.weight -> lm_head.weight
Fast path: rewrite only the safetensors JSON header (pad to original length),
copy the data blob verbatim. Self-validates each output shard.
"""
import json, struct, os, shutil, sys, glob

SRC = "/home/keyspark/aeon27b/models/tt-context"
DST = "/home/keyspark/aeon27b/models/tt-context-vllm"
os.makedirs(DST, exist_ok=True)

CT = "context_tower."
def remap(k):
    if k == "context_lm_head.weight":
        return "lm_head.weight"
    if k.startswith(CT):
        return "backbone." + k[len(CT):]
    raise ValueError("unexpected key: " + k)

def process(src, dst):
    with open(src, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n).decode("utf-8"))
    newh = {}
    for k, v in header.items():
        newh[k if k == "__metadata__" else remap(k)] = v
    js = json.dumps(newh, separators=(",", ":")).encode("utf-8")
    assert len(js) <= n, (len(js), n)
    js = js + b" " * (n - len(js))
    with open(src, "rb") as f, open(dst, "wb") as g:
        g.write(struct.pack("<Q", n))
        g.write(js)
        f.seek(8 + n)
        shutil.copyfileobj(f, g, length=64 * 1024 * 1024)

shards = sorted(glob.glob(os.path.join(SRC, "*.safetensors")))
print(f"{len(shards)} shards", flush=True)
for i, s in enumerate(shards):
    d = os.path.join(DST, os.path.basename(s))
    process(s, d)
    # validate
    from safetensors import safe_open
    with safe_open(d, framework="pt") as sf:
        ks = list(sf.keys())
        assert not any(k.startswith("context_") for k in ks), "leftover context_ key"
        if i == 0:
            t = sf.get_tensor(ks[0])  # actually read a tensor
            print("  validate read:", ks[0], tuple(t.shape), t.dtype, flush=True)
    print(f"[{i+1}/{len(shards)}] {os.path.basename(s)} OK", flush=True)

# index.json
with open(os.path.join(SRC, "model.safetensors.index.json")) as f:
    idx = json.load(f)
idx["weight_map"] = {remap(k): v for k, v in idx["weight_map"].items()}
with open(os.path.join(DST, "model.safetensors.index.json"), "w") as f:
    json.dump(idx, f, indent=2)
print("index.json remapped", flush=True)

# config.json: architectures -> NemotronHForCausalLM, drop auto_map/twotower fields
with open(os.path.join(SRC, "config.json")) as f:
    cfg = json.load(f)
cfg["architectures"] = ["NemotronHForCausalLM"]
cfg.pop("auto_map", None)
with open(os.path.join(DST, "config.json"), "w") as f:
    json.dump(cfg, f, indent=2)
print("config.json written arch=NemotronHForCausalLM", flush=True)

# copy tokenizer + generation config (dereference, plain files)
for fn in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
           "generation_config.json"]:
    src = os.path.join(SRC, fn)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(DST, fn))
        print("copied", fn, flush=True)
print("DONE", flush=True)
