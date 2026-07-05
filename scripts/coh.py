#!/usr/bin/env python3
import sys, time, json, urllib.request

BASE = "http://r3:8000"
MODEL = "nemotron-twotower-30b-bf16-context-ar"

def post(path, payload, timeout=120):
    req = urllib.request.Request(BASE+path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

# wait for health
t0=time.time()
while time.time()-t0 < 600:
    try:
        urllib.request.urlopen(BASE+"/health", timeout=5); break
    except Exception:
        time.sleep(5)
else:
    print("TIMEOUT waiting health"); sys.exit(1)
print(f"HEALTHY after {time.time()-t0:.0f}s\n")

prompts = [
    "France is a country",
    "The theory of relativity states that",
    "def fibonacci(n):\n    ",
    "Once upon a time, in a small village nestled between two mountains,",
]
for p in prompts:
    r = post("/v1/completions", {
        "model": MODEL, "prompt": p, "max_tokens": 128,
        "temperature": 0.0,
    })
    txt = r["choices"][0]["text"]
    print("="*70)
    print("PROMPT:", repr(p))
    print("OUTPUT:", txt)
    print()
