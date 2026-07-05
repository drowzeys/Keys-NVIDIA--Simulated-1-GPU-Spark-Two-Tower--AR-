#!/usr/bin/env python3
import json, urllib.request
BASE="http://r3:8000"; MODEL="nemotron-twotower-30b-bf16-context-ar"
def comp(p,mt=128,temp=0.0):
    req=urllib.request.Request(BASE+"/v1/completions",
        data=json.dumps({"model":MODEL,"prompt":p,"max_tokens":mt,"temperature":temp}).encode(),
        headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(req,timeout=120).read())["choices"][0]["text"]
prompts=[
 ("Factual", "The capital of Japan is"),
 ("Reasoning", "If a train travels 60 miles per hour for 2.5 hours, the total distance covered is"),
 ("List/knowledge", "The three primary colors are"),
 ("Multilingual (fr)", "La Tour Eiffel se trouve dans la ville de"),
 ("Completion/logic", "Water boils at 100 degrees Celsius at sea level because"),
]
for tag,p in prompts:
    print("="*70); print(f"[{tag}] PROMPT: {p!r}"); print("OUT:", comp(p).strip()[:400]); print()
