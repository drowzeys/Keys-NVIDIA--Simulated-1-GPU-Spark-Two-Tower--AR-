#!/usr/bin/env python3
import json, urllib.request, time
BASE="http://10.100.10.3:8000"; MODEL="nemotron-twotower-30b-bf16-context-ar"
def comp(p,mt=16,temp=0.0):
    req=urllib.request.Request(BASE+"/v1/completions",
        data=json.dumps({"model":MODEL,"prompt":p,"max_tokens":mt,"temperature":temp}).encode(),
        headers={"Content-Type":"application/json"})
    r=json.loads(urllib.request.urlopen(req,timeout=600).read())
    return r["choices"][0]["text"], r.get("usage",{})

SENT="The annual regional survey recorded that local production increased steadily throughout the season. "
CODE="739140"
NEEDLE=f"\n>>> IMPORTANT: The secret access code is {CODE}. Remember it. <<<\n"
QUERY="\n\nQuestion: What is the secret access code stated earlier in this document?\nAnswer: The secret access code is"

def build(target_tokens, depth=0.5):
    # ~14 tokens/sentence; size filler, insert needle at depth
    reps=max(int(target_tokens/14),2)
    at=int(reps*depth)
    body=(SENT*at)+NEEDLE+(SENT*(reps-at))
    return "Document:\n"+body+QUERY

print(f"{'target':>8} {'depth':>6} {'prompt_tok':>10} {'found':>6} {'gen_s':>6}  output")
for tgt in [6000,32000,96000,128000,256000]:
    for depth in [0.5]:
        p=build(tgt,depth)
        t0=time.perf_counter()
        try:
            out,usage=comp(p,mt=16)
            dt=time.perf_counter()-t0
            found = CODE in out
            pt=usage.get("prompt_tokens","?")
            print(f"{tgt:>8} {depth:>6} {str(pt):>10} {str(found):>6} {dt:>6.1f}  {out.strip()[:40]!r}")
        except Exception as e:
            print(f"{tgt:>8} {depth:>6} {'ERR':>10}  {str(e)[:80]}")
