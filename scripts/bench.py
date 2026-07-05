#!/usr/bin/env python3
import json, time, urllib.request, threading, sys

BASE = "http://r3:8000"
MODEL = "nemotron-twotower-30b-bf16-context-ar"

def stream_one(prompt, max_tokens=128):
    payload = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
               "temperature": 0.0, "stream": True,
               "stream_options": {"include_usage": True}}
    req = urllib.request.Request(BASE+"/v1/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"})
    t0 = time.perf_counter(); ttft=None; ntok=0; usage=None; last=t0
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line=line.decode().strip()
            if not line.startswith("data:"): continue
            d=line[5:].strip()
            if d=="[DONE]": break
            o=json.loads(d)
            if o.get("choices"):
                txt=o["choices"][0].get("text","")
                if txt:
                    if ttft is None: ttft=time.perf_counter()-t0
                    ntok+=1; last=time.perf_counter()
            if o.get("usage"): usage=o["usage"]
    ct = usage["completion_tokens"] if usage else ntok
    decode_t = max(last-(t0+(ttft or 0)),1e-9)
    return {"ttft":ttft or 0, "ct":ct, "decode_s":decode_t,
            "tok_s": (ct-1)/decode_t if ct>1 else 0, "wall": last-t0}

def concurrency(C, prompt, max_tokens=128):
    res=[None]*C; thr=[]
    t0=time.perf_counter()
    def w(i): res[i]=stream_one(prompt,max_tokens)
    for i in range(C):
        t=threading.Thread(target=w,args=(i,)); t.start(); thr.append(t)
    for t in thr: t.join()
    wall=time.perf_counter()-t0
    tot_ct=sum(r["ct"] for r in res)
    agg = sum(r["tok_s"] for r in res)               # sum of per-req decode rates
    thru = tot_ct/wall                                # end-to-end throughput
    avg_single=sum(r["tok_s"] for r in res)/C
    avg_ttft=sum(r["ttft"] for r in res)/C
    return {"C":C,"agg_tok_s":agg,"e2e_tok_s":thru,"per_req_tok_s":avg_single,
            "avg_ttft":avg_ttft,"tot_ct":tot_ct,"wall":wall}

PROMPT = "In recent years, advances in artificial intelligence have"
print("=== warmup ==="); stream_one(PROMPT); print("done\n")

print("=== CONCURRENCY / BENCHMARK SWEEP (max_tokens=128, temp=0, eager) ===")
print(f"{'C':>3} {'agg_tok/s':>10} {'e2e_tok/s':>10} {'per_req':>8} {'ttft_s':>7} {'tot_tok':>8} {'wall_s':>7}")
for C in [1,2,4,8]:
    r=concurrency(C,PROMPT)
    print(f"{r['C']:>3} {r['agg_tok_s']:>10.1f} {r['e2e_tok_s']:>10.1f} {r['per_req_tok_s']:>8.1f} {r['avg_ttft']:>7.3f} {r['tot_ct']:>8} {r['wall']:>7.2f}")
    sys.stdout.flush()
