#!/bin/bash
# Exp5: Nemotron-Labs-TwoTower-30B context-tower AR served via vLLM on r3 (single GB10).
# The context tower == frozen Nemotron-3-Nano-30B-A3B backbone. We remapped its
# checkpoint keys (context_tower.*->backbone.*, context_lm_head.weight->lm_head.weight)
# and arch->NemotronHForCausalLM so STOCK vLLM nemotron_h.py serves it, using vLLM's
# own mamba2/causal-conv1d kernels (sidesteps the HF causal_conv1d sm_121a garbage bug).
set -euo pipefail

IMG=vllm-dspark-runtime:mia-raf-pr1-nvfp4-b
MODEL_DIR=/home/keyspark/aeon27b/models/tt-context-vllm
NAME=twotower-ar
SERVED=nemotron-twotower-30b-bf16-context-ar
PORT=8000

MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.6}       # never exceed 0.86
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
EAGER=${EAGER:-1}                        # 1 = --enforce-eager
EXTRA=${EXTRA:-}

CACHE=/home/keyspark/tt-cache
mkdir -p "$CACHE/hf/modules" "$CACHE/vllm"

# --- clear GPU ---
bash /home/keyspark/gpu-clear.sh || true

# --- arm fastkill watchdog (setsid, bracket-pkill only) ---
cat > /home/keyspark/fastkill-tt.sh <<EOF
#!/bin/bash
while :; do
  a=\$(awk '/MemAvailable/{print \$2}' /proc/meminfo)
  if [ "\$a" -lt 3000000 ]; then
    docker rm -f ${NAME} 2>/dev/null
    echo "\$(date '+%F %T') FASTKILL fired at \${a}KB avail" >> /home/keyspark/oom-fastkill.log
    sleep 10
  fi
  sleep 2
done
EOF
chmod +x /home/keyspark/fastkill-tt.sh
pkill -f '[f]astkill-tt.sh' 2>/dev/null || true
setsid /home/keyspark/fastkill-tt.sh >/dev/null 2>&1 < /dev/null &
echo "fastkill-tt armed (pid $!)"

# --- (re)launch server ---
docker rm -f ${NAME} 2>/dev/null || true

EAGER_FLAG=""
[ "$EAGER" = "1" ] && EAGER_FLAG="--enforce-eager"

docker run -d --name ${NAME} --network host --ipc host --gpus all \
  --restart no \
  -v ${MODEL_DIR}:/model:ro \
  -v ${CACHE}:/cache \
  -e HF_HOME=/cache/hf \
  -e HF_MODULES_CACHE=/cache/hf/modules \
  -e VLLM_CACHE_ROOT=/cache/vllm \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_USE_FLASHINFER_MOE_FP16=0 \
  -e GENERATION_MODE=ar -e CONTEXT_ONLY=1 -e MAX_NEW_TOKENS_CAP=128 \
  --entrypoint vllm \
  ${IMG} \
  serve /model \
  --served-model-name ${SERVED} \
  --host 0.0.0.0 --port ${PORT} \
  --gpu-memory-utilization ${GPU_MEM_UTIL} \
  --max-model-len ${MAX_MODEL_LEN} \
  --max-num-seqs ${MAX_NUM_SEQS} \
  --attention-backend ${ATTN_BACKEND:-TRITON_ATTN} \
  ${EAGER_FLAG} ${EXTRA}

echo "launched ${NAME} (served=${SERVED} port=${PORT} maxlen=${MAX_MODEL_LEN} gpu=${GPU_MEM_UTIL} eager=${EAGER})"
echo "logs: docker logs -f ${NAME}"
