#!/bin/bash
# Fresh REG/LOCAL/SHARED verification for the mainline kernels.
set -e
for m in flash_attn_mma_db_addr attention_forward_cuda; do
  SO=$(find /root/.cache/torch_extensions -path "*$m/*" -name "*.so" | head -1)
  echo "== $m  ($SO)"
  stat -c '%y' "$SO"
  /usr/local/cuda-12.8/bin/cuobjdump -res-usage "$SO" | grep -E 'Function|REG'
  echo ""
done
