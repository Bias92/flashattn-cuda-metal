#!/bin/bash
# REG/LOCAL/SHARED for the precomputed-addresses and current kernels.
set -e
for m in attention_precomputed_addresses_cuda attention_forward_cuda; do
  SO=$(find /root/.cache/torch_extensions -path "*/$m/*" -name "*.so" | head -1)
  echo "== $m  ($SO)"
  stat -c '%y' "$SO"
  /usr/local/cuda-12.8/bin/cuobjdump -res-usage "$SO" | grep -E 'Function|REG'
  echo ""
done
