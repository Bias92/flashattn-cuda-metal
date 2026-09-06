#!/bin/bash
# Fresh REG/LOCAL/SHARED verification for the mainline kernels.
set -e
for m in db_addr db_full; do
  SO=$(find /root/.cache/torch_extensions -path "*mma_$m/*" -name "*.so" | head -1)
  echo "== $m  ($SO)"
  stat -c '%y' "$SO"
  /usr/local/cuda-12.8/bin/cuobjdump -res-usage "$SO" | grep -E 'Function|REG'
  echo ""
done
