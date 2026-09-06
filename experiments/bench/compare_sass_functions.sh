#!/bin/bash
# Per-function ISETP/SEL/IMAD counts: precomputed addresses vs current kernel.
set -e
for MOD in attention_precomputed_addresses_cuda attention_forward_cuda; do
  SO=$(find /root/.cache/torch_extensions -path "*/$MOD/*" -name "*.so" | head -1)
  echo "== $MOD ($SO) =="
  /usr/local/cuda-12.8/bin/cuobjdump -sass "$SO" 2>/dev/null \
    | awk '/Function : /{fn=$3} /ISETP|SEL|IMAD|LEA|HMMA/{ split($0,a," ");
            for(i=1;i<=NF;i++){ op=$i; if(op ~ /^(ISETP|SEL|IMAD|LEA|HMMA)/){
              sub(/\..*/,"",op); cnt[fn"|"op]++; break } } }
           END{ for(k in cnt) print k, cnt[k] }' \
    | sort
  echo ""
done
