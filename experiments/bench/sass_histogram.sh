#!/bin/bash
# SASS opcode histogram and line-number gaps between HMMA matches.
# Usage: sass_histogram.sh [module-name] (default: attention_double_buffer_cuda)
# The histogram covers the whole .so, including both WRITE_L instantiations.
# Counts are not isolated to a single kernel.
set -e
MOD="${1:-attention_double_buffer_cuda}"
SO=$(find /root/.cache/torch_extensions -path "*/$MOD/*" -name "*.so" | head -1)
echo "SO: $SO"
/usr/local/cuda-12.8/bin/cuobjdump -sass "$SO" > /tmp/db.sass

echo "== opcode histogram (whole file, all kernel specializations) =="
grep -oE '\*/ +[A-Z@!][A-Z0-9@!.]+' /tmp/db.sass \
  | sed -E 's|\*/ +||; s/^@!?[A-Z0-9]+ +//; s/\..*$//' \
  | sort | uniq -c | sort -rn | head -20

echo ""
echo "== largest SASS line-number gaps between consecutive HMMA matches (whole file) =="
grep -nE 'HMMA|MUFU|FMUL|FADD|FFMA|LDSM|SHFL' /tmp/db.sass \
  | sed -E 's/:.*\*\/ +/ /; s/ ;.*//' \
  | awk '{split($0,p," "); line=p[1]; op=p[2]; sub(/\..*/,"",op);
          if(op=="HMMA"){ if(prev>0) print line-prev; prev=line } }' \
  | sort -n | uniq -c | tail -8
