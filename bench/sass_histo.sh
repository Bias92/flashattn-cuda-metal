#!/bin/bash
# SASS instruction histogram + HMMA rhythm check.
# Usage: sass_histo.sh [module-substring]   (default: mma_db, excludes _addr)
# NOTE: histogram covers the WHOLE .so (both WRITE_L instantiations) --
# use for direction, not as precise per-kernel paper numbers.
set -e
PAT="${1:-mma_db/}"
SO=$(find /root/.cache/torch_extensions -path "*$PAT*" -name "*.so" | head -1)
echo "SO: $SO"
/usr/local/cuda-12.8/bin/cuobjdump -sass "$SO" > /tmp/db.sass

echo "== opcode histogram (WRITE_L=false kernel region and beyond, whole file) =="
grep -oE '\*/ +[A-Z@!][A-Z0-9@!.]+' /tmp/db.sass \
  | sed -E 's|\*/ +||; s/^@!?[A-Z0-9]+ +//; s/\..*$//' \
  | sort | uniq -c | sort -rn | head -20

echo ""
echo "== max gap (instructions) between consecutive HMMAs in the hottest stretch =="
grep -nE 'HMMA|MUFU|FMUL|FADD|FFMA|LDSM|SHFL' /tmp/db.sass \
  | sed -E 's/:.*\*\/ +/ /; s/ ;.*//' \
  | awk '{split($0,p," "); line=p[1]; op=p[2]; sub(/\..*/,"",op);
          if(op=="HMMA"){ if(prev>0) print line-prev; prev=line } }' \
  | sort -n | uniq -c | tail -8
