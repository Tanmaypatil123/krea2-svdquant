#!/usr/bin/env bash
set -euo pipefail

claude -p "You are advising on this repo's next concrete implementation for Krea2 SVDQuant inference acceleration. Use model-level reasoning for B200/Blackwell. We need actionable code/optimization suggestions, not generic advice. Focus on implementing and validating the first real speedup kernel after the packed W4A16 baseline: Triton tl.dot_scaled MXFP4/e2m1 and/or Triton Experimental Gluon tcgen05_mma_scaled for Krea2 shapes K=6144, N=4096/16384. Review current repo structure, identify pitfalls, and propose exact kernel design details. Do not edit files; return concise prioritized suggestions. ultrathink" \
  --model opus \
  --effort high \
  --allowedTools "Read,Bash(git status*),Bash(find*),Bash(ls*),Bash(sed*),Bash(python*)" \
  --max-turns 8
