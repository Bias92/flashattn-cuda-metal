# Old demo recordings

These recordings compare the earlier WMMA implementation with a separate PyTorch
matmul/softmax expression. They are not the current kernel or the PyTorch SDPA
benchmark in the main README.

The original `.cast`, GIF and MP4 files are preserved. The accompanying scripts
are `wmma_demo.py` and `naive_demo.py`; `launch_demo.sh` runs them in two tmux panes.

For the current comparison, use [compare_pytorch.py](../../bench/compare_pytorch.py).
