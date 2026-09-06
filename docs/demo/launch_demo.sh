#!/bin/bash
DEMO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
printf -v naive_command 'python3 %q' "$DEMO_DIR/naive_demo.py"
printf -v wmma_command 'sleep 1 && python3 %q' "$DEMO_DIR/wmma_demo.py"
tmux kill-session -t demo 2>/dev/null
tmux new-session -d -s demo -x 220 -y 45
tmux send-keys -t demo "$naive_command" C-m
tmux split-window -h -t demo
tmux send-keys -t demo "$wmma_command" C-m
tmux attach -t demo
