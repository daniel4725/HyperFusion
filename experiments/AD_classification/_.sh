#!/bin/bash
send_tmux_job() {
  session_name="$1"
  command="$2"

  echo "session_name: $session_name"
  echo "command: $command"

  tmux new-window -t "$session_name"
  tmux send-keys -t "$session_name" "$command" Enter
}
session="s0"
tmux new-session -d -s "$session"
script_path="/home/duenias/PycharmProjects/HyperFusion/experiments/AD_classification/sandbox.py"


#for features_set in 1 2; do
for features_set in 1; do
  cmd="python3 $script_path a1 a2 a3"
  send_tmux_job $session "$cmd"
done
tmux attach -t $session
