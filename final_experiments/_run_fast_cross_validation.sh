#!/bin/bash

send_tmux_job() {
  session_name="$1"
  command="$2"

  echo "session_name: $session_name"
  echo "command: $command"

  tmux new-window -t "$session_name"
  tmux send-keys -t "$session_name" "$command" Enter
}

#experiment_path="/home/duenias/PycharmProjects/HyperNetworks/final_experiments/baseline-resnet.sh"
#experiment_path="/home/duenias/PycharmProjects/HyperNetworks/final_experiments/baseline-tabular.sh"
#experiment_path="/home/duenias/PycharmProjects/HyperNetworks/final_experiments/Film.sh"
experiment_path="/home/duenias/PycharmProjects/HyperNetworks/final_experiments/TabularAsHyper.sh"
#experiment_path="/home/duenias/PycharmProjects/HyperNetworks/final_experiments/DAFT.sh"
#experiment_path="/home/duenias/PycharmProjects/HyperNetworks/final_experiments/ImageAsHyper.sh"
session="s2"
gpu_fold_a="2 0"
gpu_fold_b="2 1"
gpu_fold_c="1 2"
gpu_fold_d="1 3"


tmux new-session -d -s "$session"
send_tmux_job $session "sh $experiment_path $gpu_fold_a"
sleep 2
send_tmux_job $session "sh $experiment_path $gpu_fold_b"
sleep 2
send_tmux_job $session "sh $experiment_path $gpu_fold_c"
sleep 2
send_tmux_job $session "sh $experiment_path $gpu_fold_d"
tmux attach -t $session

#tmux kill-session -t s1
#tmux kill-session
