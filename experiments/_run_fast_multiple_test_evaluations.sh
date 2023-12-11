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
eval_script_path="/home/duenias/PycharmProjects/HyperNetworks/final_experiments_splitseed/_evaluate_on_test_set.sh"

project_name="HyperNetworks_final_splitseed_test"
#project_name="HyperNetworks_final_splitseed"
#project_name="HyperNets_imgNtabular"

#experiment_base_name="DAFT_cw11d14"
#experiment_base_name="Film_cw11d14"
#experiment_base_name="baseline-concat1_cw11_08_14"
#experiment_base_name="TabularAsHyper_embd_trainedTabular8_cw11d14"
#experiment_base_name="baseline-tabular_embd8"


GPU="0"
for features_set in 15; do
  for split_seed in 0 1 2; do
    for version in '_v1' '_v2' '_v3'; do
      experiment_name="$experiment_base_name$version-seed$split_seed-fs$features_set"
#      cmd="echo $eval_script_path $GPU $project_name $experiment_name $features_set $split_seed $version"
      cmd="sh $eval_script_path $GPU $project_name $experiment_name $features_set $split_seed $version"
      send_tmux_job $session "$cmd"
    done
  done
done
tmux attach -t $session
