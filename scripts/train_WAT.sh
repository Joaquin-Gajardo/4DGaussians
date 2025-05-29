#!/bin/bash
set -e

run_commands() {
    local scene=$1
    local gpu=$2
    local port=$3

    export CUDA_VISIBLE_DEVICES=$gpu
    local exp="def4DGS-5"
    #python train.py -s ../../datasets/WAT/$scene --port $port --expname "../../../output/WAT/$exp/$scene" --configs arguments/WAT/default.py
    python render.py --model_path "../../output/WAT/$exp/$scene" --skip_train --configs arguments/WAT/default.py
    python metrics.py --model_path "../../output/WAT/$exp/$scene"
}

run_commands "breville" 0 6016 &
# run_commands "car_resized" 0 6016
# run_commands "community" 0 6016
# run_commands "dyson" 0 6016
# run_commands "grill_resized" 0 6016
# run_commands "kitchen" 0 6016
# run_commands "living_room" 0 6016
# run_commands "mac" 0 6016
# run_commands "ninja" 0 6016
run_commands "spa" 1 6016
# run_commands "street" 1 6016

echo "All jobs (training, rendering, and metrics) completed."