#!/bin/bash

run_commands() {
    local scene=$1
    local gpu=$2
    local port=$3

    export CUDA_VISIBLE_DEVICES=$gpu
    python train.py -s ../../datasets/WAT/$scene --port $port --expname "../../../output/WAT/$scene" --configs arguments/WAT/default.py
    python render.py --model_path "../../output/WAT/$scene" --skip_train --configs arguments/WAT/default.py
    python metrics.py --model_path "../../output/WAT/$scene"
}

run_commands "breville" 0 6016 &
run_commands "car_resized" 1 6017 &

wait

run_commands "community" 0 6016 &
run_commands "grill_resized" 1 6017 &

wait

run_commands "kitchen" 0 6016 &
run_commands "living_room" 1 6017 &

wait

run_commands "mac" 0 6016 &
run_commands "ninja" 1 6017 &

wait

run_commands "spa" 0 6016 &
run_commands "street" 1 6017 &

wait

echo "All jobs (training, rendering, and metrics) completed."