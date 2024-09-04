#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0 && python train.py -s ../../datasets/WAT/breville --port 6016 --expname "../../../output/WAT/breville" --configs arguments/WAT/default.py &
# export CUDA_VISIBLE_DEVICES=1 && python train.py -s ../../datasets/WAT/car_resized --port 6017 --expname "../../../output/WAT/car_resized" --configs arguments/WAT/default.py &
# wait
# export CUDA_VISIBLE_DEVICES=0 && python train.py -s ../../datasets/WAT/community --port 6016 --expname "../../../output/WAT/community" --configs arguments/WAT/default.py &
# export CUDA_VISIBLE_DEVICES=1 && python train.py -s ../../datasets/WAT/grill_resized --port 6017 --expname "../../../output/WAT/grill_resized" --configs arguments/WAT/default.py &
# wait
# export CUDA_VISIBLE_DEVICES=0 && python train.py -s ../../datasets/WAT/kitchen --port 6016 --expname "../../../output/WAT/kitchen" --configs arguments/WAT/default.py &
# export CUDA_VISIBLE_DEVICES=1 && python train.py -s ../../datasets/WAT/living_room --port 6017 --expname "../../../output/WAT/living_room" --configs arguments/WAT/default.py &
# wait
# export CUDA_VISIBLE_DEVICES=0 && python train.py -s ../../datasets/WAT/mac --port 6016 --expname "../../../output/WAT/mac" --configs arguments/WAT/default.py &
# export CUDA_VISIBLE_DEVICES=1 && python train.py -s ../../datasets/WAT/ninja --port 6017 --expname "../../../output/WAT/ninja" --configs arguments/WAT/default.py &
# wait
# export CUDA_VISIBLE_DEVICES=0 && python train.py -s ../../datasets/WAT/spa --port 6016 --expname "../../../output/WAT/spa" --configs arguments/WAT/default.py &
# export CUDA_VISIBLE_DEVICES=1 && python train.py -s ../../datasets/WAT/street --port 6017 --expname "../../../output/WAT/street" --configs arguments/WAT/default.py &
# wait
# echo "All training jobs completed."

# echo "Rendering all scenes"

run_commands() {
    local scene=$1
    local gpu=$2
    local port=$3

    export CUDA_VISIBLE_DEVICES=$gpu
    python train.py -s ../../datasets/WAT/$scene --port $port --expname "../../../output/WAT/$scene" --configs arguments/WAT/default.py
    python render.py --model_path "../../output/WAT/$scene" --skip_train --configs arguments/WAT/default.py
    python metrics.py --model_path "../../output/WAT/$scene"
}

# run_commands "breville" 0 6016 &
# run_commands "car_resized" 1 6017 &

# wait

# run_commands "community" 0 6018 &
# run_commands "grill_resized" 1 6019 &

# wait

run_commands "kitchen" 0 6020 &
run_commands "living_room" 1 6021 &

wait

run_commands "mac" 0 6022 &
run_commands "ninja" 1 6023 &

wait

run_commands "spa" 0 6024 &
run_commands "street" 1 6025 &

wait

echo "All jobs (training, rendering, and metrics) completed."