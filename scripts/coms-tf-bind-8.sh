#!/bin/bash
NUM_TRIALS_PER_GPU=2
IFS=, read -ra DEVICES <<< "$CUDA_VISIBLE_DEVICES"
for DEVICE in "${DEVICES[@]}"; do
    for TRIAL in $(seq $NUM_TRIALS_PER_GPU); do
        CUDA_VISIBLE_DEVICES=$DEVICE coms \
            --logging-dir ./COMs-TFBind8-Exact-v0-$DEVICE-$TRIAL-$RANDOM \
            --task TFBind8-Exact-v0 \
            --no-task-relabel \
            --normalize-ys \
            --no-normalize-xs \
            --not-in-latent-space \
            --vae-hidden-size 64 \
            --vae-latent-size 256 \
            --vae-activation relu \
            --vae-kernel-size 3 \
            --vae-num-blocks 4 \
            --vae-lr 0.0003 \
            --vae-beta 1.0 \
            --vae-batch-size 128 \
            --vae-val-size 500 \
            --vae-epochs 10 \
            --particle-lr 0.05 \
            --particle-train-gradient-steps 50 \
            --particle-evaluate-gradient-steps 50 \
            --particle-entropy-coefficient 0.0 \
            --forward-model-activations relu \
            --forward-model-activations relu \
            --forward-model-hidden-size 2048 \
            --no-forward-model-final-tanh \
            --forward-model-lr 0.0003 \
            --forward-model-alpha 0.0 \
            --forward-model-alpha-lr 0.0 \
            --forward-model-overestimation-limit 0.5 \
            --forward-model-noise-std 0.0 \
            --forward-model-batch-size 128 \
            --forward-model-val-size 500 \
            --forward-model-epochs 50 \
            --evaluation-samples 128 & done; done; wait
