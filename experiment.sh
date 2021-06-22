#!/bin/bash
# bash experiment.sh ./ HopperController-Exact-v0 1

EXP_DIR=${1:-./}
TASK=${2:-HopperController-Exact-v0}
NUM_TRIALS_PER_GPU=${3:-1}
IFS=, read -ra DEVICES <<< "$CUDA_VISIBLE_DEVICES"

for DEVICE in "${DEVICES[@]}"; do
    for TRIAL in $(seq $NUM_TRIALS_PER_GPU); do

        CUDA_VISIBLE_DEVICES=$DEVICE coms \
            --logging-dir $EXP_DIR/COMs-$TASK-$DEVICE-$TRIAL \
            --task $TASK \
            --task-relabel false \
            --normalize-ys true \
            --normalize-xs true \
            --in-latent-space false \
            --vae-hidden-size 64 \
            --vae-latent-size 256 \
            --vae-activation relu \
            --vae-kernel-size 3 \
            --vae-num-blocks 4 \
            --vae-lr 0.0003 \
            --vae-beta 1.0 \
            --vae-batch-size 32 \
            --vae-val-size 500 \
            --vae-epochs 10 \
            --particle-lr 0.05 \
            --particle-gradient-steps 50 \
            --particle-entropy-coefficient 0.0 \
            --forward-model-activations relu \
            --forward-model-activations relu \
            --forward-model-hidden-size 2048 \
            --forward-model-final-tanh false \
            --forward-model-lr 0.0003 \
            --forward-model-alpha 1.0 \
            --forward-model-alpha-lr 0.01 \
            --forward-model-overestimation-limit 0.5 \
            --forward-model-noise-std 0.0 \
            --forward-model-batch-size 32 \
            --forward-model-val-size 500 \
            --forward-model-epochs 50 \
            --evaluation-samples 128 & done; done; wait
