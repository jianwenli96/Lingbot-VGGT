#!/bin/bash
# Batch inference on validation set and compute average Euclidean distance
#
# Usage:
#   bash script/run_batch_inference_val.sh              # Process all episodes
#   bash script/run_batch_inference_val.sh 0:10         # Process episodes 0-9
#   bash script/run_batch_inference_val.sh 50:100       # Process episodes 50-99

EPISODE_RANGE=${1:-}

CMD="ASCEND_RT_VISIBLE_DEVICES=6 \
LINGBOT_USE_NPU=1 \
python -m wan_va.batch_inference_val \
    --config-name tennis_i2va \
    --model-path /efs-gy1/lijianwen/Pretrained_models/lingbot-va/lingbot-vggt-base \
    --transformer-path /efs-gy1/lijianwen/Codes/lingbot-vggt/train_out/va_tennis_tasks/20260825_154208/checkpoints/checkpoint_step_5000 \
    --dataset-root /efs-gy1/lijianwen/Datasets/Tennis/tennis_val/tennis_dataset_lerobot_0818_big240mm_45steps_1000episodes_new \
    --output-dir /efs-gy1/lijianwen/Codes/lingbot-vggt/batch_outputs/tennis_batch_val_results_big \
    --sample-fps 15 \
    --prefix-num-frames 9 \
    --future-num-frames 16 \
    --timestamp-tolerance 0.03 \
    --max-camera-skew 0.03 \
    --vae-temporal-factor 4 \
    --seed 0 \
    --skip-video-save" 
    
if [ -n "$EPISODE_RANGE" ]; then
    CMD="$CMD --episode-range $EPISODE_RANGE"
    echo "Processing episode range: $EPISODE_RANGE"
else
    echo "Processing ALL episodes"
fi

eval $CMD
