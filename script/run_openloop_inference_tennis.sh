ASCEND_RT_VISIBLE_DEVICES=4 \
LINGBOT_USE_NPU=1 \
python -m wan_va.tennis.openloop_inference_tennis \
    --config-name tennis_i2va \
    --model-path /efs-gy1/lijianwen/Pretrained_models/lingbot-va/lingbot-vggt-base \
    --transformer-path /efs-gy1/lijianwen/Codes/lingbot-vggt/train_out/va_tennis_tasks/20260825_154208/checkpoints/checkpoint_step_5000 \
    --dataset-root /efs-gy1/lijianwen/Datasets/Tennis/tennis_val/tennis_dataset_lerobot_0818_small65mm_45steps_1000episodes_new \
    --episode-index 1 \
    --sample-fps 15 \
    --prefix-num-frames 9 \
    --future-num-frames 16 \
    --timestamp-tolerance 0.03 \
    --max-camera-skew 0.03 \
    --output-fps 15 \
    --save-root /efs-gy1/lijianwen/Codes/lingbot-vggt/outputs/tennis_batch_val_results_small \
    --plot-trajectory