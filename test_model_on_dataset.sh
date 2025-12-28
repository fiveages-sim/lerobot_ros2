#!/bin/bash
# 测试模型在训练数据上的表现

echo "=========================================="
echo "测试 act_all_down30 在训练数据上的误差"
echo "=========================================="

CUDA_VISIBLE_DEVICES=0 /home/king/miniconda3/envs/lerobot_ros2_act/bin/python \
  /home/king/lerobot_ros2/algorithms/infer_act.py \
  /home/king/lerobot_ros2/outputs/act_all_down30/checkpoints/last \
  --dataset /home/king/lerobot_ros2/dataset/down_dataset_30 \
  --num-episodes 5 \
  --compare-actions
