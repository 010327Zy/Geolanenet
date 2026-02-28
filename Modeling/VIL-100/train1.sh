#!/bin/bash

# 定义日志路径
LOG_DIR="/train1/gzp/Geolanenet/GeoLaneNet/Modeling/VIL-100/output"
mkdir -p $LOG_DIR


export CUDA_VISIBLE_DEVICES=3


# Step 1: 运行第一个训练代码
# echo "Starting first training script: ILD_seg"
# cd /train1/gzp/Geolanenet/GeoLaneNet/Modeling/VIL-100/ILD_seg/code
# nohup python -u main.py > $LOG_DIR/ILD_seg.log 2>&1 &
# wait # 等待第一个训练任务完成

# # Step 2: 运行第二个训练代码
# echo "Starting second training script: ILD_coeff"
# cd /train1/gzp/Geolanenet/GeoLaneNet/Modeling/VIL-100/ILD_coeff/code
# nohup python -u main.py > $LOG_DIR/ILD_coeff.log 2>&1 &
# wait # 等待第二个训练任务完成

# Step 3: 运行第三个训练代码
echo "Starting third training script: PLD"
cd /train1/gzp/Geolanenet/GeoLaneNet/Modeling/VIL-100/PLD/code
nohup python -u main.py > $LOG_DIR/PLD.log 2>&1 &
wait # 等待第三个训练任务完成

echo "All training tasks are completed!"
