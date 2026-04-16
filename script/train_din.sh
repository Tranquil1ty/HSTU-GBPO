#!/bin/bash

# DIN 奖励模型训练脚本
# 用法: bash train_din.sh [DATASET] [GPU_ID]
# 示例: bash train_din.sh Beauty 0
#       bash train_din.sh                # 默认在 GPU 0,1,2 上并行训练全部数据集

BASE_DIR=".."

if [ -n "$1" ]; then
    # 单数据集模式
    DATASETS=("$1")
    GPUS=("${2:-0}")
else
    # 全部数据集并行模式
    DATASETS=("Beauty" "Sports" "Toys")
    GPUS=(0 1 2)
fi

LOG_DIR="${BASE_DIR}/logs/din"
mkdir -p "$LOG_DIR"

for i in "${!DATASETS[@]}"; do
    dataset=${DATASETS[$i]}
    gpu=${GPUS[$i]}

    DATA_DIR="${BASE_DIR}/data/processed_data/${dataset}/processed_loo"
    SAVE_DIR="${BASE_DIR}/DIN/checkpoint/${dataset}"

    # 检查 LOO 数据是否存在
    if [ ! -f "${DATA_DIR}/train_x_loo.npy" ]; then
        echo "[SKIP] ${dataset}: LOO data not found. Run 'bash data_process.sh ${dataset}' first."
        continue
    fi

    # 检查是否已有 checkpoint
    if [ -d "$SAVE_DIR" ] && [ "$(ls -A "$SAVE_DIR" 2>/dev/null)" ]; then
        echo "[SKIP] ${dataset}: DIN checkpoint already exists in ${SAVE_DIR}"
        continue
    fi

    mkdir -p "$SAVE_DIR"

    echo "[TRAINING] ${dataset} on GPU ${gpu}..."
    CUDA_VISIBLE_DEVICES=$gpu nohup python3 -u "${BASE_DIR}/DIN/functions/train_loo.py" \
        --data_dir_loo "${DATA_DIR}/" \
        --save_path "${SAVE_DIR}" \
        > "$LOG_DIR/train_din_${dataset}.log" 2>&1 &

    echo "  PID: $!, Log: tail -f $LOG_DIR/train_din_${dataset}.log"
done

echo ""
echo "All DIN training jobs launched."
