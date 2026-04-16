#!/bin/bash

# 数据预处理脚本：将原始 JSON 交互数据转换为 LOO 格式的 .npy 文件
# 用法: bash data_process.sh [DATASET]
# 示例: bash data_process.sh Beauty
#       bash data_process.sh          # 默认处理全部三个数据集

BASE_PATH=".."
PROCESS_SCRIPT="${BASE_PATH}/data/process/generate_sasrec_files.py"
MAX_SEQ_LEN=20

# 如果指定了数据集，只处理该数据集；否则处理全部
if [ -n "$1" ]; then
    DATASETS=("$1")
else
    DATASETS=("Beauty" "Sports" "Toys")
fi

for DS in "${DATASETS[@]}"; do
    INTER_FILE="${BASE_PATH}/data/raw_data/${DS}/${DS}.inter.json"
    OUTPUT_DIR="${BASE_PATH}/data/processed_data/${DS}/processed_loo"

    if [ ! -f "$INTER_FILE" ]; then
        echo "[SKIP] ${DS}: raw data not found at ${INTER_FILE}"
        continue
    fi

    if [ -f "${OUTPUT_DIR}/train_x_loo.npy" ]; then
        echo "[SKIP] ${DS}: processed_loo already exists at ${OUTPUT_DIR}"
        continue
    fi

    echo "[PROCESSING] ${DS}..."
    python3 "$PROCESS_SCRIPT" \
        --inter_file "$INTER_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --max_seq_len "$MAX_SEQ_LEN"
    echo "[DONE] ${DS}"
    echo ""
done
