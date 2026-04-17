#!/bin/bash

# --- 1. Runtime Environment ---
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUBLAS_WORKSPACE_CONFIG=:4096:8
GPUS=8
PORT=29502
SCRIPT="./main.py"

# --- 2. Basic Configuration ---
DATASET=${1:-"Beauty"}       # Dataset: Beauty, Sports, Toys
MODE=${2:-"SFT"}             # Mode: SFT/GRPO/GBPO/GUPO/GSPO/GPPO/DUAL_PPO/SAPO/DAPO/DPO
RUN_TYPE=${3:-"nohup"}       # Run type: torchrun (foreground) / nohup (background)
BASE_PATH=".."

# --- 3. HSTU Architecture ---
HIDDEN_SIZE=${10:-"128"}
NUM_HEADS=${11:-"2"}
NUM_LAYERS=${12:-"2"}
MAX_SEQ_LEN="20"
SEED=${13:-"42"}
export PYTHONHASHSEED=$SEED

# --- 4. Build Python Args ---
PY_ARGS="--dataset $DATASET --mode $MODE --base_path $BASE_PATH --seed $SEED"
PY_ARGS="$PY_ARGS --hidden_size $HIDDEN_SIZE --num_heads $NUM_HEADS --num_layers $NUM_LAYERS --max_seq_len $MAX_SEQ_LEN"

BATCH_SIZE=128
LR="1e-3"
if [ "$MODE" == "SFT" ]; then
    BATCH_SIZE=1024
    LR="1e-3"
fi

PY_ARGS="$PY_ARGS --batch_size $BATCH_SIZE --learning_rate $LR --epochs 500"

# Optional switches (positional args 4-9)
IS_ON_POLICY=${4:-"false"}
IS_VERBOSE=${5:-"false"}
GEN_STRATEGY=${6:-"rollout"}
REWARD_TOPK=${7:-"5"}
NUM_CANDIDATES=${8:-"32"}

PY_ARGS="$PY_ARGS --num_candidates $NUM_CANDIDATES --rollout_batch_size $NUM_CANDIDATES --use_bf16"

[ "$IS_ON_POLICY" == "true" ] && PY_ARGS="$PY_ARGS --on_policy"
PY_ARGS="$PY_ARGS --verbose $IS_VERBOSE"
PY_ARGS="$PY_ARGS --reward_topk $REWARD_TOPK"
if [ "$MODE" != "SFT" ]; then
    PY_ARGS="$PY_ARGS --gen_strategy $GEN_STRATEGY --dropout 0.2"
fi

# --- 5. Log Path ---
create_log_path() {
    local script_path="$1"
    local mode="$2"
    local on_policy="$3"
    local gen_strategy="$4"
    local reward_topk="$5"
    local num_candidates="$6"

    local abs_script_path=$(realpath "$script_path")
    local script_dir=$(dirname "$abs_script_path")
    local parent_dir=$(dirname "$script_dir")
    local logs_dir="$parent_dir/logs_rl"
    local script_name=$(basename "$script_path" .py)

    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local policy_str=""
    if [[ "$mode" != "SFT" ]]; then
        if [ "$on_policy" == "true" ]; then
            policy_str="_OnPolicy"
        else
            policy_str="_OffPolicy"
        fi
        policy_str="${policy_str}_${gen_strategy}_topk${reward_topk}_K${num_candidates}"
    fi

    local log_file="$logs_dir/${DATASET}/${script_name}_${mode}${policy_str}_${timestamp}.log"
    mkdir -p "$(dirname "$log_file")"
    echo "$log_file"
}

# --- 6. Launch ---
if [ "$RUN_TYPE" == "nohup" ]; then
    LOG_FILE=$(create_log_path "$SCRIPT" "$MODE" "$IS_ON_POLICY" "$GEN_STRATEGY" "$REWARD_TOPK" "$NUM_CANDIDATES")

    echo "Starting background training..."
    echo "Log: tail -f $LOG_FILE"

    nohup torchrun --nproc_per_node=$GPUS --master_port=$PORT $SCRIPT $PY_ARGS > "$LOG_FILE" 2>&1 &
    echo "PID: $!"
else
    echo "Starting foreground training..."
    torchrun --nproc_per_node=$GPUS --master_port=$PORT $SCRIPT $PY_ARGS
fi
