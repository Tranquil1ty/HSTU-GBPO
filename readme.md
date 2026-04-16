# HSTU for RL-based Recommendation

基于 Benchmark-v2 中的 HSTU 模型，适配 GBPO 的 LOO 数据格式和强化学习训练流程。

## 目录结构

```
HSTU/
├── model.py          # HSTUBlock + HSTUForRL (GBPO 兼容接口)
├── trainer.py        # 训练器 (SFT + 9种RL算法)
├── RL/               # 强化学习模块 (GRPO/GBPO/GUPO/GSPO/GPPO/DUAL_PPO/SAPO/DAPO/DPO)
├── DIN/functions/    # DIN 奖励模型
├── script/
│   ├── main.py           # 入口脚本
│   ├── run.sh            # 训练启动脚本 (torchrun 多卡)
│   └── data_process.sh   # 数据预处理脚本
└── data/
    ├── raw_data/         # 原始 JSON 数据 (Beauty, Sports, Toys)
    ├── processed_data/   # 处理后的 LOO .npy 文件
    └── process/          # 数据处理脚本
        └── generate_sasrec_files.py
```

## 数据准备

项目自带原始数据，首次使用需要运行预处理脚本生成 LOO 格式的训练数据。

### 原始数据

`data/raw_data/{Dataset}/` 下包含：
- `{Dataset}.inter.json` — 用户交互序列 `{"user_id": [item_id1, item_id2, ...]}`
- `{Dataset}.item.json` — 物品元数据

支持的数据集：Beauty, Sports, Toys

### 生成 LOO 数据

```bash
cd HSTU/script

# 处理全部数据集
bash data_process.sh

# 只处理某个数据集
bash data_process.sh Beauty
```

脚本会在 `data/processed_data/{Dataset}/processed_loo/` 下生成：
- `train_x_loo.npy`, `train_y_loo.npy` — 训练集（滑动窗口）
- `valid_x_loo.npy`, `valid_y_loo.npy` — 验证集（倒数第二个 item）
- `test_x_loo.npy`, `test_y_loo.npy` — 测试集（最后一个 item）

已处理过的数据集会自动跳过，不会重复生成。

## 快速开始

所有命令在 `script/` 目录下执行：

```bash
cd HSTU/script

# 1. 数据预处理（首次必须执行）
bash data_process.sh

# 2. SFT 训练
bash run.sh Beauty SFT torchrun

# 3. RL 训练（需先完成 SFT）
bash run.sh Beauty GBPO torchrun
```

## 启动方式详解

### SFT 训练

```bash
# 前台运行 (Beauty 数据集)
bash run.sh Beauty SFT torchrun

# 后台运行
bash run.sh Beauty SFT nohup

# 其他数据集
bash run.sh Sports SFT torchrun
bash run.sh Toys SFT torchrun
```

### GBPO 强化学习训练

```bash
# 前台运行 (Off-Policy, rollout 策略)
bash run.sh Beauty GBPO torchrun

# 后台运行
bash run.sh Beauty GBPO nohup

# On-Policy 模式
bash run.sh Beauty GBPO torchrun true

# 指定 beam 策略, reward_topk=5, 32个候选
bash run.sh Beauty GBPO torchrun false false beam 5 32
```

### 其他 RL 算法

```bash
bash run.sh Beauty GRPO torchrun
bash run.sh Beauty GUPO torchrun
bash run.sh Beauty GSPO torchrun
bash run.sh Beauty GPPO torchrun
bash run.sh Beauty DUAL_PPO torchrun
bash run.sh Beauty SAPO torchrun
bash run.sh Beauty DAPO torchrun
bash run.sh Beauty DPO torchrun
```

### run.sh 参数说明

```
bash run.sh <DATASET> <MODE> <RUN_TYPE> <ON_POLICY> <VERBOSE> <GEN_STRATEGY> <REWARD_TOPK> <NUM_CANDIDATES> [<unused>] <HIDDEN_SIZE> <NUM_HEADS> <NUM_LAYERS>
```

| 位置 | 参数 | 默认值 | 说明 |
|------|------|--------|------|
| $1 | DATASET | Beauty | 数据集: Beauty / Sports / Toys |
| $2 | MODE | SFT | 模式: SFT / GRPO / GBPO / GUPO / GSPO / GPPO / DUAL_PPO / SAPO / DAPO / DPO |
| $3 | RUN_TYPE | nohup | 运行方式: torchrun (前台) / nohup (后台) |
| $4 | ON_POLICY | false | 是否使用 On-Policy |
| $5 | VERBOSE | false | 是否打印梯度范数 |
| $6 | GEN_STRATEGY | rollout | 生成策略: rollout / beam |
| $7 | REWARD_TOPK | 5 | Reward 阈值个数 |
| $8 | NUM_CANDIDATES | 32 | 候选数量 K |
| $10 | HIDDEN_SIZE | 128 | HSTU 隐藏层维度 |
| $11 | NUM_HEADS | 2 | 注意力头数 |
| $12 | NUM_LAYERS | 2 | HSTU Block 层数 |

### GPU 配置

默认使用 8 张 GPU。如需修改，编辑 `run.sh` 中的：
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
GPUS=8
```

## 注意事项

- RL 模式需要先完成 SFT 训练，生成 checkpoint 到 `results/{Dataset}/sft_checkpoints/best_checkpoint.pth`
- RL 模式使用 DIN 奖励模型，需要 `DIN/checkpoint/{Dataset}/best_model_fixed.pth`
- SFT 默认 batch_size=1024，RL 默认 batch_size=256
- 日志输出到 `logs_rl/{Dataset}/` 目录
