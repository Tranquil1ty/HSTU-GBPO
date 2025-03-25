#!/bin/bash

# 检查是否提供了参数
if [ $# -eq 0 ]; then
    echo "Usage: \$0 <filename>"
    exit 1
fi

# 获取文件名参数
FILENAME=$1

# 执行第一个 Python 命令
echo "Executing: python3 $FILENAME --mode predict --no_kai_v2"
python3 $FILENAME --mode predict --no_kai_v2

# 检查第一个命令是否成功
if [ $? -ne 0 ]; then
    echo "Error: Failed to execute python3 $FILENAME --mode predict --no_kai_v2"
    exit 1
fi

# 执行第二个 Python 命令
echo "Executing: python3 -u check_infer_yaml_v2.py dnn-plugin.yaml predict/config/dnn_model.yaml"
python3 -u check_infer_yaml_v2.py dnn-plugin.yaml predict/config/dnn_model.yaml

# 检查第二个命令是否成功
if [ $? -ne 0 ]; then
    echo "Error: Failed to execute python3 -u check_infer_yaml_v2.py dnn-plugin.yaml predict/config/dnn_model.yaml"
    exit 1
fi

echo "Both commands executed successfully."
