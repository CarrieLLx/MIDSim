#!/bin/bash

# 获取当前脚本所在目录
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 配置端口列表和GPU ID列表
port_list=(9888 9889 9891)
gpu_list=(0 1 2)
model_path="models/Qwen2.5-14B-Instruct"

# 启动每个服务器
for i in "${!port_list[@]}"; do
    port="${port_list[$i]}"
    gpu_id="${gpu_list[$i]}"
    bash "$script_dir/launch_llm.sh" "$port" "$gpu_id" "$model_path" &
done

wait
echo "✅ All LLM API servers have been started successfully."
