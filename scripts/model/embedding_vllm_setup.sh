#!/bin/bash

set -e

# Default values for model path, port, and GPU
model_name_or_path="openai/gpt-3.5-turbo"  # Default model path (can be replaced with a different model)
port=9890
gpu_id=0

# Parsing command line arguments
while getopts "m:p:g:" flag; do
    case "${flag}" in
        m) model_name_or_path=${OPTARG};;  # Set the model path
        p) port=${OPTARG};;  # Set the port
        g) gpu_id=${OPTARG};;  # Set the GPU ID (e.g. 0 for GPU 0)
        *) echo "Usage: $0 [-m model_name_or_path] [-p port] [-g gpu_id]" && exit 1;;
    esac
done

# Set the GPU ID for CUDA
export CUDA_VISIBLE_DEVICES=$gpu_id
# Set environment variables to avoid conflicts and limit threads
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Log file path
current_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${current_dir}/vllm_embedding.log"

# Write startup information to the log file
{
    echo "===== [Launching vLLM Embedding API Server] ====="
    echo "Time: $(date)"
    echo "Model Path: $model_name_or_path"
    echo "Port: $port"
    echo "GPU ID: $CUDA_VISIBLE_DEVICES"
    echo "==================================="
} >> "$LOG_FILE" 2>&1

# Check if the specified port is already in use
if lsof -i ":$port" > /dev/null 2>&1; then
    echo "❌ Error: Port $port is already in use. Aborting." | tee -a "$LOG_FILE"
    exit 1
fi

# Build the command to start the vLLM server with embedding task
# 使用已安装 vllm 的 Python（直接使用 env 的 python 路径，避免 conda run 子进程用到 /usr/bin/python）
if [ -n "$PYTHON" ]; then
    PYTHON_CMD="$PYTHON"
elif [ -n "$CONDA_PREFIX" ]; then
    PYTHON_CMD="$CONDA_PREFIX/bin/python"
elif [ -f "/home/liulexi/workspace/envs/env_yulan/bin/python" ]; then
    PYTHON_CMD="/home/liulexi/workspace/envs/env_yulan/bin/python"
else
    PYTHON_CMD="python3"
fi
cmd="\"$PYTHON_CMD\" -m vllm.entrypoints.openai.api_server"
cmd+=" --model \"$model_name_or_path\""
cmd+=" --port \"$port\""
cmd+=" --dtype auto"  # Automatically determine data type
cmd+=" --pipeline-parallel-size 1"  # Disable pipeline parallelism for single-device use
cmd+=" --trust-remote-code"  # Trust remote code for custom models
cmd+=" --seed 42"  # Set random seed for reproducibility
# vLLM 0.18 + 当前 torch 组合下，编译/图捕获可能触发 FakeTensorMode 相关异常；强制 eager 更稳。
cmd+=" --enforce-eager"
cmd+=" --gpu-memory-utilization 0.9"  # Keep lower utilization for shared GPU scenarios

# Start the server and log the command
echo "Starting server with command:" >> "$LOG_FILE"
echo "$cmd" >> "$LOG_FILE"
eval "$cmd" >> "$LOG_FILE" 2>&1 &
launcher_pid=$!

# Wait for the server to initialize
sleep 5

# Capture the actual server PID
server_pid=$(pgrep -P "$launcher_pid" -f "vllm.entrypoints.openai.api_server" | head -n 1)

if [ -z "$server_pid" ]; then
    echo "❌ Error: vLLM Server process not found. Check log: $LOG_FILE" | tee -a "$LOG_FILE"
    exit 1
fi

# Save the actual API server PID
echo "$server_pid" >> "$current_dir/vllm_server.pid"

# Success message
echo "✅ vLLM Embedding API server is running on port $port with PID $server_pid" >> "$LOG_FILE"
echo "Started successfully! (Log: $LOG_FILE, PID file: $current_dir/vllm_embedding.pid)"
