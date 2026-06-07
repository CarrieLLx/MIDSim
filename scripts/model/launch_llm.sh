#!/bin/bash

set -e

# --- 参数校验 ---
if [ $# -lt 3 ]; then
    echo "Usage: $0 <port> <gpu_id> <model_path> [lora_dir]"
    echo ""
    echo "环境变量（可选）："
    echo "  LAUNCH_LLM_ENFORCE_EAGER=1|0   默认 1。为 1 时加 --enforce-eager，绕过 torch.compile（可修复 FakeTensorMode / standalone_compile 报错）。"
    echo "  LAUNCH_LLM_TP=N               设置 --tensor-parallel-size（例：72B 单卡 80G 易 OOM，用 gpu_id=0,1 且 LAUNCH_LLM_TP=2）。"
    echo "  LAUNCH_LLM_GPU_MEM_UTIL=0.92  覆盖 --gpu-memory-utilization（默认 0.7；TP≥2 时脚本默认 0.92，避免 KV cache blocks 报满）。"
    echo "  LAUNCH_LLM_MAX_MODEL_LEN=N    --max-model-len；未设置且 TP≥2 时默认 16384（否则 vLLM 常按模型 32768 预留 KV，2×80G 上易报 KV 不足）。"
    echo "  LAUNCH_LLM_SERVED_MODEL_NAME=x  设置 --served-model-name；model_config.json 里 model_name 须与此一致（否则 OpenAI 请求 404）。"
    echo "  PYTHON=/path/to/python   指定带 vllm 的解释器。"
    exit 1
fi

# --- 参数读取 ---
port="$1"
gpuid="$2"
model_path="$3"
lora_dir="$4"
# vLLM 将 OpenAI 字段 model 与启动时的 served 名做字符串完全匹配；路径末尾 / 常与 model_config 不一致导致 404
model_path="${model_path%/}"

# --- 环境变量设置 ---
export CUDA_VISIBLE_DEVICES="$gpuid"
# vLLM 0.18+ 可能忽略 VLLM_USE_V1；若日志出现 FakeTensorMode / standalone_compile / Engine core failed，用下面 enforce-eager
LAUNCH_LLM_ENFORCE_EAGER="${LAUNCH_LLM_ENFORCE_EAGER:-1}"
# TP≥2 时若仍用 0.7，权重占满每卡后 KV block 常为 0（ValueError: No available memory for the cache blocks）
if [[ -n "${LAUNCH_LLM_GPU_MEM_UTIL:-}" ]]; then
    GPU_MEM_UTIL="${LAUNCH_LLM_GPU_MEM_UTIL}"
elif [[ -n "${LAUNCH_LLM_TP:-}" ]] && [[ "${LAUNCH_LLM_TP}" =~ ^[0-9]+$ ]] && (( LAUNCH_LLM_TP >= 2 )); then
    GPU_MEM_UTIL="0.92"
else
    GPU_MEM_UTIL="0.7"
fi
# TP≥2 且未指定时压低上下文上限：否则 vLLM 按模型 max_position_embeddings（如 32768）算 KV，72B 双卡常出现「需 5GiB KV 仅 2.55GiB 可用」
if [[ -z "${LAUNCH_LLM_MAX_MODEL_LEN:-}" ]] && [[ -n "${LAUNCH_LLM_TP:-}" ]] && [[ "${LAUNCH_LLM_TP}" =~ ^[0-9]+$ ]] && (( LAUNCH_LLM_TP >= 2 )); then
    LAUNCH_LLM_MAX_MODEL_LEN="16384"
fi
# 限制线程数，避免系统资源不足
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# 使用已安装 vllm 的 Python（未激活 conda 时 /usr/bin/python 没有 vllm 会报 ModuleNotFoundError）
if [ -n "$PYTHON" ]; then
    PYTHON_CMD="$PYTHON"
elif [ -n "$CONDA_PREFIX" ]; then
    PYTHON_CMD="$CONDA_PREFIX/bin/python"
else
    PYTHON_CMD="python"
fi

# --- 目录及日志 ---
current_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${current_dir}/llms.log"
PID_FILE="${current_dir}/llms.pid"

# --- 本地模型路径校验（HuggingFace Hub 名如 Qwen/Qwen2.5-... 不以 / 开头，跳过）---
if [[ "$model_path" == /* ]]; then
    if [[ ! -d "$model_path" ]]; then
        echo "❌ 模型目录不存在: $model_path" | tee -a "$LOG_FILE"
        exit 1
    fi
    if [[ ! -f "$model_path/config.json" ]]; then
        echo "❌ 未找到 config.json（请确认是已下载的 Transformers 模型目录）: $model_path" | tee -a "$LOG_FILE"
        exit 1
    fi
fi

# --- 写入启动日志 ---
{
    echo "===== [Launching LLM Server] ====="
    echo "Time: $(date)"
    echo "Port: $port"
    echo "GPU ID: $CUDA_VISIBLE_DEVICES"
    echo "LAUNCH_LLM_ENFORCE_EAGER: $LAUNCH_LLM_ENFORCE_EAGER"
    [ -n "${LAUNCH_LLM_TP:-}" ] && echo "LAUNCH_LLM_TP: $LAUNCH_LLM_TP"
    echo "GPU_MEM_UTIL (--gpu-memory-utilization): $GPU_MEM_UTIL"
    echo "LAUNCH_LLM_MAX_MODEL_LEN (--max-model-len): ${LAUNCH_LLM_MAX_MODEL_LEN:-<vLLM default from model>}"
    echo "Model Path: $model_path"
    [ -n "${LAUNCH_LLM_SERVED_MODEL_NAME:-}" ] && echo "LAUNCH_LLM_SERVED_MODEL_NAME: $LAUNCH_LLM_SERVED_MODEL_NAME"
    [ -n "$lora_dir" ] && echo "LoRA Directory: $lora_dir"
    echo "==================================="
} >> "$LOG_FILE" 2>&1

# --- 检查端口是否被监听（仅 LISTEN 算占用；CLOSE_WAIT 等客户端连接不阻止启动）---
if lsof -i ":$port" 2>/dev/null | grep -q LISTEN; then
    echo "❌ Error: Port $port is already in use (LISTEN). Aborting." | tee -a "$LOG_FILE"
    exit 1
fi

# --- 构建启动命令 ---
cmd="\"$PYTHON_CMD\" -m vllm.entrypoints.openai.api_server"
cmd+=" --model \"$model_path\""
if [[ -n "${LAUNCH_LLM_SERVED_MODEL_NAME:-}" ]]; then
    cmd+=" --served-model-name \"${LAUNCH_LLM_SERVED_MODEL_NAME}\""
fi
cmd+=" --port \"$port\""
cmd+=" --dtype auto"
cmd+=" --pipeline-parallel-size 1"
cmd+=" --trust-remote-code"
cmd+=" --enable-prefix-caching"
cmd+=" --tokenizer-mode auto"
cmd+=" --seed 42"
cmd+=" --disable-frontend-multiprocessing"
cmd+=" --gpu-memory-utilization ${GPU_MEM_UTIL}"
# 关闭 piecewise torch.compile / 部分 CUDA graph，避免与当前 PyTorch 组合在 profile_run 阶段崩溃（日志常见 FakeTensorMode）
if [[ "${LAUNCH_LLM_ENFORCE_EAGER}" == "1" ]] || [[ "${LAUNCH_LLM_ENFORCE_EAGER}" == "true" ]] || [[ "${LAUNCH_LLM_ENFORCE_EAGER}" == "yes" ]]; then
    cmd+=" --enforce-eager"
fi
# 多卡张量并行：CUDA_VISIBLE_DEVICES 需与 TP 一致（例：gpuid=0,1 且 LAUNCH_LLM_TP=2）
if [[ -n "${LAUNCH_LLM_TP:-}" ]]; then
    cmd+=" --tensor-parallel-size ${LAUNCH_LLM_TP}"
fi
if [[ -n "${LAUNCH_LLM_MAX_MODEL_LEN:-}" ]]; then
    cmd+=" --max-model-len ${LAUNCH_LLM_MAX_MODEL_LEN}"
fi

# --- LoRA逻辑判断 ---
if [ -n "$lora_dir" ]; then
    cmd+=" --enable-lora"
    cmd+=" --lora-modules lora=\"$lora_dir\""
elif [ -f "${model_path}/adapter_config.json" ]; then
    echo "Detected adapter_config.json, enabling built-in LoRA." >> "$LOG_FILE"
    cmd+=" --enable-lora"
fi

# --- 启动模型服务 ---
echo "Starting server with command:" >> "$LOG_FILE"
echo "$cmd" >> "$LOG_FILE"
eval "$cmd" >> "$LOG_FILE" 2>&1 &
launcher_pid=$!

# --- 等待 HTTP 就绪（大模型可能要几十秒～数分钟；仅 sleep+pgrep 会「假成功」）---
ready=0
http_code=""
for _ in $(seq 1 120); do
    http_code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 2 "http://127.0.0.1:${port}/health" 2>/dev/null || true)
    if [[ "$http_code" == "200" ]]; then
        ready=1
        break
    fi
    # /v1/models 部分版本也可用
    http_code2=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 2 "http://127.0.0.1:${port}/v1/models" 2>/dev/null || true)
    if [[ "$http_code2" == "200" ]]; then
        ready=1
        break
    fi
    if ! kill -0 "$launcher_pid" 2>/dev/null; then
        echo "❌ vLLM 启动进程已退出，请查看日志末尾:" | tee -a "$LOG_FILE"
        tail -50 "$LOG_FILE" >&2
        exit 1
    fi
    sleep 2
done

if [ "$ready" -ne 1 ]; then
    echo "❌ 在约 4 分钟内未检测到服务就绪 (HTTP 200)。可能仍在加载、显存不足或引擎崩溃。日志末尾:" | tee -a "$LOG_FILE"
    tail -60 "$LOG_FILE" >&2
    exit 1
fi

# --- 捕捉 API Server PID（用于记录；多实例时取匹配本端口的）---
server_pid=""
if command -v pgrep >/dev/null 2>&1; then
    server_pid=$(pgrep -f "vllm.entrypoints.openai.api_server.*--port.*${port}" 2>/dev/null | head -n 1 || true)
fi
if [ -z "$server_pid" ]; then
    server_pid="$launcher_pid"
fi

# --- 记录 PID ---
echo "$server_pid" >> "$PID_FILE"

# --- 成功提示 ---
echo "✅ vLLM API server 已就绪 (HTTP 200) on port $port, PID ~ $server_pid" >> "$LOG_FILE"
echo "Started successfully! (Log: $LOG_FILE, PID file: $PID_FILE)"
