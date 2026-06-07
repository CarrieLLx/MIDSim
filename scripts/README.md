# MIDSim Scripts Guide

This directory contains helper scripts for running MIDSim simulations and local model services.

## Run Simulations

MIDSim scenarios are launched via `yulan-onesim-cli`. Example commands (run from the project root):

```bash
export LOGURU_LEVEL=INFO

# Rednote
yulan-onesim-cli \
  --config config/config_rednote.json \
  --model_config config/model_config.json \
  --mode single \
  --env midsim_rednote

# Twitter
yulan-onesim-cli \
  --config config/config_twitter.json \
  --model_config config/model_config.json \
  --mode single \
  --env midsim_twitter

# Weibo
yulan-onesim-cli \
  --config config/config_weibo.json \
  --model_config config/model_config.json \
  --mode single \
  --env midsim_weibo
```

### `run.sh`

Legacy wrapper that calls `src/main.py` directly. Prefer the `yulan-onesim-cli` commands above unless you maintain a custom entry point.

## Distributed Run Scripts

### `distributed/distributed.sh`

Start MIDSim in distributed mode (one master node and multiple worker nodes).

```bash
bash scripts/distributed/distributed.sh [options]
```

**Options:**
- `-a, --address MASTER_ADDRESS` — Master node address (default: `127.0.0.1`)
- `-p, --port MASTER_PORT` — Master node port (default: `10051`)
- `-w, --workers NUM_WORKERS` — Number of worker nodes (default: `2`)
- `-c, --config CONFIG_PATH` — Scenario config path (default: `config/config.json`)
- `-m, --model MODEL_CONFIG_PATH` — Model config path (default: `config/model_config.json`)
- `-h, --help` — Show help

Logs are saved under `logs/`.

### `distributed/kill_distributed.sh`

Stop all processes started in distributed mode.

```bash
bash scripts/distributed/kill_distributed.sh
```

## Model Service Scripts

Local LLM and embedding services are managed under `scripts/model/`. Ensure ports match `config/model_config.json` (`base_url`).

### Quick start (example)

```bash
# Chat LLM on port 9889, GPU 0
bash scripts/model/launch_llm.sh 9889 0 /root/autodl-tmp/models/Qwen2.5-14B-Instruct

# Embedding model on port 9890 (default), GPU 0
bash scripts/model/embedding_vllm_setup.sh -m /root/autodl-tmp/models/bge-base-zh-v1.5 -g 0
```

### `model/launch_llm.sh`

Start a single chat LLM server with vLLM.

```bash
bash scripts/model/launch_llm.sh <port> <GPU_ID> <model_path> [lora_dir]
```

**Parameters:**
- `port` — API server port
- `GPU_ID` — GPU to use
- `model_path` — Path to the LLM weights
- `lora_dir` — (Optional) LoRA adapter directory

Optional environment variables (see script header for details): `LAUNCH_LLM_ENFORCE_EAGER`, `LAUNCH_LLM_TP`, `LAUNCH_LLM_GPU_MEM_UTIL`, `LAUNCH_LLM_MAX_MODEL_LEN`, `LAUNCH_LLM_SERVED_MODEL_NAME`, `PYTHON`.

### `model/launch_all_llm.sh`

Launch multiple LLM servers on different GPUs.

```bash
bash scripts/model/launch_all_llm.sh
```

By default uses ports `9881`–`9888` and GPUs `0`–`7`. Edit `port_list`, `gpu_list`, and `model_path` in the script to customize.

### `model/embedding_vllm_setup.sh`

Start a vLLM embedding service.

```bash
bash scripts/model/embedding_vllm_setup.sh [-m model_path] [-p port] [-g gpu_id]
```

**Options:**
- `-m` — Model directory (e.g. BGE embedding model)
- `-p` — Service port (default: `9890`)
- `-g` — GPU ID (default: `0`)

### Stop model services

```bash
# Stop vLLM processes
pkill -9 -f VLLM

```