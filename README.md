# MIDSim

**English** | [中文](README_zh.md)

**MIDSim** (Multi-channel **I**nformation **D**iffusion **Sim**ulator) is an LLM-driven social simulation platform for studying information diffusion across recommendation and social channels.

Built on [YuLan-OneSim](https://github.com/RUC-GSAI/YuLan-OneSim), MIDSim models how users browse feeds, post content, comment, reply, mention others, and interact with multiple recommender systems in a unified multi-round simulation environment.

## Overview

MIDSim simulates a social platform where:

- **User agents** have profiles, memory, social ties (follow / fan), and heterogeneous activity levels.
- **Recommender agents** serve different algorithms (e.g., hot, random, embedding-based, social-graph-based).
- **Information flows** through both the recommendation channel and the social channel (@ / comment / reply notifications).
- **Metrics** are collected per round (comment volume, reply structure, similarity, user behavior, etc.).

Supported scenario backbones: **Rednote**, **Twitter**, and **Weibo**.

## Requirements

- Python 3.10
- CUDA-capable GPU(s) (recommended for local vLLM inference)
- Sufficient disk space for models and simulation outputs

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/CarrieLLx/MIDSim.git
cd MIDSim
```

### 2. Configure your settings

Before starting, set up API keys, model paths, and service endpoints in:

- `config/config_rednote.json` / `config/config_twitter.json` / `config/config_weibo.json` — simulation settings
- `config/model_config.json` — LLM and embedding endpoints

Make sure `config_name` values referenced in scenario configs match entries in `model_config.json`, and that `base_url` ports align with the ports used when launching local vLLM services.

### 3. Prepare datasets

Simulation backbones are stored under the project-root `datasets/` directory. See [Datasets](#datasets) for layout and setup.

At minimum, after cloning, fetch backbone data with Git LFS:

```bash
git lfs install
git lfs pull
```

Run simulations from the **project root** so relative paths inside `env_data.json` (e.g. `datasets/rednote/embeddings/...`) resolve correctly.

### 4. Install dependencies

```bash
# Install in editable mode
pip install -e .

# Install with tuning dependencies if needed
pip install -e .[tune]

```

## Model Services

Local model scripts live under `scripts/model/`. See [scripts/README.md](scripts/README.md) for full documentation (`launch_llm.sh`, `launch_all_llm.sh`, `embedding_vllm_setup.sh`, etc.).

### Start models

Example: launch a chat LLM on port 9889 (GPU 0) and an embedding model on port 9890 (GPU 0):

```bash
bash scripts/model/launch_llm.sh 9889 0 /root/autodl-tmp/models/Qwen2.5-14B-Instruct

bash scripts/model/embedding_vllm_setup.sh -m /root/autodl-tmp/models/bge-base-zh-v1.5 -g 0
```

Update model paths and ports to match your environment, then reflect the same ports in `config/model_config.json`.

### Stop models

```bash
pkill -9 -f VLLM
```

## Quick Start

Choose a scenario and run the simulation:

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

### Scenario reference

| Config | Environment | Description |
|--------|-------------|-------------|
| `config/config_rednote.json` | `midsim_rednote` | Rednote backbone |
| `config/config_twitter.json` | `midsim_twitter` | Twitter backbone |
| `config/config_weibo.json` | `midsim_weibo` | Weibo backbone |

### Inspect outputs

Simulation artifacts are written under:

```
src/envs/<scenario_name>/runs/<timestamp>/
├── metrics_plots/     # per-step metrics and scene-level plots data
├── datasets/          # per-step runtime snapshots (see note below)
└── log/               # runtime logs
```

The `datasets/` folder inside a run directory holds **simulation outputs** (e.g. `step_1/`, `step_2/` content-pool snapshots). Do not confuse it with the project-root **`datasets/` input backbones** described below.

Large run directories are ignored by git via `.gitignore`.

## Datasets

MIDSim uses two different `datasets` paths:

| Path | Role |
|------|------|
| `datasets/` (project root) | **Input** — static backbone data for each platform (posts, users, embeddings) |
| `src/envs/<env>/runs/<timestamp>/datasets/` | **Output** — per-round snapshots exported during a run |

The root `datasets/` directory contains platform backbones (large `UserAgent.json` files are tracked via Git LFS). Clone with `git lfs pull` to fetch them.

### Directory layout

```
datasets/
├── rednote/
│   ├── env_data.json                              # seed posts / content pool
│   ├── UserAgent.json                             # user profiles & social graph
│   ├── RecommenderAgent.json                      # recommender configs & stats
│   └── embeddings/
│       └── bge-base-zh-v1.5_embeddings.json       # precomputed post embeddings
├── twitter/
│   └── ...
└── weibo/
    └── ...
```

Each platform directory contains the same four components:

| File | Description |
|------|-------------|
| `env_data.json` | Initial simulation state: `content_pool` (seed posts/tweets/notes), timestamps, and `reference_embedding_path` for similarity metrics |
| `UserAgent.json` | User agent profiles (nickname, interests, activity level, follow/fan IDs, etc.) |
| `RecommenderAgent.json` | Recommender agent definitions (algorithm type, limits, platform-specific ranking statistics) |
| `embeddings/bge-base-zh-v1.5_embeddings.json` | Precomputed BGE embeddings keyed by post ID, used for comment/topic similarity metrics |

### Scenario mapping

| Platform | Dataset directory |
|----------|-------------------|
| Rednote | `datasets/rednote/` |
| Twitter | `datasets/twitter/` |
| Weibo | `datasets/weibo/` |

Each directory is self-contained: agent profiles, initial content pool, and precomputed embeddings all live here. The `reference_embedding_path` field inside `env_data.json` should point to the embedding file under the same platform folder, e.g. `datasets/rednote/embeddings/bge-base-zh-v1.5_embeddings.json`.

Default agent counts in `config/config_*.json`:

| Scenario | UserAgent | RecommenderAgent |
|----------|-----------|------------------|
| Rednote | 1476 | 7 |
| Twitter | 1067 | 3 |
| Weibo | 130 | 7 |

### Setup checklist

```bash
# Example: prepare Rednote backbone under the project root
mkdir -p datasets/rednote/embeddings

# Copy or download backbone files into datasets/
# cp /path/to/{env_data.json,UserAgent.json,RecommenderAgent.json} datasets/rednote/
# cp /path/to/bge-base-zh-v1.5_embeddings.json datasets/rednote/embeddings/
```

If you publish or download data from Hugging Face, the repo includes `hfd.sh` as a generic download helper:

```bash
bash hfd.sh <ORG/REPO> --dataset --local-dir datasets/<platform>
```

After placing files, verify that `reference_embedding_path` in `datasets/<platform>/env_data.json` uses a **repo-root-relative** path.

## Project Structure

```
MIDSim/
├── config/                          # Simulation and model configs
│   ├── config_rednote.json
│   ├── config_twitter.json
│   ├── config_weibo.json
│   └── model_config.json            # LLM / embedding endpoints
├── datasets/                        # Input backbones (see Datasets section)
│   ├── rednote/
│   ├── twitter/
│   └── weibo/
├── scripts/
│   ├── model/                       # vLLM launch / kill helpers
│   └── README.md                    # Script usage notes
├── src/
│   ├── onesim/                      # Core simulation framework (YuLan-OneSim)
│   └── envs/
│       ├── midsim_rednote/
│       ├── midsim_twitter/
│       └── midsim_weibo/
│           └── code/                # SimEnv, UserAgent, RecommenderAgent, ...
├── hfd.sh                           # Hugging Face download helper
├── setup.py
├── README.md                        # English docs
├── README_zh.md                     # Chinese docs
└── ...
```

## Configuration Notes

### Simulation (`config/config_*.json`)

Typical fields:

- `simulator.environment.max_steps`: number of simulation rounds
- `simulator.environment.interval`: simulated time between rounds (seconds)
- `agent.profile`: agent counts and profile/schema paths
- `agent.memory`: short-term / long-term memory settings

### Models (`config/model_config.json`)

- `chat`: LLM backends used by user and recommender agents
- `embedding`: embedding model for memory retrieval and similarity

Replace placeholder API keys and local paths before running.

## Acknowledgments

MIDSim is developed on top of [YuLan-OneSim](https://github.com/RUC-GSAI/YuLan-OneSim). If you use the underlying framework, please also cite the YuLan-OneSim paper:

```bibtex
@misc{wang2025yulanonesimgenerationsocialsimulator,
  title={YuLan-OneSim: Towards the Next Generation of Social Simulator with Large Language Models},
  author={Lei Wang and Heyang Gao and Xiaohe Bo and Xu Chen and Ji-Rong Wen},
  year={2025},
  eprint={2505.07581},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2505.07581}
}
```

## License

This project inherits the [Apache-2.0 License](LICENSE) from YuLan-OneSim.
