# MIDSim

[English](README.md) | **中文**

**MIDSim**（Multi-channel **I**nformation **D**iffusion **Sim**ulator，多渠道信息扩散模拟器）是一个由大语言模型驱动的社交仿真平台，用于研究信息在推荐渠道与社交渠道中的扩散过程。

本项目基于 [YuLan-OneSim](https://github.com/RUC-GSAI/YuLan-OneSim) 构建，在统一的多轮仿真环境中模拟用户浏览信息流、发帖、评论、回复、@ 他人，以及与多种推荐系统交互的行为。

## 概述

MIDSim 仿真一个社交平台，其中：

- **用户智能体** 拥有画像、记忆、社交关系（关注 / 粉丝）以及差异化的活跃度。
- **推荐智能体** 提供多种算法（如热门、随机、Embedding 相似、社交图等）。
- **信息流动** 同时经过推荐渠道与社交渠道（@、评论、回复通知）。
- **指标采集** 按轮次记录评论量、回复结构、相似度、用户行为等。

当前支持的场景骨干数据：**小红书（Rednote）**、**Twitter**、**微博（Weibo）**。

## 环境要求

- Python 3.10
- 支持 CUDA 的 GPU（推荐，用于本地 vLLM 推理）
- 足够的磁盘空间（模型权重与仿真输出）

## 安装

### 1. 克隆仓库

```bash
git clone https://github.com/CarrieLLx/MIDSim.git
cd MIDSim
```

### 2. 配置

启动前，请在以下文件中设置 API Key、模型路径与服务端点：

- `config/config_rednote.json` / `config/config_twitter.json` / `config/config_weibo.json` — 仿真配置
- `config/model_config.json` — LLM 与 Embedding 服务配置

请确保场景配置中引用的 `config_name` 与 `model_config.json` 中的条目一致，且 `base_url` 端口与本地 vLLM 启动脚本使用的端口一致。

### 3. 准备 datasets

仿真所需的骨干数据存放在项目根目录的 `datasets/` 下。目录结构与说明见 [数据集](#数据集)。

请在**项目根目录**下启动仿真，以便 `env_data.json` 中的相对路径（如 `datasets/rednote/embeddings/...`）能正确解析。

### 4. 安装依赖

```bash
# 可编辑模式安装
pip install -e .

# 如需微调相关依赖
pip install -e .[tune]
```

## 模型服务

本地模型脚本位于 `scripts/model/`。完整说明见 [scripts/README.md](scripts/README.md)（`launch_llm.sh`、`launch_all_llm.sh`、`embedding_vllm_setup.sh` 等）。

### 启动模型

示例：在 GPU 0 上启动对话 LLM（端口 9889）与 Embedding 模型（端口 9890）：

```bash
bash scripts/model/launch_llm.sh 9889 0 /root/autodl-tmp/models/Qwen2.5-14B-Instruct

bash scripts/model/embedding_vllm_setup.sh -m /root/autodl-tmp/models/bge-base-zh-v1.5 -g 0
```

请根据实际环境修改模型路径与端口，并在 `config/model_config.json` 中同步更新。

### 停止模型

```bash
pkill -9 -f VLLM
```

## 快速开始

选择场景并运行仿真：

```bash
export LOGURU_LEVEL=INFO

# 小红书
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

# 微博
yulan-onesim-cli \
  --config config/config_weibo.json \
  --model_config config/model_config.json \
  --mode single \
  --env midsim_weibo
```

### 场景对照

| 配置文件 | 环境名 | 说明 |
|----------|--------|------|
| `config/config_rednote.json` | `midsim_rednote` | 小红书骨干 |
| `config/config_twitter.json` | `midsim_twitter` | Twitter 骨干 |
| `config/config_weibo.json` | `midsim_weibo` | 微博骨干 |

### 查看输出

仿真产物写入：

```
src/envs/<scenario_name>/runs/<timestamp>/
├── metrics_plots/     # 逐步指标与场景级绘图数据
├── datasets/          # 每轮运行时快照（见下方说明）
└── log/               # 运行日志
```

运行目录下的 `datasets/` 是**仿真输出**（如 `step_1/`、`step_2/` 内容池快照），请勿与项目根目录下作为**输入**的 `datasets/` 骨干数据混淆。

大型运行目录已通过 `.gitignore` 忽略。

## 数据集

MIDSim 中有两处名为 `datasets` 的路径，含义不同：

| 路径 | 作用 |
|------|------|
| `datasets/`（项目根目录） | **输入** — 各平台的静态骨干数据（帖子、用户、Embedding） |
| `src/envs/<env>/runs/<timestamp>/datasets/` | **输出** — 仿真过程中按轮导出的快照 |

根目录 `datasets/` 包含 Rednote、Twitter、Weibo 各平台的骨干数据。

### 目录结构

```
datasets/
├── rednote/
│   ├── env_data.json                              # 种子帖子 / 内容池
│   ├── UserAgent.json                             # 用户画像与社交关系
│   ├── Algorithm.json                      # 推荐器配置与统计
│   └── embeddings/
│       └── bge-base-zh-v1.5_embeddings.json       # 预计算的帖子 Embedding
├── twitter/
│   └── ...
└── weibo/
    └── ...
```

每个平台目录包含相同的四类文件：

| 文件 | 说明 |
|------|------|
| `env_data.json` | 初始仿真状态：`content_pool`（种子帖/推文/笔记）、时间戳，以及用于相似度指标的 `reference_embedding_path` |
| `UserAgent.json` | 用户智能体画像（昵称、兴趣、活跃度、关注/粉丝 ID 等） |
| `Algorithm.json` | 推荐智能体定义（算法类型、召回上限、平台相关排序统计） |
| `embeddings/bge-base-zh-v1.5_embeddings.json` | 以帖子 ID 为键的 BGE 预计算向量，用于评论/话题相似度等指标 |

### 场景与目录对应

| 平台 | 数据目录 |
|------|----------|
| 小红书（Rednote） | `datasets/rednote/` |
| Twitter | `datasets/twitter/` |
| 微博（Weibo） | `datasets/weibo/` |

每个目录自包含所需数据：智能体画像、初始内容池与预计算 Embedding 均在此目录下。`env_data.json` 中的 `reference_embedding_path` 应指向同一平台子目录下的 Embedding 文件，例如 `datasets/rednote/embeddings/bge-base-zh-v1.5_embeddings.json`。

`config/config_*.json` 中的默认智能体数量：

| 场景 | UserAgent | Algorithm |
|------|-----------|------------------|
| Rednote | 1476 | 7 |
| Twitter | 1067 | 7 |
| Weibo | 130 | 7 |

### 准备步骤

若使用独立数据镜像，也可通过仓库自带的 `hfd.sh` 从 Hugging Face 下载：

```bash
bash hfd.sh <ORG/REPO> --dataset --local-dir datasets/<platform>
```

请确认 `datasets/<platform>/env_data.json` 中的 `reference_embedding_path` 使用**相对于项目根目录**的路径。

## 项目结构

```
MIDSim/
├── config/                          # 仿真与模型配置
│   ├── config_rednote.json
│   ├── config_twitter.json
│   ├── config_weibo.json
│   ├── params_rednote_qwen25_14B.json   # 实验参数（由主 config 引用）
│   ├── params_twitter_qwen25_14B.json
│   ├── params_weibo_qwen25_14B.json
│   └── model_config.json            # LLM / Embedding 端点
├── datasets/                        # 输入骨干数据（见「数据集」）
│   ├── rednote/
│   ├── twitter/
│   └── weibo/
├── scripts/
│   ├── model/                       # vLLM 启动 / 停止脚本
│   └── README.md                    # 脚本说明
├── src/
│   ├── onesim/                      # 仿真核心框架（YuLan-OneSim）
│   └── envs/
│       ├── midsim_rednote/
│       ├── midsim_twitter/
│       └── midsim_weibo/
│           └── code/                # SimEnv、UserAgent、Algorithm 等
├── hfd.sh                           # Hugging Face 下载辅助脚本
├── setup.py
├── README.md                        # 英文说明
└── README_zh.md                     # 中文说明（本文件）
```

## 配置说明

### 仿真配置（`config/config_*.json`）

常用字段：

- `simulator.environment.max_steps`：仿真轮数
- `simulator.environment.interval`：轮次间隔（模拟时间，秒）
- `agent.profile`：智能体数量与画像/ schema 路径
- `agent.memory`：短期 / 长期记忆设置
- `simulator.environment.additional_config.params_path`：实验参数文件路径（如 `config/params_rednote_qwen25_14B.json`）

### 实验参数（`config/params_*.json`）

各场景独立的实验参数，启动时加载到环境的 `midsim_params` 中。由主配置文件 `additional_config.params_path` 引用，**无需改 CLI 命令**，只改 params 文件即可调参。

示例（`config/params_rednote_qwen25_14B.json`）：

```json
{
  "exposure": {
    "social": { "probability": 1 },
    "recommendation": {
      "probability": 1,
      "types": ["Interest Recommendation"],
      "alpha": 0.2,
      "interest_recommendation": { "interest_k": 20, "target_k": 1 }
    },
    "search": {
      "types": ["Relevant Search"],
      "alpha": 0.5
    },
    "notification": { "attention_budget": 10 }
  },
  "user": {
    "own_note_cap_days": 7.0,
    "memory_similarity": {
      "policy": "memory_nonempty,keyword,embedding",
      "multi_combine": "or",
      "keyword_enabled": true,
      "embedding_enabled": true,
      "min_common_tokens": 2,
      "embed_threshold": 0.65,
      "include_historical_summary": true,
      "embedding_config_path": "",
      "embed_max_chars": 400,
      "embed_max_chunks": 12,
      "embed_chunk_agg": "mean"
    },
    "freshness": {
      "stale_days": 7.0,
      "low_activity_time_module_threshold": 0.75
    },
    "activity": {
      "remap": { "out_min": 0.4, "out_max": 0.8 },
      "low_activity_memory_gate_threshold": 0.65
    },
    "interaction_threshold": {
      "same_targets": { "support": [1, 2, 3, 4], "probs": [0.8, 0.1, 0.07, 0.03] },
      "diff_targets": { "support": [1, 2, 3, 4, 5], "probs": [0.7, 0.2, 0.08, 0.02, 0.01] },
      "keep_following": { "support": [0, 1], "probs": [0.9784, 0.0252] }
    }
  },
  "simulator": {
    "max_span_days": 24.0,
    "max_step": 8,
    "timestamp_schedule_type": "power",
    "timestamp_power_p": 1.6,
    "timestamp_sigmoid_scale": 1.2,
    "timestamp_sigmoid_center_ratio": 0.5
  }
}
```

| 字段 | 说明 |
|------|------|
| `exposure.social.probability` | 有关注流内容时发送社交推荐事件的概率 |
| `exposure.recommendation.probability` | 每种算法推荐请求独立触发的概率 |
| `exposure.recommendation.types` | 每轮用户请求的推荐算法类型列表 |
| `exposure.recommendation.alpha` | 推荐流深度的伯努利继续概率 |
| `exposure.recommendation.interest_recommendation.interest_k` | 兴趣推荐（LLM）从用户候选池抽取的上限 |
| `exposure.recommendation.interest_recommendation.target_k` | 兴趣推荐（LLM）从当前信息流抽取的上限 |
| `exposure.search.types` | 用户搜索时使用的搜索算法类型列表 |
| `exposure.search.alpha` | 搜索结果深度的伯努利继续概率 |
| `exposure.notification.attention_budget` | 每轮最多处理的 @ 提及数，超出则随机抽样 |
| `user.own_note_cap_days` | 自有笔记/关注流时间下界最多回溯天数；0 表示不截断 |
| `user.memory_similarity.policy` | Memory 相似度门控注入策略（`memory_nonempty` / `keyword` / `embedding` 等，逗号分隔） |
| `user.memory_similarity.multi_combine` | 多策略合成：`or`（任一命中）或 `and`（全部命中） |
| `user.memory_similarity.keyword_enabled` / `embedding_enabled` | 是否启用关键词重叠 / 向量相似度分支 |
| `user.memory_similarity.min_common_tokens` | 话题与 memory 共同词下限 |
| `user.memory_similarity.embed_threshold` | 向量相似度余弦阈值 |
| `user.memory_similarity.include_historical_summary` | memory 侧是否拼上 `historical_summary` |
| `user.memory_similarity.embedding_config_path` | Embedding 配置路径；空则用 `config/model_config.json` |
| `user.memory_similarity.embed_max_chars` / `embed_max_chunks` / `embed_chunk_agg` | 长文本 embedding 切分与聚合 |
| `user.freshness.stale_days` | 时效门控判定内容「时效已弱」的天数阈值 |
| `user.freshness.low_activity_time_module_threshold` | （可选）低活跃度下才注入时间 coaching 的阈值 |
| `user.activity.remap.out_min` / `out_max` | 将 profile `activity_level` 线性重映射到 `[out_min, out_max]` |
| `user.activity.low_activity_memory_gate_threshold` | 低活跃度下启用严格 memory 门控的 `activity_level` 上限 |
| `user.interaction_threshold.*` | 互动强度采样：`support` 与 `probs` 数组 |
| `simulator.max_step` | 仿真总轮数（写入 `StartEvent.max_step`） |
| `simulator.max_span_days` | 仿真总时间跨度（天），各轮时间窗长度之和的上界 |
| `simulator.timestamp_schedule_type` | 各轮时间窗长度分配方式：`power` 或 `sigmoid` |
| `simulator.timestamp_power_p` | 幂函数调度指数（`schedule_type=power` 时生效） |
| `simulator.timestamp_sigmoid_scale` | Sigmoid 调度平滑参数（`schedule_type=sigmoid` 时生效） |
| `simulator.timestamp_sigmoid_center_ratio` | Sigmoid 拐点位置比例（0–1） |

profile 中的 `limit` 以及代码内的上下界（推荐 ≤ 15、搜索 ≤ 50）不在 params 里配置。

### 模型配置（`config/model_config.json`）

- `chat`：用户与推荐智能体使用的 LLM 后端
- `embedding`：记忆检索与相似度计算用的 Embedding 模型

运行前请替换占位 API Key 与本地路径。

## 致谢

MIDSim 基于 [YuLan-OneSim](https://github.com/RUC-GSAI/YuLan-OneSim) 开发。若使用底层框架，请一并引用 YuLan-OneSim 论文：

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

## 许可证

本项目继承 YuLan-OneSim 的 [Apache-2.0 许可证](LICENSE)。
