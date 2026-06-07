# MIDSim

**MIDSim** (Multi-channel **I**nformation **D**iffusion **Sim**ulator) is an LLM-driven simulator for studying information diffusion across **recommendation** and **social** channels.

User agents interact via feeds, comments, replies, and @ notifications; recommender agents serve multiple algorithms. Metrics are logged per simulation round.

Built on [YuLan-OneSim](https://github.com/RUC-GSAI/YuLan-OneSim). Supports scenarios on ACL, ICLR, Weibo, and Twitter backbones.

## Repository Status

Anonymous submission — full code and run instructions will be released here. See the paper for experimental details.

## Structure (Preview)

```
MIDSim/
├── config/
├── scripts/
├── src/
│   ├── midsim/
│   └── envs/
└── README.md
```

## Requirements

- Python 3.10+
- CUDA GPU(s) recommended (local vLLM)

## License

Apache-2.0
