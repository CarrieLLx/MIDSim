# MIDSim

MIDSim is a multi-channel information diffusion simulator built on [YuLan-OneSim](https://github.com/RUC-GSAI/YuLan-OneSim). This release contains the **Facebook**, **Instagram** and **X** environments and the shared OneSim runtime.

User agents have profiles, memories and social relationships. They can encounter posts through recommendation, following, search, bookmarks and notifications. X also supports browsing historical replies on user profiles.

## Included files

```text
src/
  main.py                         # Main simulation entry point
  app.py                          # Compatibility entry point for the same CLI
  envs/facebook/
  envs/instagram/
  envs/x/
  onesim/
config/                           # Seven runtime JSON configurations
datasets/facebook/                # Source tables and reference vectors
datasets/ins/
datasets/x/
docs/                             # Current behavior documentation
scripts/check_release.py          # Offline configuration and input checks
setup.py
LICENSE
```

This is a command-line simulation release. It does not include a web application, Docker deployment, collection pipelines, preprocessing pipelines, experiment outputs, model weights or local transfer tools. Recommendation candidates are already included in the environment JSON files, so `datasets/additional` is not required.

## Installation

Python 3.11 is the tested local runtime; package metadata permits Python 3.10 or newer. Install Git LFS before cloning so the large relationship and historical-comment tables are downloaded as data rather than pointer files.

```bash
git lfs install
git clone https://github.com/CarrieLLx/MIDSim.git
cd MIDSim
git lfs pull
python -m venv .venv
# Activate the virtual environment for your operating system.
python -m pip install -e .
```

Run commands from the project root. The prepared environment bundles and datasets remain in the checkout; they are not embedded into an installed wheel.

## Model configuration

Edit `config/model_config.json` for your chat and embedding services. The published file is an example with local endpoints and placeholder credentials. Set model names to the exact names exposed by your services, and replace credentials locally when needed.

- `vllm-qwen-9888` is the chat configuration used by the example.
- `vllm-embedding-bert` is the embedding configuration referenced by memory and similarity features.
- Runtime reference vectors were generated with BGE base English embeddings of dimension 768. Use a compatible embedding model when comparing generated content with those references.

The seven published configuration files are `model_config.json`, `config_facebook.json`, `config_instagram.json`, `config_x.json`, `params_facebook.json`, `params_instagram.json` and `params_x.json`. Model configuration names referenced by the simulation files must exist in the model configuration.

Model serving is separate from the simulator. You may use existing compatible endpoints; weights, download scripts and machine-specific launch scripts are not part of this release.

## Run

Validate local inputs first; this does not call a model service:

```bash
python scripts/check_release.py
```

Choose an environment:

```bash
python src/main.py --config config/config_facebook.json --model_config config/model_config.json --env facebook --mode single
python src/main.py --config config/config_instagram.json --model_config config/model_config.json --env instagram --mode single
python src/main.py --config config/config_x.json --model_config config/model_config.json --env x --mode single
```

`python src/app.py` accepts the same arguments. It starts the simulator, not a web server. The installed `midsim` command and the legacy `yulan-onesim-cli` alias also call the same entry point.

The shorter launchers `python src/envs/facebook/run.py`, `python src/envs/instagram/run.py` and `python src/envs/x/run.py` use the corresponding default configuration. Explicit command-line options override those defaults.

Topic selection is under `simulator.environment.additional_config`: `batch_from_manifest: true` selects every topic in the environment index. Otherwise a nonempty `batch_scenarios` list takes precedence over the single `scenario` field. Use existing slugs from `topics/index.json`.

## Inputs and outputs

The authoritative simulation inputs are `src/envs/<platform>/topics/<slug>/`: user profiles, relationships, seed posts, recommendations and metadata. Observed evaluation posts and comments are in `doc/real_content_pool/<slug>/`. Initial simulation comments are empty.

The root `datasets/` directories contain source records and reference vectors. These source tables can describe an earlier selection than an independently adjusted topic bundle, particularly the Instagram housing topic; use the topic bundle for the actual run population and counts. Collection and rebuilding procedures are not included in this runtime release.

See the [dataset README](datasets/README.md) for file formats, record counts, key fields, reference vectors and Git LFS download instructions.

The four reference-vector files required by the current topics are:

- `datasets/facebook/embeddings/bge-base-en-v1.5_embeddings.json`
- `datasets/ins/embeddings/bge-base-en-v1.5_embeddings.json`
- `datasets/ins/embeddings/rent_bge-base-en-v1.5_embeddings.json`
- `datasets/x/embeddings/bge-base-en-v1.5_embeddings.json`

Generated outputs are written under `src/envs/<platform>/runs/<timestamp>/`, including runtime snapshots, metrics and logs. They are excluded from version control.

Project documentation, comments, topic titles and generated labels are English. Original social text, biographies and display names retain their source language; account IDs and relationships are unchanged.

## Environment and mechanism guides

- [Facebook](src/envs/facebook/README.md)
- [Instagram](src/envs/instagram/README.md)
- [X](src/envs/x/README.md)
- [BM25 search](docs/bm25_search.md)
- [Comment views](docs/comment_views.md)
- [Model request retries](docs/model_request_retries.md)

## License and attribution

See [LICENSE](LICENSE). MIDSim builds on YuLan-OneSim; the shared runtime retains the upstream attribution and license.
