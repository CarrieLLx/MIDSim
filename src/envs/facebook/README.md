# facebook

Facebook simulation environment. Use `facebook` as the environment name. Agent IDs, roles and counts are configured in `config/config_facebook.json` under `agent.profile`; behavior and exposure parameters are in `config/params_facebook.json`. All paths below are relative to the project root.

## Layout

```text
facebook/
  actions.json / events.json / scene_info.json / system_data_model.json
  run.py
  schema/                         # UserAgent, Algorithm, Platform and Relationship
  recommendation.json             # Shared background candidates
  topics/
    index.json
    <slug>/
      meta.json
      env_data.json
      recommendation.json
      UserAgent.json
      Relationship.csv
  doc/real_content_pool/<slug>/    # Observed posts and comments for evaluation
  code/                           # Runtime code and English prompts
```

Recommendation candidates have two layers: `coarse_interest_recommendation_content_pool` in the environment-level file and `fine_interest_recommendation_content_pool` in each topic's file, with optional `candidate_post_ids_by_user`. They are merged into the Algorithm's `interest_recommendation_content_pool`; topic entries override shared entries with the same ID. Experiment seed posts remain in `env_data.json` and follow the `current_posts` path.

## Run

Configure the chat model and embedding service first. Model `config_name` values must match the `model_config_name` references in `config/config_facebook.json`. Keep API credentials in your local model configuration.

```bash
python src/main.py --config config/config_facebook.json --model_config config/model_config.json --env facebook --mode single
# Cross-platform launcher with the same defaults:
python src/envs/facebook/run.py
```

Explicit command-line options override launcher defaults. Topic selection is configured under `simulator.environment.additional_config`:

- `batch_from_manifest: true` selects every topic in `topics/index.json`.
- Otherwise, a nonempty `batch_scenarios` selects the listed slugs, for example `["living_wage_bcd48e"]`.
- To use the single `scenario` field, first disable `batch_from_manifest` and clear `batch_scenarios`.

## Input format

User `id` is the original account identifier; `username` is the display name. Redundant identity aliases such as `nickname` and `profile_name`, and the unused `statuses_count` field, are omitted from stored profiles. Historical and seed posts share `id`, `content`, `time` (integer milliseconds), `url`, `user_id` and `username`. Comments and relationships refer to account IDs. Interest tags, historical summaries and activity levels are preserved for each user.

When editing users, update `UserAgent.json`, `Relationship.csv`, the `portrait.UserAgent` count in `meta.json`, and `topics/index.json` together. Observed comments are stored in `doc/real_content_pool/<slug>/`; seed comments start empty.

## Source data and publication

`datasets/facebook/` contains source post, comment, profile, relationship and history tables plus the runtime reference vectors. The prepared topic bundles are authoritative for simulation input. Collection archives, preprocessing scripts and dataset-selection manifests are not part of this runtime release.

Local author-history analysis, screenshots, comparison snapshots, Chinese reading copies, tests, Windows launchers and precomputed text-diversity reports are excluded. Runtime English TXT prompts, TOML policies, topic inputs and evaluation references are included. Original social text and display names retain their source language.
