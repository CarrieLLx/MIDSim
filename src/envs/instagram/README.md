# instagram

Instagram simulation environment. Use `instagram` as the environment name. Agent IDs, roles and counts are configured in `config/config_instagram.json` under `agent.profile`; behavior parameters are in `config/params_instagram.json`. All paths below are relative to the project root.

The source data directory is `datasets/ins/`. Keep that existing data path while using `instagram` for environment and configuration names.

## Layout

```text
instagram/
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
  doc/real_content_pool/<slug>/
    meta.json
    content_pool_snapshot.json
    user_comments.json
  code/
```

The environment-level recommendation file provides shared coarse-interest candidates. Each topic's file provides fine-interest candidates and user candidate IDs. The loader combines both layers; experiment seed posts are supplied separately by `env_data.json`.

## Run

Configure the chat model and embedding service first. Model `config_name` values must match `model_config_name` references in `config/config_instagram.json`.

```bash
python src/main.py --config config/config_instagram.json --model_config config/model_config.json --env instagram --mode single
# Cross-platform launcher; explicit options override defaults:
python src/envs/instagram/run.py --model_config path/to/local_model_config.json
```

Without overrides, the launcher uses `config/config_instagram.json`, `config/model_config.json`, `--env instagram` and `--mode single`.

Topic selection is configured under `simulator.environment.additional_config`:

- `batch_from_manifest: true` selects every topic in the index.
- Otherwise, a nonempty `batch_scenarios` selects those slugs. The supplied configuration selects all indexed topics through `batch_from_manifest`.
- To use the single `scenario` field, first disable `batch_from_manifest` and clear `batch_scenarios`.

Generated run output is excluded by Git ignore rules.

## Current topic inputs

The `topics` and `scenarios` entries in `topics/index.json` use the same seven slugs. Each slug matches its topic directory, reference-data directory and metadata.

| Topic slug | Seed posts | Observed comments | Users | Relationships |
| --- | ---: | ---: | ---: | ---: |
| `fmla_c334ab` | 46 | 653 | 475 | 137 |
| `food_stamps_ecd548` | 10 | 47 | 42 | 14 |
| `hidden_fees_21044d` | 3 | 6 | 4 | 2 |
| `living_wage_bcd48e` | 8 | 120 | 85 | 72 |
| `no_tax_on_tips_f18fbe` | 14 | 1694 | 1613 | 310 |
| `rent_too_high_aebcfb` | 4 | 20 | 11 | 16 |
| `universal_school_meals_6d389a` | 7 | 2 | 5 | 4 |
| Total | 92 | 2542 | 2235 | 555 |

These counts describe the current topic files. Users and relationships are summed across topics without cross-topic deduplication. The housing topic currently uses `zoning reform` as its keyword while retaining the canonical slug `rent_too_high_aebcfb`.

`comment_counts.json` and `doc/selection_24day_report.json` describe the original 24-day selection, where the housing topic had 10 posts and 63 comments. They are historical selection records; the current simulation uses the inputs in `topics/`.

## Data and profiles

- `env_data.json` contains seed posts, timestamps and reference-vector paths. Observed comments are not preloaded into the simulation comment pool.
- `UserAgent.json` uses account IDs in `id` and display names in `username`. It retains historical posts, summaries, interests, activity and author-response tiers, including missing-data states.
- `Relationship.csv` records within-topic follows: `source_agent` follows `target_agent`.
- `content_pool_snapshot.json` contains observed posts and comments; `user_comments.json` organizes the same comments and context by user.
- The housing topic additionally retains `observed_comments.json` and `observed_comment_threads.json` as comment and thread evidence.

Posts use `id/content/time/url/user_id/username`, with integer millisecond timestamps. Comments retain linkage fields such as `comment_id`, `post_id`, `user_id` and `parent_comment_id`. Missing parents remain missing; mentions are not used to invent reply relationships. Original social text and display names retain their source language. Project-generated labels and topic titles are English.

Source tables and vectors live in `datasets/ins/`, including posts, comments, profiles, history, relationships, historical author threads and embeddings. Filter historical evidence by the topic time scope when deriving priors.

## Editing and rebuilding

Keep `UserAgent.json`, `Relationship.csv`, `meta.json` and `topics/index.json` consistent when editing users. Check the corresponding files in `doc/real_content_pool/<slug>/` when editing evaluation references.

This release provides the prepared topic snapshots. Collection pipelines, preprocessing scripts and original selection manifests are excluded. Rebuilding from the original source selection would restore the earlier housing dataset, so it is not a startup step for the adjusted snapshot.


Backups, Chinese prompt reading copies, local tests, precomputed metrics, historical migration reports and Windows launchers are excluded from publication. Runtime English TXT prompts and TOML policies are included.
