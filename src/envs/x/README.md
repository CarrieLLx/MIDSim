# x

X simulation environment. Use `x` as the environment name, with `config/config_x.json` and `config/params_x.json`. Topic inputs follow the shared Facebook/Instagram layout. X additionally supports browsing historical replies on user profiles.

## Layout and input format

```text
x/
  code/                         # Runtime code and English prompts
  schema/                       # Agent and relationship fields
  topics/
    index.json                  # Topic list and counts
    <slug>/
      meta.json
      env_data.json
      UserAgent.json
      Relationship.csv
      recommendation.json
  doc/real_content_pool/<slug>/
    meta.json
    content_pool_snapshot.json
    user_comments.json
  recommendation.json           # Shared background candidates
  run.py                        # Cross-platform launcher
```

Each topic directory contains five input files. Observed posts and comments are in `doc/real_content_pool/<slug>/`. Individual collection records remain in `datasets/x/comments.jsonl`. Selection windows, rules and per-post counts are recorded in the bundled `comment_counts.json` report. Its provenance references may name collection files outside this runtime release.

User `id` is the stable account identifier and `username` is the display name. Historical posts use `id`, `content`, `time`, `url`, `user_id` and `username`, without a duplicate `post_id`. Collection evidence, numeric account mappings and collection cutoffs remain in the source data. Empty audit, bookmark and supplementary-comment review state is initialized by the schema at runtime. Historical reply roots and ancestor chains, and the `protected` flag, are retained for profile browsing and visibility checks.

Project labels and topic titles are English. Original social text and display names retain their source language.

## Run and selected data

Run from the project root:

```bash
python src/envs/x/run.py
# Equivalent full command:
python src/main.py --config config/config_x.json --model_config config/model_config.json --env x --mode single
```

Configure the chat model and embedding service first. Model names must match the `model_config_name` references in `config/config_x.json`. Platform routes recommendation and search requests to two Algorithm agents. Explicit command-line options override launcher defaults.

The supplied configuration sets `batch_from_manifest=true` to run all seven topics. Topic selection precedence is: all topics when `batch_from_manifest=true`, then a nonempty `batch_scenarios`, then the single `scenario` field. Dataset rebuilding does not require selecting all topics in the run configuration.

The seven fixed 24-day post windows contain 98 seed posts, 359 filtered comments and 425 user instances across topics (421 distinct users).

| Topic slug | Seed posts | Observed comments | Users |
| --- | ---: | ---: | ---: |
| `fmla_c334ab` | 11 | 29 | 38 |
| `food_stamps_ecd548` | 20 | 84 | 101 |
| `hidden_fees_21044d` | 4 | 1 | 5 |
| `living_wage_bcd48e` | 11 | 105 | 112 |
| `no_tax_on_tips_f18fbe` | 25 | 67 | 73 |
| `rent_too_high_aebcfb` | 18 | 35 | 51 |
| `universal_school_meals_6d389a` | 9 | 38 | 45 |

Only post publication times are filtered by the UTC `[start, end)` window, within the original filtered account sample. Comments cascade from selected posts and are deduplicated by `sample_id`; comment timestamps do not affect selection. Users are the union of selected post and filtered comment authors. Relationships retain edges whose endpoints are in the same topic's user set. User history is background context and is not restricted to the 24-day window.

`env_data.json` contains only seed posts, starts at the window's beginning and has empty initial comments. Observed comments are evaluation references; simulation comments are created at runtime.

## Reference embeddings

Reference vectors are stored in `datasets/x/embeddings/bge-base-en-v1.5_embeddings.json`: 87 vectors of 768 dimensions, built by grouping all 359 observed comments by post. The other 11 posts have no observed comments and therefore no reference vectors. Long texts were token-chunked, averaged and normalized.

The release includes prepared topic bundles and reference vectors. Collection archives, account-cohort manifests and rebuilding scripts are not needed to run them and are not included.

## Recommendation background

The environment-level `recommendation.json` contains 374 shared interest posts. Each topic's file contains 20 fine-interest posts. The loader combines 394 synthetic background candidates per topic; these do not enter the experiment seed pool. The original generation spreadsheets and `datasets/additional` are not runtime dependencies and are excluded.

## Profile browsing

Interaction decisions may include `has_profile_visit` and `profile_visits`. An interaction-triggered visit requires very high interest in a specific visible statement, an explicit reason, and valid author/post/comment evidence. Relevance, agreement, disagreement or merely wanting to reply is insufficient. A separate follow decision requires very high sustained interest in the account as a whole. Protected accounts are not automatically approved for new follows.

Each active round also reviews followed accounts. Up to five accounts are presented directly. For more than five, profile embeddings rank all followed accounts and select the five most similar. Embedding failure is recorded as `embedding_unavailable` and skips that channel. Candidate cards do not reveal historical replies in advance.

The following-review prompt requires a decision for every candidate. Its strict output validation checks fields, types, order and consistency; one retry uses the same prompt and input. Each user may visit a given target at most once per round, with eligibility reevaluated in the next round.

Profile visits use a separate `ProfileVisitEvent`, the `profile_history_context.txt` prompt and a `ProfileVisitCompletedEvent`. Visitors may comment on the root post, reply to a displayed comment or decline to interact. The workflow shares memory, interaction quotas, bookmarks, search and notification handling with other channels.

The default visit displays the target's two newest visible historical replies, including each reply's full root-to-parent path. Configure the nonnegative integer `exposure.profile_visit.history_comment_limit` in `config/params_x.json`; `0` displays none. Parent context does not count against this limit. Future records, unavailable timestamps and inaccessible content are filtered; missing ancestors and cycles are marked without inventing text.

History may come from live comments, `historical_posts[*].comments` or `historical_comments`. A historical reply retains `comment_id`, `post_id`, `user_id`, `content`, `time`, `parent_comment_id`, an optional root `post`, and `ancestor_comments`. Roots can also be resolved from live or historical posts. Missing roots cannot create replyable conversations. Historical context enters the live pool only after an explicit bookmark or submitted comment. Topic input files are not changed by browsing.

`profile_visit_audit` records candidates, accepted visits, times and paths. Normal LLM logs retain prompts and outputs. The English templates are `following_profile_review.txt`, `profile_discovery.txt` and `profile_history_context.txt` under `code/prompts/tasks/`.

Per-round review, interest thresholds, embedding top-five selection and visit deduplication are experimental assumptions. The simulator uses root posts with nested comments and does not implement full blocking, follow approval or per-post reply permissions.

## Offline checks and publication

Run `python scripts/check_release.py` from the project root to validate the published configurations and input paths without calling a model service. Development regression tests remain local.

Publication includes runtime code, English TXT/TOML prompts, schemas, topic inputs and evaluation references. Ignore rules exclude backups, Chinese reading copies, local tests, Windows launchers, identity-review HTML, precomputed text-diversity analysis, logs and caches.
