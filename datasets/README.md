# MIDSim datasets

This directory contains the published source records and reference embeddings for **Facebook**, **Instagram** and **X**. The Instagram source directory is named `ins`; its simulation environment is named `instagram`.

The source tables support inspection and reuse of the collected data. The simulator uses the prepared topic bundles under [`src/envs`](../src/envs), together with the reference embeddings described below.

## Source tables and simulation inputs

These two layers have different scopes:

- **Source records** in this directory include posts, comments, profiles, relationship observations and historical activity. They may contain accounts or records outside the final simulation population.
- **Prepared simulation inputs** are in `src/envs/<platform>/topics/<slug>/`: `env_data.json`, `UserAgent.json`, `Relationship.csv`, `recommendation.json` and `meta.json`. These files define the population and seed posts used by a run.
- **Observed evaluation data** are in `src/envs/<platform>/doc/real_content_pool/<slug>/`. Observed comments are kept here; initial comments in the simulation's `env_data.json` are empty.
- **Reference embeddings** remain in this directory and are loaded using each topic's `reference_embedding_path`.

The complete source tables are not required by the simulation entry point after the topic bundles have been prepared. They are included in this release as source data. The release checker also checks that these published tables are present.

Source-table counts must not be used as simulation counts. In particular, the Instagram housing source selection contains 10 posts and 63 comments, while the current housing topic bundle contains 4 posts and 20 comments. The topic bundle is authoritative for a run.

## File inventory

Each platform contains six source tables:

| Data | Facebook and Instagram | X |
| --- | --- | --- |
| Selected posts | `posts.csv` | `posts.jsonl` |
| Observed comments | `comments.csv` | `comments.jsonl` |
| Account profiles | `profiles.csv` | `user_profiles.jsonl` |
| Relationship observations | `relationships.csv` | `relationships.csv` |
| Historical posts | `historical_posts.csv` | `historical_posts.jsonl` |
| Comments on historical posts | `historical_post_comments.csv` | `historical_comments.jsonl` |

The following counts were measured from the published files. CSV headers and blank JSONL lines are excluded. Counts refer to rows or JSON objects, not necessarily distinct accounts or unique graph edges.

| Source records | Facebook | Instagram | X |
| --- | ---: | ---: | ---: |
| Selected posts | 23 | 98 | 98 |
| Observed comments | 1,121 | 2,585 | 359 |
| Account profiles | 898 | 2,287 | 421 |
| Relationship observations | 179,526 | 2,683,755 | 2,522 |
| Historical posts | 11,923 | 31,792 | 25,394 |
| Comments on historical posts | 15,125 | 19,099 | 134,977 |

The prepared bundles contain seven topics per platform:

| Prepared simulation inputs | Facebook | Instagram | X |
| --- | ---: | ---: | ---: |
| Seed posts | 23 | 92 | 98 |
| Observed evaluation comments | 1,121 | 2,542 | 359 |
| User instances across topics | 882 | 2,235 | 425 |
| Relationship rows across topics | 101 | 555 | 1,263 |

A user appearing in multiple topics contributes multiple user instances in the second table. For example, X has 421 source profile records and 425 user instances across its prepared topic bundles.

## Formats and key fields

All text files use UTF-8. CSV files have a header row and may contain quoted multiline text or JSON within individual cells. JSONL files contain one JSON object per nonempty line. Preserve account and post identifiers as strings when loading data.

### Facebook and Instagram

- `posts.csv`: `id`, `content`, `time`, `url`, `user_id`, `username`, `topic_slug`, `topic`, `keyword` and source identifiers describe each selected post.
- `comments.csv`: `post_id` links to a selected post; `comment_id`, `thread_id`, `parent_comment_id` and `message_role` describe comment structure. `content`, `user_id`, `username` and time fields contain the observed message and author information. Fields prefixed with `source_` preserve source linkage.
- `profiles.csv`: account identifiers, display names, biographies, profile attributes and collection metadata. Facebook includes structured profile information in JSON-valued cells. Instagram includes collection status and cutoff fields.
- `relationships.csv`: `source_username`, `target_username` and `relation_type`, with account identifiers, display names and collection metadata. Interpret `relation_type` when constructing graph edges; the prepared topic `Relationship.csv` files already contain the runtime graph.
- `historical_posts.csv`: Facebook stores text in `post_text`; Instagram uses `caption`. Both include source account, URL and time information.
- `historical_post_comments.csv`: Facebook stores comment text in `comment_text`; Instagram uses `text`. Parent identifiers and reply fields describe the observed comment structure.

### X

- `posts.jsonl`: `post_id`, `username`, `text`, `url`, `created_time`, `topics` and `topic_slugs`; `source_records` retains selection and source annotations.
- `comments.jsonl`: `post_id`, `comment_id`, `thread_id`, `user_id`, `content`, `author_display` and `message_role`; time-status and source-link fields describe provenance.
- `user_profiles.jsonl`: the account key is `username`; profile details are nested under `profile`, alongside collection status, topic membership and earlier profile observations.
- `relationships.csv`: source and target identifiers, `relation_type`, shared topics and evidence/time-status fields. This is a source observation table, not the final topic-specific runtime graph.
- `historical_posts.jsonl`: `post_id`, `username`, `text`, `created_time`, `role` and `source`; the original record is retained in `source_record`.
- `historical_comments.jsonl`: each object identifies `root_post_id`, `root_username`, `root_url` and `fetched_at`, with the collected comment object nested under `comment`.

Timestamps retain the representation of their source fields. Do not assume that every numeric timestamp across these tables uses the same unit. Display names are descriptive text, not join keys.

See the [X data notes](x/README.md) for additional details about the X snapshot.

## Topics

All three platforms use the same seven topic slugs:

| Slug | Topic |
| --- | --- |
| `fmla_c334ab` | Paid family and parental leave |
| `food_stamps_ecd548` | College students' access to SNAP food assistance |
| `hidden_fees_21044d` | Full-price disclosure for concert and sports tickets |
| `living_wage_bcd48e` | Raising the federal minimum wage |
| `no_tax_on_tips_f18fbe` | Tax exemption for tip income |
| `rent_too_high_aebcfb` | Expanding affordable housing supply |
| `universal_school_meals_6d389a` | Expanding free school meals |

The complete English topic titles and per-topic counts are stored in each environment's `topics/index.json` and topic `meta.json` files.

## Reference embeddings

These JSON arrays contain post-level reference vectors. Entries identify a post with `post_id` and/or `note_id`, and store a numeric `embedding` array. Some entries also include `embedding_meta` describing the observed comments and encoding process.

| File | Entries | Dimensions |
| --- | ---: | ---: |
| `facebook/embeddings/bge-base-en-v1.5_embeddings.json` | 15 | 768 |
| `ins/embeddings/bge-base-en-v1.5_embeddings.json` | 73 | 768 |
| `ins/embeddings/rent_bge-base-en-v1.5_embeddings.json` | 3 | 768 |
| `x/embeddings/bge-base-en-v1.5_embeddings.json` | 87 | 768 |

Use the vector file specified by the topic's `env_data.json`; the Instagram housing topic uses its dedicated file. A seed post with no observed comments need not have a reference vector. Vector counts therefore differ from seed-post counts. Use compatible BGE base English embeddings when comparing generated text with these reference vectors.

## Downloading the large files

Two source tables use Git LFS:

| File | Approximate size | Contents |
| --- | ---: | --- |
| `ins/relationships.csv` | 459 MB | 2,683,755 relationship observations |
| `x/historical_comments.jsonl` | 377 MB | 134,977 nested historical-comment records |

Sizes here use decimal megabytes. These files account for most of the source-data download size.

After cloning the repository, run:

```bash
git lfs install
git lfs pull
python scripts/check_release.py
```

The checker runs offline. It checks topic counts, profile and relationship IDs, required source files, reference-vector coverage and dimensions, and whether a downloaded file is still a Git LFS pointer.

## Loading examples

Run these examples from the project root. Iterate through large files instead of loading every row into memory.

```python
import csv
import json

with open("datasets/ins/posts.csv", encoding="utf-8-sig", newline="") as stream:
    for row in csv.DictReader(stream):
        post_id = row["id"]
        content = row["content"]

with open("datasets/x/comments.jsonl", encoding="utf-8") as stream:
    for line in stream:
        if line.strip():
            comment = json.loads(line)
            post_id = comment["post_id"]
            content = comment["content"]
```

## Language and publication scope

Project topic labels, role annotations and documentation are English. Original social text, biographies and display names retain their source language. Account IDs, post IDs and source relationships are preserved.

The release includes prepared data and reference vectors. Collection/preprocessing pipelines, source spreadsheets under `datasets/additional`, experimental outputs, backups and model weights are not included. Recommendation candidates are already embedded in the environment bundles.

For execution instructions, see the [project README](../README.md) and the [Facebook](../src/envs/facebook/README.md), [Instagram](../src/envs/instagram/README.md) and [X](../src/envs/x/README.md) environment guides.
