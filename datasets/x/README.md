# X source data

The source tables describe 98 selected posts, 359 filtered observed comments and 421 distinct users (425 user instances summed across seven topics).

Only seed-post publication times were filtered by the fixed UTC 24-day windows. Comments cascade from selected posts, users from their authors, and relationships from users in the same topic. Comment and relationship timestamps do not introduce additional selection filters.

| File | Contents |
| --- | --- |
| `posts.jsonl` | Selected source posts with topic membership and collection provenance. |
| `comments.jsonl` | Individual observed comments, deduplicated by `sample_id` and linked by `post_id`. Missing reply timestamps and native IDs remain missing. |
| `user_profiles.jsonl` | Source profiles of selected post and comment authors. |
| `relationships.csv` | Collected following/follower records with both endpoints in a topic's user cohort. Reverse `followers` records to interpret follower-to-followee direction; runtime topic tables deduplicate directed edges. |
| `historical_posts.jsonl` | Historical background for selected users. |
| `historical_comments.jsonl` | Collected conversations under historical posts, including replies by other accounts needed for context. |
| `embeddings/bge-base-en-v1.5_embeddings.json` | 87 reference vectors of dimension 768, from all 359 observed comments grouped by post; 11 posts have no observed comments and no reference vector. |

The source tables retain original social text, names, IDs and timestamps. Publication normalizes project-owned topic and role labels to English. Historical background is not restricted to the experiment window; profile browsing applies visibility at the visit time.

The actual simulation inputs are the prepared bundles in `src/envs/x/topics/`. Their seed comments are empty. Observed evaluation data is in `src/envs/x/doc/real_content_pool/`.

This runtime release does not include raw collection archives, preprocessing scripts, account-review reports or local transfer files. No reconstruction from those excluded files is required to run the prepared topic bundles.
