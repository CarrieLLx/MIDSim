# Comment and reply views

The published Facebook, Instagram and X environments use `src/onesim/utils/comment_view.py` to prepare bounded comment views.

Root comments are ranked by reply count, then likes. Ties use embedding relevance to the post. Each selected root comment includes two direct replies by default; replies use the same popularity ordering, with embedding relevance to their parent as a tie-breaker. Equal similarity is resolved by stable IDs rather than random selection.

Ordinary views retain five root comments and their selected replies. Notification views additionally retain the target comment, its available ancestor chain and its replies. Author-comment previews preserve `parent_comment_id`. These are input views: they do not add or modify environment comments, change interaction budgets or rewrite prompt policies.

An optional parameter controls replies per comment:

```json
{
  "exposure": {
    "comment_view": {
      "replies_per_comment": 2
    }
  }
}
```

Embedding uses the environment's configured similarity service, falling back to the embedding section of the model configuration when no override is set. If embedding fails, the code logs a warning and falls back to popularity and stable-ID ordering while preserving required parent context.

Recommendation retains the full comment candidates until the view is prepared at the browsing timestamp. Future comments and replies that depend on hidden parents are excluded. Embedding runs in a worker thread without holding the environment state lock.

X profile-history browsing retains its own target-selection rules and historical-reply limit; it is separate from the ordinary feed comment limit.
