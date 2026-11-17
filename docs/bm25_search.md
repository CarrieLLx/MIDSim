# BM25 search

The published `facebook`, `instagram` and `x` environments support BM25 through `src/onesim/utils/bm25.py`. Their current parameter files select `BM25 Search`.

Configure search under `exposure.search` in the corresponding `config/params_<platform>.json`:

```json
{
  "types": ["BM25 Search"],
  "alpha": 0.5,
  "limit": 10,
  "bm25": {"k1": 1.2, "b": 0.75}
}
```

The optional `bm25` block defaults to `k1=1.2` and `b=0.75`. `k1` must be nonnegative and `b` must be between zero and one. Existing search-count sampling and result events still apply.

The index uses the current post pool supplied to the Algorithm, the user's query and the requested result count. Searchable text includes titles, body text (`content`, with `desc` as a fallback) and `tags_list`. Comments and historical background are not indexed. The upstream pool determines time and visibility scope.

Results retain the original post dictionaries and are sorted by BM25 score. Empty queries, stop-word-only queries and unmatched queries return no results; unrelated posts are not added as a fallback.

BM25 requires no training, embedding request or LLM request. English is tokenized by word. Chinese matching uses an installed jieba tokenizer when available and adjacent character pairs otherwise. This multilingual matching support remains available even though project documentation is English.

Each search engine builds its index on first use and reuses it until posts or indexed text change. Comment and counter changes do not trigger rebuilding. Indexing and scoring run in a worker thread. Queries score matching postings and retain the top results, while cache validation still scans the current indexed text.
