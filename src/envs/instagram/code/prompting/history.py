"""Select historical evidence for the current observation, without editing profiles."""
import json
import re
from loguru import logger
from ..embedding import EmbeddingService, cosine_similarity

_STOP = set('the a an and or to of in on for with is are was were i you my your it this that as at be have has had from by not'.split())


def observation_contents(observation):
    decoder = json.JSONDecoder()
    found = []
    def visit(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in ('content', 'content_preview', 'mention_comment_content', 'comment_content') and isinstance(child, str):
                    found.append(child)
                elif key not in ('historical_posts', 'historical_summary', 'description'):
                    visit(child)
        elif isinstance(value, list):
            for child in value: visit(child)
    pos = 0
    while pos < len(observation):
        match = re.search(r'[\[{]', observation[pos:])
        if not match: break
        pos += match.start()
        try:
            value, size = decoder.raw_decode(observation[pos:])
            visit(value); pos += size
        except ValueError:
            pos += 1
    # Notification comment bodies are rendered outside their post JSON.
    for match in re.finditer(r'(?m)^\s*([^\n:]+):\s*\n(.*?)\n\s*\1 ID:', observation, re.S):
        found.append(match.group(2).strip())
    return list(dict.fromkeys(t.strip() for t in found if t.strip()))


def select_history(history, observation, *, limit=2, threshold=0.65, max_content_chars=500):
    queries = observation_contents(observation)
    rows = list(history.values()) if isinstance(history, dict) else history if isinstance(history, list) else []
    candidates = []; seen = set()
    for row in rows:
        if not isinstance(row, dict): continue
        text = str(row.get('content') or '').strip()
        key = re.sub(r'\W+', '', text.casefold())
        if len(key) < 20 or key in seen: continue
        # Current-post content is not historical evidence for this decision.
        if any(key in re.sub(r'\W+', '', q.casefold()) or re.sub(r'\W+', '', q.casefold()) in key
               for q in queries if len(re.sub(r'\W+', '', q)) >= 60): continue
        seen.add(key); candidates.append(row)
    if not queries or not candidates: return [], 'no_relevant_input'
    texts = [str(r['content'])[:2000] for r in candidates]
    try:
        vectors = EmbeddingService().embed([q[:2000] for q in queries] + texts)
        if len(vectors) != len(queries)+len(texts): raise ValueError('embedding count mismatch')
        scores = [max(cosine_similarity(v, q) for q in vectors[:len(queries)]) for v in vectors[len(queries):]]
        method = 'embedding'
    except Exception as exc:
        logger.warning('Historical evidence retrieval unavailable: {}; using lexical overlap', type(exc).__name__)
        tokens = lambda t: set(re.findall(r'[a-z]{3,}', t.lower())) - _STOP
        qs = [tokens(q) for q in queries]
        scores = [max((len(tokens(t)&q)/max(1,min(len(tokens(t)),len(q))) if len(tokens(t)&q)>=2 else 0) for q in qs) for t in texts]
        threshold = 0.3; method = 'lexical_fallback'
    selected = []
    for i in sorted(range(len(scores)), key=lambda i: (-scores[i], i)):
        if scores[i] < threshold: continue
        text = texts[i]
        words = set(re.findall(r'\w+', text.lower()))
        if any(len(words & prev)/max(1,len(words | prev)) > .8 for _,prev in selected): continue
        selected.append(({'content': str(candidates[i]['content']), 'time': candidates[i].get('time'),
                          'url': candidates[i].get('url'), 'relevance': round(scores[i],4)}, words))
        if len(selected) >= limit: break
    # Share a batch-wide character budget; the ellipsis counts toward it.
    result = [r for r,_ in selected]
    remaining = max(0, int(max_content_chars))
    bounded = []
    for index, row in enumerate(result):
        allowance = remaining // (len(result) - index)
        if allowance <= 0:
            break
        content = row['content']
        if len(content) > allowance:
            content = content[:allowance - 1] + '\u2026'
            row['content_truncated'] = True
        row['content'] = content
        remaining -= len(content)
        bounded.append(row)
    return bounded, method
