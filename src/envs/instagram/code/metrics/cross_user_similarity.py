"""Same-post, cross-user cosine similarity; missing comparisons remain None."""
from collections import defaultdict
from itertools import combinations
import math

try:
    from .embedding_metrics import embed_content_pool_comments
except ImportError:
    from embedding_metrics import embed_content_pool_comments


def summarize(rows):
    users = defaultdict(list)
    for uid, vector in rows:
        if uid and vector and all(math.isfinite(x) for x in vector):
            norm = math.sqrt(sum(x*x for x in vector))
            if norm: users[uid].append([x/norm for x in vector])
    means = []; maxima = []; pairs = 0
    for left, right in combinations(users.values(), 2):
        values = [max(-1.0, min(1.0, sum(x*y for x,y in zip(a,b))))
                  for a in left for b in right if len(a)==len(b)]
        if values:
            means.append(sum(values)/len(values)); maxima.append(max(values)); pairs += len(values)
    return {'mean': sum(means)/len(means) if means else None,
            'max': max(maxima) if maxima else None,
            'n_users': len(users), 'n_comments': sum(map(len,users.values())),
            'n_user_pairs': len(means), 'n_comment_pairs': pairs}


def compute_cross_user_similarity(pool, records):
    vectors = {(r.post_id,r.comment_id):r.embedding for r in records}
    posts = {}
    for pid, post in pool.items():
        pid = str(post.get('post_id') or pid)
        groups = {'all': [], 'roots': [], 'replies': []}
        missing_user = 0
        for cid, comment in (post.get('comments') or {}).items():
            uid = comment.get('user_id')
            if not uid: missing_user += 1; continue
            vector = vectors.get((pid,str(comment.get('comment_id') or cid)))
            row = (str(uid),vector)
            groups['all'].append(row)
            groups['replies' if comment.get('parent_comment_id') else 'roots'].append(row)
        posts[pid] = {k:summarize(v) for k,v in groups.items()}
        posts[pid]['missing_user_comments'] = missing_user
    result = {'per_post': posts}
    for scope in ('all','roots','replies'):
        eligible = [p[scope] for p in posts.values() if p[scope]['mean'] is not None]
        result[scope+'_mean'] = sum(p['mean'] for p in eligible)/len(eligible) if eligible else None
        result[scope+'_max'] = max((p['max'] for p in eligible), default=None)
        result[scope+'_eligible_posts'] = len(eligible)
    return result


def calculate_cross_user_similarity(data):
    from loguru import logger
    try:
        pool = data.get('content_pool') or {}
        records, _ = embed_content_pool_comments(pool, data)
        return compute_cross_user_similarity(pool, records)
    except Exception as exc:
        logger.exception('Cross-user similarity metric failed')
        result = compute_cross_user_similarity({}, [])
        result['error'] = str(exc)
        return result
