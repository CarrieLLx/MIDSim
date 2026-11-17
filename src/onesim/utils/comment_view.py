"""Read-only comment views: popular roots with replies and semantic tie breaks."""
from collections import Counter, defaultdict
from copy import deepcopy
import asyncio
import math
import threading
from loguru import logger

_cache_setup_lock = threading.Lock()


def cached_legacy_embeddings(env, texts, path, load_config, get_embeddings):
    with _cache_setup_lock:
        if not hasattr(env, '_comment_vector_cache'):
            env._comment_vector_cache = {}
            env._comment_vector_lock = threading.Lock()
    base, model = load_config(path)
    with env._comment_vector_lock:
        cache = env._comment_vector_cache
        keys = [(base, model, text) for text in texts]
        missing = list(dict.fromkeys(key for key in keys if key not in cache))
        if missing:
            vectors = get_embeddings(base, model, [key[2] for key in missing])
            if len(vectors) != len(missing):
                raise ValueError('Incomplete embedding response')
            if len(cache) + len(missing) > 20000:
                cache.clear()
                missing = list(dict.fromkeys(keys))
                vectors = get_embeddings(base, model, [key[2] for key in missing])
                if len(vectors) != len(missing):
                    raise ValueError('Incomplete embedding response')
            cache.update(zip(missing, vectors))
        return [cache[key] for key in keys]


def text_of(row):
    return str(row.get('content') or row.get('comment_content') or row.get('desc') or row.get('title') or '')


def parent_of(row):
    return str(row.get('parent_comment_id') or row.get('replied_tweet_id') or row.get('replyedTweet_id') or row.get('reposted_blog_id') or '')


def heat(row, child_count=0):
    def number(key):
        try:
            value = float(row.get(key) or 0)
            return value if math.isfinite(value) and value > 0 else 0
        except (TypeError, ValueError):
            return 0
    return (max(child_count, number('sub_comment_count'), number('reply_count')),
            max(number('like_count'), number('likes_count'), number('digg_count'), number('likes')))


def cosine(a, b):
    if len(a) != len(b) or not a:
        raise ValueError('Invalid embedding dimensions')
    return sum(x*y for x,y in zip(a,b)) / max(1e-12, math.sqrt(sum(x*x for x in a)*sum(y*y for y in b)))


def rank_comments(ids, comments, children, query, embed=None, scores=None):
    """Popularity is primary. Resolve heat ties by embedding, then stable ID."""
    groups = defaultdict(list)
    for cid in ids:
        groups[heat(comments[cid], len(children.get(cid, [])))].append(cid)
    ranked = []
    for popularity in sorted(groups, reverse=True):
        tied = sorted(groups[popularity])
        relevance = {cid: (scores or {}).get(cid, 0.0) for cid in tied}
        if len(tied) > 1 and embed is not None and str(query).strip():
            try:
                texts = [str(query)] + [text_of(comments[cid]) for cid in tied]
                vectors = []
                for start in range(0, len(texts), 32):
                    vectors.extend(embed(texts[start:start+32]))
                if len(vectors) != len(texts):
                    raise ValueError('Incomplete embedding response')
                relevance = {cid: cosine(vectors[0], v) for cid,v in zip(tied, vectors[1:])}
                if not all(math.isfinite(v) for v in relevance.values()):
                    raise ValueError('Non-finite embedding similarity')
            except Exception as exc:
                # Do not replace unavailable semantic evidence with random order.
                logger.warning('Comment view embedding unavailable; using stable heat/ID order: {}', exc)
        ranked.extend(sorted(tied, key=lambda cid: (-relevance[cid], cid)))
    return ranked


def select_comment_view(post, *, root_limit=5, reply_limit=2, embed=None, field='comments', anchors=()):
    result = deepcopy(post)
    comments = {str(k): deepcopy(v) for k,v in (post.get(field) or {}).items() if isinstance(v,dict)}
    children = defaultdict(list)
    for cid,c in comments.items():
        parent = parent_of(c)
        children[parent if parent in comments else ''].append(cid)
    roots = rank_comments(children[''], comments, children, text_of(post), embed)[:max(0,root_limit)]
    visible = {}
    post_ids = {str(post[k]) for k in ('id','post_id','note_id','tweet_id','blog_id') if post.get(k)}
    def include(cid):
        chain=[];cursor=cid
        while cursor in comments and cursor not in chain and cursor not in visible:
            chain.append(cursor);cursor=parent_of(comments[cursor])
        if cursor in chain:
            logger.warning('Comment view excludes cyclic parent chain at {}', cid)
            return False
        for key in reversed(chain):
            visible[key]=comments[key]
            if parent_of(comments[key]) and parent_of(comments[key]) not in comments and parent_of(comments[key]) not in post_ids:
                visible[key]['parent_context_missing']=True
        return cid in visible
    for cid in dict.fromkeys(roots + [str(c) for c in anchors]):
        if not include(cid):continue
        for child in rank_comments(children.get(cid, []), comments, children, text_of(comments[cid]), embed)[:max(0,reply_limit)]:
            include(child)
    result[field]=visible
    return result


def settings(env):
    raw=(getattr(env,'data',{}) or {}).get('midsim_params') or {}
    cfg=(raw.get('exposure') or {}).get('comment_view') or {}
    reply_limit=cfg.get('replies_per_comment',2)
    if type(reply_limit) is not int or reply_limit < 0:
        raise ValueError('exposure.comment_view.replies_per_comment must be a non-negative integer')
    return reply_limit


def select_for_env(env, post, root_limit=5, *, field='comments', anchors=()):
    return select_comment_view(post, root_limit=root_limit, reply_limit=settings(env),
                               embed=env._comment_view_embed, field=field, anchors=anchors)


async def select_posts_for_agent(agent, posts, *, as_of=None):
    """Used by legacy feed handlers and author views, outside environment locks."""
    env=getattr(agent,'env',None)
    if env is None or not hasattr(env,'_comment_view_embed'):
        return posts
    if as_of is not None:
        return await env.snapshot_visible_comments(posts, as_of)
    def build():
        return {pid: select_for_env(env,p,field=getattr(env,'comment_view_field','comments')) for pid,p in posts.items()}
    return await asyncio.to_thread(build)


async def refresh_mention_views(agent, entries, field, id_field, timestamp):
    env = getattr(agent, 'env', None)
    if env is None or not hasattr(env, '_comment_view_embed'):
        return
    for entry in entries:
        pid = str(entry[id_field])
        anchors = [str(entry['mention_comment_id'])] if entry.get('mention_comment_id') else []
        view = await env.snapshot_visible_comments({pid:entry[field]}, timestamp, anchors={pid:anchors})
        entry[field] = view[pid]


async def snapshot_threads(env, posts, as_of, *, limit_comments=True, anchors=None):
    """Copy committed state under lock, but run embedding after releasing it."""
    field=getattr(env,'comment_view_field','comments')
    def stamp(row):
        raw=row.get('timestamp',row.get('time'))
        if raw is None:return None
        try:
            value=float(raw)
            # Preserve small synthetic timestamps; real Unix seconds -> ms.
            return value*1000 if 1e8 <= abs(value) < 1e11 else value
        except (TypeError,ValueError):return None
    cutoff=float(as_of)
    if 1e8 <= abs(cutoff) < 1e11:cutoff*=1000
    async with env._lock:
        pool=env.data.get('content_pool',{})
        result={}
        for pid,post in posts.items():
            row=deepcopy(post);source=pool.get(pid,post)
            comments=source.get(field) or {}
            if getattr(env,'comment_view_linked',False):
                # Legacy Twitter replies / Weibo discussion posts live in the pool.
                comments=dict(comments)
                descendants={str(pid)}
                while True:
                    found={str(k):v for k,v in pool.items() if str(k) not in descendants and parent_of(v) in descendants}
                    if not found:break
                    comments.update(found);descendants.update(found)
            visible={str(k):deepcopy(v) for k,v in comments.items() if isinstance(v,dict) and stamp(v) is not None and stamp(v)<=cutoff}
            # A future/hidden parent must not leak a child into the visible tree.
            excluded=set(map(str,comments))-set(visible)
            while True:
                remove={k for k,v in visible.items() if parent_of(v) in excluded}
                if not remove:break
                excluded.update(remove)
                for k in remove:visible.pop(k)
            counts=Counter(parent_of(v) for v in visible.values())
            for cid,c in visible.items():c['sub_comment_count']=counts[cid]
            row[field]=visible
            if field=='comments':row['comment_count']=len(visible)
            result[pid]=row
    if not limit_comments:return result
    return await asyncio.to_thread(lambda: {pid:select_for_env(env,p,field=field,anchors=(anchors or {}).get(pid,())) for pid,p in result.items()})
