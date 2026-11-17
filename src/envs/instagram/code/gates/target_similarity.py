"""Compare visible response targets with individual memory passages."""
import json
from ..embedding import EmbeddingService, cosine_similarity
from ..embedding.service import _chunk_text


def compare_targets(cfg, topic_text, memory_text):
    try:
        targets = json.loads(topic_text)
        if not isinstance(targets, list):
            raise ValueError()
    except (ValueError, TypeError):
        targets = [{'target': 'input', 'text': topic_text}]
    memories = [p for line in memory_text.split('\n') if line.strip()
                for p in _chunk_text(line, cfg.embed_max_chars)]
    # No global first-N truncation: later targets and memories must remain visible.
    passages = [[p for p in _chunk_text(t['text'], cfg.embed_max_chars)] for t in targets]
    texts = list(dict.fromkeys(memories + [p for group in passages for p in group]))
    vectors = {}
    service = EmbeddingService(cfg.embedding_config_path)
    for start in range(0, len(texts), 32):
        batch = texts[start:start + 32]
        result = service.embed(batch)
        if len(result) != len(batch):
            raise ValueError('Embedding response count mismatch')
        vectors.update(zip(batch, result))
    rows = []
    for target, chunks in zip(targets, passages):
        best = None
        for chunk in chunks:
            for memory in memories:
                score = cosine_similarity(vectors[chunk], vectors[memory])
                if best is None or score > best[0]:
                    best = (score, chunk, memory)
        rows.append({**target, 'similarity': best[0] if best else None,
                     'matched_input': best[1] if best else None,
                     'matched_memory': best[2] if best else None,
                     'inject': bool(best and best[0] >= cfg.embed_threshold)})
    scores = [r['similarity'] for r in rows if r['similarity'] is not None]
    return {'inject': any(r['inject'] for r in rows),
            'similarity': max(scores) if scores else None,
            'threshold': cfg.embed_threshold, 'targets': rows,
            'aggregation': 'max_target_passage_to_memory_passage',
            'memory_passages': memories}
