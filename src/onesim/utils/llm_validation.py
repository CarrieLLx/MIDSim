"""Validate model responses without turning failures into simulated inactivity."""
import json
import re


class LLMRequestError(RuntimeError):
    """A logical model request exhausted its retries."""


def json_objects(text):
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(text or ''):
        match = re.search(r'[\[{]', text[pos:])
        if not match:
            return
        start = pos + match.start()
        try:
            value, length = decoder.raw_decode(text[start:])
        except ValueError:
            pos = start + 1
            continue
        yield value
        pos = start + length


def observation_posts(observation):
    result = {}

    def visit(value):
        if isinstance(value, dict):
            if 'comment_id' not in value and ('content' in value or 'desc' in value):
                pid = value.get('post_id') or (value.get('id') if 'comments' in value else None)
                if pid is not None:
                    result[str(pid)] = value
            for child in value.values():
                if isinstance(child, (dict, list)):
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for obj in json_objects(observation or ''):
        visit(obj)
    return result


def validate_reaction(value, instruction='', observation=''):
    if not isinstance(value, dict) or not value:
        raise ValueError('Return a nonempty JSON object in the requested schema.')
    expects_decisions = '"decisions"' in instruction
    if expects_decisions or 'decisions' in value:
        decisions = value.get('decisions')
        if not isinstance(decisions, list):
            raise ValueError('decisions must be an array; include explicit false decisions for skipped input items.')
        posts = observation_posts(observation)
        seen = set()
        for row in decisions:
            if not isinstance(row, dict):
                raise ValueError('Every decision must be a JSON object.')
            # Other environments may use note/tweet IDs and a different action schema.
            if '"post_id"' not in instruction and 'post_id' not in row:
                continue
            pid = row.get('post_id')
            if not isinstance(pid, str) or not pid.strip():
                raise ValueError('Each decision requires the exact input post_id string.')
            seen.add(pid)
            if type(row.get('comment')) is not bool:
                raise ValueError('comment must be a JSON boolean, not a string or missing field.')
            content = row.get('comment_content')
            if not isinstance(content, str):
                raise ValueError('comment_content must be a string.')
            if row['comment'] and not content.strip():
                raise ValueError('comment=true requires nonempty comment_content.')
            if '"parent_comment_id"' in instruction and 'parent_comment_id' not in row:
                raise ValueError('Every decision requires parent_comment_id, using null for a root or skipped comment.')
            parent = row.get('parent_comment_id')
            if parent is not None and (not isinstance(parent, str) or not parent.strip()):
                raise ValueError('parent_comment_id must be null or a nonempty input comment ID.')
            if not row['comment'] and (content.strip() or row.get('parent_comment_id') is not None):
                raise ValueError('comment=false requires empty content and a null parent.')
            for flag in ('bookmark',):
                if ('"' + flag + '"' in instruction or flag in row) and type(row.get(flag)) is not bool:
                    raise ValueError(f'{flag} must be a JSON boolean.')
            if '"mention_reasoning"' in instruction and not isinstance(row.get('mention_reasoning'), list):
                raise ValueError('mention_reasoning must be an array, including [] when no extra mention is intended.')
        if posts and seen != set(posts):
            raise ValueError('Include a decision for every input post and use no unknown post IDs; skipped posts require comment=false.')
    for flag in ('search', 'has_profile_visit'):
        if ('"' + flag + '"' in instruction or flag in value) and type(value.get(flag)) is not bool:
            raise ValueError(f'{flag} must be a JSON boolean.')
    if value.get('search') is True:
        pid, keyword = value.get('search_post_id'), value.get('search_keyword')
        if not isinstance(pid, str) or not pid.strip() or not isinstance(keyword, str) or not keyword.strip():
            raise ValueError('search=true requires a triggering post ID and a nonempty search_keyword.')
        posts = observation_posts(observation)
        if posts and pid not in posts:
            raise ValueError('search_post_id must be an input post ID.')
    if '"profile_visits"' in instruction:
        visits = value.get('profile_visits')
        if type(value.get('has_profile_visit')) is not bool or not isinstance(visits, list):
            raise ValueError('Return has_profile_visit as a boolean and profile_visits as an array.')
        for row in visits:
            if not isinstance(row, dict) or not isinstance(row.get('user_id'), str) or type(row.get('view_profile')) is not bool:
                raise ValueError('Each profile visit requires user_id and a boolean view_profile.')
        if value['has_profile_visit'] != any(row['view_profile'] for row in visits):
            raise ValueError('has_profile_visit must equal whether any visit is true.')
    return value


def validate_memory(value):
    if not isinstance(value, dict):
        raise ValueError('Return a JSON object with a memory string.')
    memory = value.get('memory', value.get('Memory'))
    if not isinstance(memory, str) or not memory.strip():
        raise ValueError('memory must be a nonempty string describing this event.')
    return memory.strip()


def response_metadata(response):
    raw = getattr(response, 'raw', None)
    if not isinstance(raw, dict):
        return {}
    choices = raw.get('choices') or []
    first = choices[0] if choices and isinstance(choices[0], dict) else {}
    return {key: value for key, value in {
        'response_id': raw.get('id'), 'model': raw.get('model'),
        'selected_model': raw.get('selected_model'),
        'finish_reason': first.get('finish_reason'),
    }.items() if value is not None}


def recommendation_candidates(observation):
    marker = 'Full candidate post_id set (rank only inside this set):'
    if marker in (observation or ''):
        tail = observation.split(marker, 1)[1].lstrip()
        value, _ = json.JSONDecoder().raw_decode(tail)
        if isinstance(value, list) and all(isinstance(pid, str) for pid in value):
            return value
    return None
