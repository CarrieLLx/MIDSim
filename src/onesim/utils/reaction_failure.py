"""Explicit skip markers and audit of exhausted model requests; no topic stop quota."""
import json
import os
from pathlib import Path
import re
import tempfile

from loguru import logger
from .llm_validation import observation_posts


class SkippedReaction(dict):
    """Internal no-action result, never a valid model decision or a new memory."""


class SkippedMemory:
    """Internal marker: omit the memory write and keep the successful reaction."""


class SkippedRecommendation(dict):
    """Internal empty ranking, never an accepted model selection."""


def skipped_social_reaction(instruction, observation):
    """Skip only known comment or followed-profile review contracts."""
    if not all('"' + field + '"' in instruction for field in ('decisions', 'post_id', 'comment')):
        if all('"' + field + '"' in instruction for field in ('has_profile_visit', 'profile_visits')):
            try:
                data = json.loads(observation or '')
            except (TypeError, ValueError):
                return None
            if isinstance(data, dict) and isinstance(data.get('followed_accounts'), list):
                # The handler consumes this marker before validating account rows.
                # No artificial interest classification is assigned to anyone.
                return SkippedReaction(has_profile_visit=False, profile_visits=[])
        return None
    observation = observation or ''
    notifications = re.split(r'(?m)^\s*## Notification \d+\s*$', observation)
    if len(notifications) > 1:
        post_ids = []
        for notification in notifications[1:]:
            posts = observation_posts(notification)
            if len(posts) != 1:
                return None
            # Preserve repeated posts: each notification needs its own false row
            # so the handler acknowledges it instead of retrying it next round.
            post_ids.append(next(iter(posts)))
    else:
        post_ids = list(observation_posts(observation))
    if not post_ids:
        return None
    return SkippedReaction(
        decisions=[dict(post_id=pid, comment=False, parent_comment_id=None,
                        comment_content='', decision_reason='system_skip: invalid_model_output',
                        expression_reason='', bookmark=False, bookmark_reason='',
                        mention_reasoning=[]) for pid in post_ids],
        search=False, search_post_id=None, search_keyword='', search_reason='',
        has_profile_visit=False, profile_visits=[],
    )


def register_output_failure(env, failure):
    """Audit each exhausted logical request once; failures never accumulate into a stop.

    Keep the legacy counter and filename for existing analysis readers. The kind
    now also distinguishes recommendation and planning requests.
    """
    failures = getattr(env, '_reaction_output_failures', None)
    if failures is None:
        failures = []
        env._reaction_output_failures = failures
    for existing in failures:
        if existing['request_id'] == failure['request_id']:
            return existing
    count = len(failures) + 1
    record = dict(failure, count=count, stop_at=None, disposition='skipped')
    failures.append(record)
    # Appending before any await keeps the audit sequence consistent for
    # simultaneous users in the environment's event loop.
    directory = getattr(env, 'output_dir', None)
    if directory:
        temporary = None
        try:
            path = Path(directory) / 'reaction_output_errors.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False,
                                             dir=path.parent, prefix='.reaction_output_errors.',
                                             suffix='.tmp') as f:
                temporary = f.name
                json.dump(dict(failure_count=count, stop_at=None,
                               skipped_count=sum(r['disposition']=='skipped' for r in failures),
                               failures=failures), f, ensure_ascii=False, indent=2)
            os.replace(temporary, path)
            temporary = None
        except OSError as exc:
            logger.warning('Could not save model-output error audit: {}', exc)
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
    return record
