"""One bounded budget per logical model request, shared across simulation entry points."""
from contextlib import contextmanager
from contextvars import ContextVar
import asyncio
import os
import time
import uuid

from loguru import logger

MAX_MODEL_ATTEMPTS = 20  # Includes the first request, not twenty additional retries.
_single_attempt = ContextVar('onesim_single_model_attempt', default=False)
_audit_owner = ContextVar('onesim_model_audit_owner', default=None)


def model_attempt_limit():
    # Legacy per-kind FORMAT_RETRIES / topic failure limits no longer override this policy.
    try:
        return max(1, min(MAX_MODEL_ATTEMPTS, int(os.environ.get('ONESIM_LLM_MAX_ATTEMPTS', '20'))))
    except ValueError:
        return MAX_MODEL_ATTEMPTS


def single_attempt_active():
    return _single_attempt.get()


@contextmanager
def single_model_attempt():
    """Tell routing and transport layers that the caller owns the retry budget."""
    token = _single_attempt.set(True)
    try:
        yield
    finally:
        _single_attempt.reset(token)


class ModelRequestSkipped(RuntimeError):
    """Exhausted request with no safe empty schema: skip its event, not the run."""


@contextmanager
def model_audit_scope(owner):
    """Let memory internals record failures alongside the owning simulation."""
    token = _audit_owner.set(owner)
    try:
        yield
    finally:
        _audit_owner.reset(token)


async def validated_auxiliary_request(model, prompt, validator, *, kind, owner, agent_id=None):
    """Bound auxiliary memory calls without inventing scores or reflection text."""
    from .reaction_failure import register_output_failure
    from .llm_validation import response_metadata

    request_id, limit = str(uuid.uuid4()), model_attempt_limit()
    for attempt in range(1, limit + 1):
        stage = 'transport'
        try:
            with single_model_attempt():
                response = await model.acall(prompt)
            stage = 'output'
            if response_metadata(response).get('finish_reason') == 'length':
                raise ValueError('Response reached its output token limit')
            return validator(response)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning('{} request={} attempt={}/{} failed: {}', kind, request_id, attempt, limit, exc)
            if attempt == limit:
                audit_owner = _audit_owner.get()
                if audit_owner is None:
                    audit_owner = owner
                register_output_failure(audit_owner, {
                    'agent_id': agent_id, 'kind': kind, 'request_id': request_id,
                    'attempts': attempt, 'error': str(exc)[:1000], 'error_stage': stage,
                    'fallback': 'skip_' + kind, 'step': getattr(audit_owner, 'current_step', None),
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                })
                raise ModelRequestSkipped(f'{kind} skipped after {attempt} attempts: {exc}') from exc
            delay = max(0.0, float(os.environ.get('ONESIM_LLM_RETRY_WAIT_SECONDS', '2')))
            await asyncio.sleep(min(delay * 2 ** (attempt - 1), 30.0))
