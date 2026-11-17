# Model requests: at most 20 attempts, then skip

The shared model-request budget covers recommendation ranking, user reactions, memory generation, memory importance scoring, reflection questions and insights, and planning.

The maximum of 20 attempts includes the first request. Retries retain the same logical input and candidate set; they may add output-validation feedback, but do not resample exposures or interaction budgets. Invalid JSON, failed validation, truncated output, timeouts and connection errors consume the same budget.

When the budget is exhausted:

- Recommendation returns a system-skip marker and produces no recommendation for that candidate group.
- A user reaction performs no comment, bookmark or search action and creates no fabricated memory. Notifications are acknowledged in order.
- Failed memory generation skips that memory while preserving any successful reaction.
- Failed importance scoring marks the score as skipped and retains the memory text; it does not assign a default score or repeatedly retry the same exhausted item in the current strategy instance.
- Failed reflection skips that reflection and retains the original short-term memories without adding placeholder insights.
- Failed planning is omitted while normal reaction generation continues.
- A custom reaction without a safe empty result raises `ModelRequestSkipped` to skip the current event.

Accumulated model failures do not terminate a topic. Task cancellation, user interruption and actual non-model programming errors retain their separate handling.

While the main request owns a retry budget, the load balancer selects one endpoint per attempt, the vLLM adapter avoids nested network retries, and the SDK uses `max_retries=0`. This prevents multiplying retries across endpoints and layers.

`ONESIM_LLM_MAX_ATTEMPTS` can lower the budget for testing; its upper limit is 20. Missing or invalid values use 20. Legacy format-retry and reaction-failure-limit variables do not override this budget. `ONESIM_LLM_RETRY_WAIT_SECONDS` sets the backoff base, defaulting to two seconds, with each wait capped at 30 seconds.

Main reaction attempts are recorded in `decisions_*.json`. Exhaustion records `skipped_error`, `system_skipped=true` and failed response validation. Planning and auxiliary-memory attempts are recorded in run logs. The cumulative audit remains `reaction_output_errors.json`, including recommendation, planning and auxiliary-memory failures, with `stop_at=null`.

These skips reflect system failures rather than an agent's voluntary choice not to participate; account for them separately when interpreting results.
