"""Validate recommendation output before it can become a feed selection."""
from typing import Any, Collection, Optional


class RecommendationFormatError(ValueError):
    """The model did not return a usable recommendation ranking."""


def validate_recommendation(
    value: Any, candidate_post_ids: Optional[Collection[str]] = None
) -> dict:
    if not isinstance(value, dict):
        raise RecommendationFormatError("Return a JSON object, not an array or text.")
    fields = ("ranked_post_ids", "selected_post_ids")
    if candidate_post_ids is None:
        fields += (
            "ranked_note_ids", "selected_note_ids", "ranked_tweet_ids",
            "selected_tweet_ids", "ranked_blog_ids", "selected_blog_ids",
        )
    field = next((name for name in fields if name in value), None)
    if field is None:
        raise RecommendationFormatError(f"Missing ranking field: {fields[0]}.")
    ids = value[field]
    if not isinstance(ids, list) or any(not isinstance(pid, str) or not pid.strip() for pid in ids):
        raise RecommendationFormatError(f"{field} must be an array of nonempty ID strings.")
    if len(ids) != len(set(ids)):
        raise RecommendationFormatError(f"{field} contains duplicate IDs.")
    if candidate_post_ids is not None:
        candidates = set(candidate_post_ids)
        unknown = set(ids) - candidates
        missing = candidates - set(ids)
        if unknown or missing:
            raise RecommendationFormatError(
                f"Rank every input candidate exactly once: {len(unknown)} unknown IDs, "
                f"{len(missing)} missing IDs. Copy IDs exactly from the input."
            )
    return value
