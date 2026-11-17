from onesim.events import Event
from typing import List, Any, Optional, Dict, Union
from onesim.utils.common import gen_id

PLATFORM_AGENT_ID = "PLATFORM"


class ProfileVisitEvent(Event):
    """A deliberate visit to a user's Replies tab, separate from feed exposure."""
    def __init__(self, from_agent_id, to_agent_id, timestamp, timestamp_duration,
                 current_step, max_step, target_user_id, as_of_ms, reason, source,
                 trigger_post_id=None, trigger_comment_id=None, **kwargs):
        super().__init__(from_agent_id, to_agent_id, **kwargs)
        self.timestamp = timestamp
        self.timestamp_duration = timestamp_duration
        self.current_step = current_step
        self.max_step = max_step
        self.target_user_id = target_user_id
        self.as_of_ms = int(as_of_ms)
        self.reason = reason
        self.source = source
        self.trigger_post_id = trigger_post_id
        self.trigger_comment_id = trigger_comment_id

class ProfileVisitCompletedEvent(Event):
    """Finish a profile-history visit, including visits with no visible replies."""
    def __init__(self, from_agent_id, to_agent_id, timestamp, timestamp_duration,
                 current_step, max_step, **kwargs):
        super().__init__(from_agent_id, to_agent_id, **kwargs)
        self.timestamp = timestamp
        self.timestamp_duration = timestamp_duration
        self.current_step = current_step
        self.max_step = max_step


class AddCommentEvent(Event):
    def __init__(self,
                 from_agent_id: str,
                 to_agent_id: str,
                 source_type: str, 
                 target_type: str,  # "ENV"
                 key: str,
                 value: Any,
                 **kwargs) -> None:

        super().__init__(from_agent_id, to_agent_id, **kwargs)
        self.request_id = kwargs.get("request_id", gen_id())
        self.source_type = source_type
        self.target_type = target_type
        self.key = key
        self.value = value

class AddCommentResponseEvent(Event):
    def __init__(self,
                 from_agent_id: str,
                 to_agent_id: str,
                 request_id: str,
                 key: str,
                 success: bool = True,
                 error: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(from_agent_id, to_agent_id, **kwargs)
        self.request_id = request_id
        self.key = key
        self.success = success
        self.error = error
    
class MentionPoolUpdateEvent(Event):
    def __init__(self,
                 from_agent_id: str,
                 to_agent_id: str,
                 source_type: str, 
                 target_type: str,  # "ENV"
                 key: str,
                 value: Any,
                 target_user_id: Optional[str] = None,
                 comment_id: Optional[str] = None,
                 **kwargs) -> None:

        super().__init__(from_agent_id, to_agent_id, **kwargs)
        self.request_id = kwargs.get("request_id", gen_id())
        self.source_type = source_type
        self.target_type = target_type
        self.key = key
        self.value = value
        self.target_user_id = target_user_id
        self.comment_id = comment_id

class MentionPoolUpdateResponseEvent(Event):
    def __init__(self,
                 from_agent_id: str,
                 to_agent_id: str,
                 request_id: str,
                 key: str,
                 success: bool = True,
                 error: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(from_agent_id, to_agent_id, **kwargs)
        self.request_id = request_id
        self.key = key
        self.success = success
        self.error = error

class StartEvent(Event):
    def __init__(self,
        from_agent_id: str,
        to_agent_id: str,
        timestamp: int,
        timestamp_duration: int,
        current_step: int,
        max_step: int,
        simulation_cap_timestamp: Optional[int] = None,
        current_posts: Dict[str, Dict[str, Any]] = {},
        mentions: Dict[str, Dict[str, Any]] = {},
        **kwargs: Any
    ) -> None:
        rec_id = str(kwargs.pop("recommendation_agent_id", "") or "").strip()
        search_id = str(kwargs.pop("search_agent_id", "") or "").strip()
        super().__init__(from_agent_id=from_agent_id, to_agent_id=to_agent_id, **kwargs)
        self.timestamp = timestamp
        self.timestamp_duration = timestamp_duration
        self.simulation_cap_timestamp = simulation_cap_timestamp
        self.current_step = current_step
        self.max_step = max_step
        self.current_posts = current_posts
        self.mentions = mentions
        self.recommendation_agent_id = rec_id
        self.search_agent_id = search_id

class SocialRecommendationEvent(Event):
    """Social recommendation event"""
    def __init__(self, 
        from_agent_id: str, 
        to_agent_id: str, 
        timestamp: int,
        current_step: int,
        max_step: int,
        timestamp_duration: int = 0,
        recommendations: Dict[str, Dict[str, Any]] = {},
        **kwargs: Any) -> None:
        super().__init__(from_agent_id=from_agent_id, to_agent_id=to_agent_id, **kwargs)
        self.timestamp = timestamp
        self.timestamp_duration = int(timestamp_duration or 0)
        self.current_step = current_step
        self.max_step = max_step
        self.recommendations = recommendations

class BookmarkEvent(Event):
    """Keep following event"""
    def __init__(self,
        from_agent_id: str,
        to_agent_id: str,
        timestamp: int,
        current_step: int,
        max_step: int,
        timestamp_duration: int = 0,
        recommendations: Dict[str, Dict[str, Any]] = {},
        **kwargs: Any) -> None:
        super().__init__(from_agent_id=from_agent_id, to_agent_id=to_agent_id, **kwargs)
        self.timestamp = timestamp
        self.timestamp_duration = int(timestamp_duration or 0)
        self.current_step = current_step
        self.max_step = max_step
        self.recommendations = recommendations

class AlgorithmRecommendationEvent(Event):
    """Algorithm recommendation event"""
    def __init__(self, 
        from_agent_id: str, 
        to_agent_id: str, 
        timestamp: int,
        current_step: int,
        max_step: int,
        timestamp_duration: int = 0,
        recommendations: Dict[str, Dict[str, Any]] = {},
        **kwargs: Any) -> None:
        super().__init__(from_agent_id=from_agent_id, to_agent_id=to_agent_id, **kwargs)
        self.timestamp = timestamp
        self.timestamp_duration = int(timestamp_duration or 0)
        self.current_step = current_step
        self.max_step = max_step
        self.recommendations = recommendations

class GetAlgorithmRecommendationEvent(Event):
    """User request algorithm recommendation event"""
    def __init__(self,
        from_agent_id: str,
        to_agent_id: str,
        timestamp: int,
        current_step: int,
        max_step: int,
        user_profile: Dict[str, Any],
        current_posts: Dict[str, Dict[str, Any]],
        recommended_post_ids: List[str],
        timestamp_duration: int = 0,
        algorithm_type: str = "",
        **kwargs: Any) -> None:
        super().__init__(from_agent_id=from_agent_id, to_agent_id=to_agent_id, **kwargs)
        self.timestamp = timestamp
        self.timestamp_duration = int(timestamp_duration or 0)
        self.current_step = current_step
        self.max_step = max_step
        self.user_profile = user_profile
        self.current_posts = current_posts
        self.recommended_post_ids = recommended_post_ids
        self.type = algorithm_type

class GetSearchResultEvent(Event):
    """User request search event"""
    def __init__(self,
        from_agent_id: str,
        to_agent_id: str,
        timestamp: int,
        current_step: int,
        max_step: int,
        user_profile: Dict[str, Any],
        timestamp_duration: int = 0,
        algorithm_type: str = "",
        search_query: str = "",
        **kwargs: Any) -> None:
        super().__init__(from_agent_id=from_agent_id, to_agent_id=to_agent_id, **kwargs)
        self.timestamp = timestamp
        self.timestamp_duration = int(timestamp_duration or 0)
        self.current_step = current_step
        self.max_step = max_step
        self.user_profile = user_profile
        self.type = algorithm_type
        self.search_query = (search_query or "").strip()

class SearchResultEvent(Event):
    """Search result event"""
    def __init__(self, 
        from_agent_id: str, 
        to_agent_id: str, 
        timestamp: int,
        current_step: int,
        max_step: int,
        timestamp_duration: int = 0,
        recommendations: Dict[str, Dict[str, Any]] = {},
        **kwargs: Any) -> None:
        super().__init__(from_agent_id=from_agent_id, to_agent_id=to_agent_id, **kwargs)
        self.timestamp = timestamp
        self.timestamp_duration = int(timestamp_duration or 0)
        self.current_step = current_step
        self.max_step = max_step
        self.recommendations = recommendations

class RecommendationSpreadingEvent(Event):
    """Recommendation spreading event"""
    def __init__(self,
        from_agent_id: str,
        to_agent_id: str,
        timestamp: int,
        current_step: int,
        max_step: int,
        timestamp_duration: int = 0,
        **kwargs: Any) -> None:
        super().__init__(from_agent_id=from_agent_id, to_agent_id=to_agent_id, **kwargs)
        self.timestamp = timestamp
        self.timestamp_duration = int(timestamp_duration or 0)
        self.current_step = current_step
        self.max_step = max_step

class MentionEvent(Event):
    """@ mention event, sent when user is @, commented or replied to"""
    def __init__(self,
        from_agent_id: str,
        to_agent_id: str,
        timestamp: int,
        current_step: int,
        max_step: int,
        timestamp_duration: int = 0,
        mentions: Dict[str, Dict[str, Any]] = {},
        **kwargs: Any) -> None:
        super().__init__(from_agent_id=from_agent_id, to_agent_id=to_agent_id, **kwargs)
        self.timestamp = timestamp
        self.timestamp_duration = int(timestamp_duration or 0)
        self.current_step = current_step
        self.max_step = max_step
        self.mentions = mentions

class MentionSpreadingEvent(Event):
    """Mention spreading event"""
    def __init__(self,
        from_agent_id: str,
        to_agent_id: str,
        timestamp: int,
        current_step: int,
        max_step: int,
        timestamp_duration: int = 0,
        **kwargs: Any) -> None:
        super().__init__(from_agent_id=from_agent_id, to_agent_id=to_agent_id, **kwargs)
        self.timestamp = timestamp
        self.timestamp_duration = int(timestamp_duration or 0)
        self.current_step = current_step
        self.max_step = max_step
