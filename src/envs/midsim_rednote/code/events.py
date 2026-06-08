from onesim.events import Event
from typing import Dict, List, Any
from datetime import datetime
from typing import List, Any, Optional, Dict, Union
from onesim.utils.common import gen_id

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
                 **kwargs) -> None:

        super().__init__(from_agent_id, to_agent_id, **kwargs)
        self.request_id = kwargs.get("request_id", gen_id())
        self.source_type = source_type
        self.target_type = target_type
        self.key = key
        self.value = value

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
        current_notes: Dict[str, Dict[str, Any]] = {},
        mentions: Dict[str, Dict[str, Any]] = {},
        **kwargs: Any
    ) -> None:
        super().__init__(from_agent_id=from_agent_id, to_agent_id=to_agent_id, **kwargs)
        self.timestamp = timestamp
        self.timestamp_duration = timestamp_duration
        self.simulation_cap_timestamp = simulation_cap_timestamp
        self.current_step = current_step
        self.max_step = max_step
        self.current_notes = current_notes
        self.mentions = mentions

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

class KeepFollowingEvent(Event):
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

class GetAlgorithmRecomendationEvent(Event):
    """User request algorithm recommendation event"""
    def __init__(self,
        from_agent_id: str,
        to_agent_id: str,
        timestamp: int,
        current_step: int,
        max_step: int,
        user_profile: Dict[str, Any],
        current_notes: Dict[str, Dict[str, Any]],
        recommended_note_ids: List[str],
        timestamp_duration: int = 0,
        algorithm_type: str = "",
        **kwargs: Any) -> None:
        super().__init__(from_agent_id=from_agent_id, to_agent_id=to_agent_id, **kwargs)
        self.timestamp = timestamp
        self.timestamp_duration = int(timestamp_duration or 0)
        self.current_step = current_step
        self.max_step = max_step
        self.user_profile = user_profile
        self.current_notes = current_notes
        self.recommended_note_ids = recommended_note_ids
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