from onesim.events import Event
from typing import Dict, List, Any
from datetime import datetime
from typing import List, Any, Optional, Dict, Union
from onesim.utils.common import gen_id

class AddRepostEvent(Event):
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

class AddRepostResponseEvent(Event):
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
        simulation_cap_timestamp: int,
        current_step: int,
        max_step: int,
        current_blogs: Dict[str, Dict[str, Any]] = {},
        mentions: Dict[str, Dict[str, Any]] = {},
        **kwargs: Any
    ) -> None:
        super().__init__(from_agent_id=from_agent_id, to_agent_id=to_agent_id, **kwargs)
        self.timestamp = timestamp
        self.timestamp_duration = timestamp_duration
        self.simulation_cap_timestamp = simulation_cap_timestamp
        self.current_step = current_step
        self.max_step = max_step
        self.current_blogs = current_blogs
        self.mentions = mentions

class SocialRecommendationEvent(Event):
    """社交推荐事件"""
    def __init__(self, 
        from_agent_id: str, 
        to_agent_id: str, 
        timestamp: int,
        timestamp_duration: int,
        current_step: int,
        max_step: int,
        recommendations: Dict[str, Dict[str, Any]] = {},
        **kwargs: Any) -> None:
        super().__init__(from_agent_id=from_agent_id, to_agent_id=to_agent_id, **kwargs)
        self.timestamp = timestamp
        self.timestamp_duration = timestamp_duration
        self.current_step = current_step
        self.max_step = max_step
        self.recommendations = recommendations

class KeepFollowingEvent(Event):
    """保持关注事件"""
    def __init__(self,
        from_agent_id: str,
        to_agent_id: str,
        timestamp: int,
        timestamp_duration: int,
        current_step: int,
        max_step: int,
        recommendations: Dict[str, Dict[str, Any]] = {},
        **kwargs: Any) -> None:
        super().__init__(from_agent_id=from_agent_id, to_agent_id=to_agent_id, **kwargs)
        self.timestamp = timestamp
        self.timestamp_duration = timestamp_duration
        self.current_step = current_step
        self.max_step = max_step
        self.recommendations = recommendations

class AlgorithmRecommendationEvent(Event):
    """算法推荐事件"""
    def __init__(self, 
        from_agent_id: str, 
        to_agent_id: str, 
        timestamp: int,
        timestamp_duration: int,
        current_step: int,
        max_step: int,
        recommendations: Dict[str, Dict[str, Any]] = {},
        **kwargs: Any) -> None:
        super().__init__(from_agent_id=from_agent_id, to_agent_id=to_agent_id, **kwargs)
        self.timestamp = timestamp
        self.timestamp_duration = timestamp_duration
        self.current_step = current_step
        self.max_step = max_step
        self.recommendations = recommendations

class GetAlgorithmRecomendationEvent(Event):
    """用户请求算法推荐事件（携带用户画像与指定算法类型）"""
    def __init__(self,
        from_agent_id: str,
        to_agent_id: str,
        timestamp: int,
        timestamp_duration: int,
        current_step: int,
        max_step: int,
        user_profile: Dict[str, Any],
        current_blogs: Dict[str, Dict[str, Any]],
        recommended_blog_ids: List[str],
        algorithm_type: str = "",
        **kwargs: Any) -> None:
        super().__init__(from_agent_id=from_agent_id, to_agent_id=to_agent_id, **kwargs)
        self.timestamp = timestamp
        self.timestamp_duration = timestamp_duration
        self.current_step = current_step
        self.max_step = max_step
        self.user_profile = user_profile
        self.current_blogs = current_blogs
        self.recommended_blog_ids = recommended_blog_ids
        self.type = algorithm_type

class SearchEvent(Event):
    """用户请求搜索事件（携带用户画像与指定搜索类型）"""
    def __init__(self,
        from_agent_id: str,
        to_agent_id: str,
        timestamp: int,
        timestamp_duration: int,
        current_step: int,
        max_step: int,
        user_profile: Dict[str, Any],
        algorithm_type: str = "",
        search_keyword: str = "",
        **kwargs: Any) -> None:
        super().__init__(from_agent_id=from_agent_id, to_agent_id=to_agent_id, **kwargs)
        self.timestamp = timestamp
        self.timestamp_duration = timestamp_duration
        self.current_step = current_step
        self.max_step = max_step
        self.user_profile = user_profile
        self.type = algorithm_type
        self.search_keyword = search_keyword

class SearchRecommendationEvent(Event):
    """搜索推荐事件"""
    def __init__(self, 
        from_agent_id: str, 
        to_agent_id: str, 
        timestamp: int,
        timestamp_duration: int,
        current_step: int,
        max_step: int,
        recommendations: Dict[str, Dict[str, Any]] = {},
        **kwargs: Any) -> None:
        super().__init__(from_agent_id=from_agent_id, to_agent_id=to_agent_id, **kwargs)
        self.timestamp = timestamp
        self.timestamp_duration = timestamp_duration
        self.current_step = current_step
        self.max_step = max_step
        self.recommendations = recommendations

class RecommendationSpreadingEvent(Event):
    """传播内容更新事件"""
    def __init__(self,
        from_agent_id: str,
        to_agent_id: str,
        timestamp: int,
        timestamp_duration: int,
        current_step: int,
        max_step: int,
        **kwargs: Any) -> None:
        super().__init__(from_agent_id=from_agent_id, to_agent_id=to_agent_id, **kwargs)
        self.timestamp = timestamp
        self.timestamp_duration = timestamp_duration
        self.current_step = current_step
        self.max_step = max_step

class MentionEvent(Event):
    """@提醒事件，当用户被@、被评论或被回复时发送"""
    def __init__(self,
        from_agent_id: str,
        to_agent_id: str,
        timestamp: int,
        timestamp_duration: int,
        current_step: int,
        max_step: int,
        mentions: Dict[str, Dict[str, Any]] = {},
        **kwargs: Any) -> None:
        super().__init__(from_agent_id=from_agent_id, to_agent_id=to_agent_id, **kwargs)
        self.timestamp = timestamp
        self.timestamp_duration = timestamp_duration
        self.current_step = current_step
        self.max_step = max_step
        self.mentions = mentions

class MentionSpreadingEvent(Event):
    """提醒传播内容更新事件"""
    def __init__(self,
        from_agent_id: str,
        to_agent_id: str,
        timestamp: int,
        timestamp_duration: int,
        current_step: int,
        max_step: int,
        **kwargs: Any) -> None:
        super().__init__(from_agent_id=from_agent_id, to_agent_id=to_agent_id, **kwargs)
        self.timestamp = timestamp
        self.timestamp_duration = timestamp_duration
        self.current_step = current_step
        self.max_step = max_step