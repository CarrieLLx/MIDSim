from typing import Dict, Any, List, Optional, Union, Callable
import math
from loguru import logger

# 数据验证和处理工具函数
def safe_get(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    安全地从字典中获取值，避免KeyError
    
    Args:
        data: 数据字典
        key: 键名
        default: 默认值，当键不存在时返回
        
    Returns:
        数据值或默认值
    """
    if not isinstance(data, dict):
        return default
    return data.get(key, default)

def safe_number(value: Any, default: Union[int, float] = 0) -> Union[int, float]:
    """
    安全地将值转换为数字，处理None和异常情况
    
    Args:
        value: 要转换的值
        default: 转换失败时的默认值
        
    Returns:
        转换后的数字或默认值
    """
    if value is None:
        return default
    try:
        num = float(value)
        # 检查是否为有效数字（非NaN或无穷大）
        if math.isnan(num) or math.isinf(num):
            return default
        return num
    except (ValueError, TypeError):
        return default

def safe_list(value: Any) -> List:
    """
    确保值是列表类型，处理None和非列表值
    
    Args:
        value: 输入值
        
    Returns:
        列表形式的值
    """
    if value is None:
        return []
    if not isinstance(value, list):
        return [value]
    return value

def safe_sum(values: List, default: Union[int, float] = 0) -> Union[int, float]:
    """
    安全地计算列表总和，处理None值和空列表
    
    Args:
        values: 值列表
        default: 列表为空或计算失败时的默认值
        
    Returns:
        列表总和或默认值
    """
    if not values:
        return default
    try:
        # 过滤掉None值并转换为数字
        valid_values = [safe_number(v) for v in values if v is not None]
        return sum(valid_values) if valid_values else default
    except Exception as e:
        logger.error(f"计算总和时出错: {e}")
        return default

def safe_avg(values: List, default: Union[int, float] = 0) -> Union[int, float]:
    """
    安全地计算列表平均值，处理None值、空列表和除零错误
    
    Args:
        values: 值列表
        default: 列表为空或计算失败时的默认值
        
    Returns:
        列表平均值或默认值
    """
    if not values:
        return default
    try:
        # 过滤掉None值并转换为数字
        valid_values = [safe_number(v) for v in values if v is not None]
        return sum(valid_values) / len(valid_values) if valid_values else default
    except Exception as e:
        logger.error(f"计算平均值时出错: {e}")
        return default

def safe_max(values: List, default: Union[int, float] = 0) -> Union[int, float]:
    """
    安全地计算列表最大值，处理None值和空列表
    
    Args:
        values: 值列表
        default: 列表为空或计算失败时的默认值
        
    Returns:
        列表最大值或默认值
    """
    if not values:
        return default
    try:
        # 过滤掉None值并转换为数字
        valid_values = [safe_number(v) for v in values if v is not None]
        return max(valid_values) if valid_values else default
    except Exception as e:
        logger.error(f"计算最大值时出错: {e}")
        return default

def safe_min(values: List, default: Union[int, float] = 0) -> Union[int, float]:
    """
    安全地计算列表最小值，处理None值和空列表
    
    Args:
        values: 值列表
        default: 列表为空或计算失败时的默认值
        
    Returns:
        列表最小值或默认值
    """
    if not values:
        return default
    try:
        # 过滤掉None值并转换为数字
        valid_values = [safe_number(v) for v in values if v is not None]
        return min(valid_values) if valid_values else default
    except Exception as e:
        logger.error(f"计算最小值时出错: {e}")
        return default

def safe_count(values: List, predicate: Callable = None) -> int:
    """
    安全地计算列表中满足条件的元素数量
    
    Args:
        values: 值列表
        predicate: 判断函数，默认计算非None元素数量
        
    Returns:
        满足条件的元素数量
    """
    if not values:
        return 0
    if predicate is None:
        return len([v for v in values if v is not None])
    try:
        return len([v for v in values if v is not None and predicate(v)])
    except Exception as e:
        logger.error(f"计算计数时出错: {e}")
        return 0

def log_metric_error(metric_name: str, error: Exception, context: Dict = None):
    """
    记录指标计算错误
    
    Args:
        metric_name: 指标名称
        error: 异常对象
        context: 上下文信息，例如输入数据
    """
    error_msg = f"计算指标 {metric_name} 时出错: {error}"
    if context:
        # 限制上下文大小以避免日志爆炸
        context_str = str(context)
        if len(context_str) > 500:
            context_str = context_str[:500] + "..."
        error_msg += f", 上下文: {context_str}"
    logger.error(error_msg)


_HEAVY_EMBEDDING_KEYS = frozenset({
    "_comment_embeddings",
    "_repost_embeddings",
    "content_pool_with_embeddings",
})


def _is_embedding_vector(value: Any) -> bool:
    if not isinstance(value, list) or len(value) < 4:
        return False
    sample = value[: min(8, len(value))]
    return all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in sample)


def contains_embedding_payload(obj: Any, _depth: int = 0) -> bool:
    """Return True if obj (recursively) contains embedding vector payloads."""
    if _depth > 12:
        return False
    if isinstance(obj, dict):
        if any(k in _HEAVY_EMBEDDING_KEYS for k in obj):
            return True
        if _is_embedding_vector(obj.get("embedding")):
            return True
        return any(contains_embedding_payload(v, _depth + 1) for v in obj.values())
    if isinstance(obj, list):
        if _is_embedding_vector(obj):
            return True
        if not obj:
            return False
        if isinstance(obj[0], dict) and _is_embedding_vector(obj[0].get("embedding")):
            return True
        return any(contains_embedding_payload(v, _depth + 1) for v in obj[:32])
    return False


def _summarize_comment_embeddings(items: Any) -> Dict[str, Any]:
    if not isinstance(items, list):
        return {"_summary": "_comment_embeddings", "count": 0}
    dim: Optional[int] = None
    sample: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        emb = item.get("embedding")
        if dim is None and _is_embedding_vector(emb):
            dim = len(emb)
        if len(sample) < 3:
            sample.append(
                {k: item.get(k) for k in ("note_id", "comment_id", "blog_id", "tweet_id", "text") if k in item}
            )
    return {"_summary": "_comment_embeddings", "count": len(items), "dim": dim, "sample": sample}


def _summarize_content_pool_with_embeddings(pool: Any) -> Dict[str, Any]:
    if not isinstance(pool, dict):
        return {"_summary": "content_pool_with_embeddings", "type": type(pool).__name__}
    comment_count = 0
    embedded_count = 0
    for note in pool.values():
        if not isinstance(note, dict):
            continue
        comments = note.get("comments") or {}
        if isinstance(comments, dict):
            iterable = comments.values()
        elif isinstance(comments, list):
            iterable = comments
        else:
            continue
        for comment in iterable:
            comment_count += 1
            if isinstance(comment, dict) and _is_embedding_vector(comment.get("embedding")):
                embedded_count += 1
    return {
        "_summary": "content_pool_with_embeddings",
        "note_count": len(pool),
        "comment_count": comment_count,
        "embedded_comment_count": embedded_count,
    }


def _summarize_for_log(obj: Any, _depth: int = 0) -> Any:
    if _depth > 12:
        return "..."
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for key, value in obj.items():
            if key in _HEAVY_EMBEDDING_KEYS:
                if key in ("_comment_embeddings", "_repost_embeddings"):
                    out[key] = _summarize_comment_embeddings(value)
                else:
                    out[key] = _summarize_content_pool_with_embeddings(value)
            elif key == "embedding" and _is_embedding_vector(value):
                out[key] = f"<vector dim={len(value)}>"
            else:
                out[key] = _summarize_for_log(value, _depth + 1)
        return out
    if isinstance(obj, list):
        if _is_embedding_vector(obj):
            return f"<vector dim={len(obj)}>"
        if obj and isinstance(obj[0], dict) and _is_embedding_vector(obj[0].get("embedding")):
            return _summarize_comment_embeddings(obj)
        if len(obj) > 20 and all(isinstance(x, (dict, list, str, int, float, bool)) or x is None for x in obj):
            summarized = [_summarize_for_log(x, _depth + 1) for x in obj[:5]]
            return {"_summary": "list", "count": len(obj), "head": summarized}
        return [_summarize_for_log(x, _depth + 1) for x in obj]
    return obj


def summarize_metric_result_for_log(result: Any) -> Any:
    """Return a compact representation for logging when result contains embeddings."""
    if result is None or not contains_embedding_payload(result):
        return result
    return _summarize_for_log(result)


def create_line_chart_option(title: str, x_axis_data: List[str], series_data: List[Any], series_name: str = "Value") -> Dict[str, Any]:
    """Creates a minimal ECharts line chart option with only essential data."""
    x_axis_data = x_axis_data if x_axis_data is not None else []
    series_data = series_data if series_data is not None else []
    
    return {
        "xAxis": {"data": x_axis_data},
        "series": [{"name": series_name, "type": "line", "data": series_data}]
    }

def create_pie_chart_option(title: str, series_data: List[Dict[str, Any]], series_name: str = "Distribution") -> Dict[str, Any]:
    """Creates a minimal ECharts pie chart option with only essential data."""
    series_data = series_data if series_data is not None else []
    
    return {
        "series": [{
            "name": series_name,
            "type": "pie",
            "data": series_data
        }]
    }

def create_bar_chart_option(title: str, x_axis_data: List[str], series_data: List[Any], series_name: str = "Value") -> Dict[str, Any]:
    """Creates a minimal ECharts bar chart option with only essential data."""
    x_axis_data = x_axis_data if x_axis_data is not None else []
    series_data = series_data if series_data is not None else []
    
    if not series_data or not isinstance(series_data[0], dict):
        series_list = [{
            "name": series_name,
            "type": "bar",
            "data": series_data
        }]
    else:
        series_list = series_data
        
    return {
        "xAxis": {"data": x_axis_data},
        "series": series_list
    }

def create_time_series_chart_option(title: str, series_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Creates a minimal ECharts time-series chart option with only essential data."""
    series_list = series_list if series_list is not None else []
    
    for s in series_list:
        s["type"] = "line"
        
    return {
        "xAxis": {"type": 'time'},
        "series": series_list
    }


# --- End ECharts Option Helpers --- 