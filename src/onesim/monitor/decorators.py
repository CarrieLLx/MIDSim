from typing import List, Dict, Any, Callable, Optional
from functools import wraps
from .metric import MetricDefinition, VariableSpec

def metric(
    name: str,
    description: str,
    variables: List[Dict],
    visualization_type: str = "line",
    update_interval: int = 60,
    visualization_config: Optional[Dict] = None
) -> Callable:
    """
    Metric definition decorator, simplifying the process of creating metrics for users
    """
    def decorator(func: Callable) -> Callable:
        # Convert variable specifications
        variable_specs = []
        for var in variables:
            variable_specs.append(VariableSpec(
                name=var["name"],
                source_type=var["source_type"],
                path=var["path"],
                agent_type=var.get("agent_type"),
                required=var.get("required", True)
            ))
            
        # Create metric definition
        metric_def = MetricDefinition(
            name=name,
            description=description,
            visualization_type=visualization_type,
            variables=variable_specs,
            calculation_function=func,
            update_interval=update_interval,
            visualization_config=visualization_config or {}
        )
        
        # Attach the metric definition to the function
        func.metric_definition = metric_def
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
            
        wrapper.metric_definition = metric_def
        return wrapper
        
    return decorator


def register_custom_metric(
    name: str,
    description: str,
    variables: List[VariableSpec],
    calculation_function: Callable,
    visualization_type: str = "line",
    update_interval: int = 60,
    visualization_config: Optional[Dict] = None
) -> MetricDefinition:
    """
    Register custom metric
    """
    return MetricDefinition(
        name=name,
        description=description,
        visualization_type=visualization_type,
        variables=variables,
        calculation_function=calculation_function,
        update_interval=update_interval,
        visualization_config=visualization_config or {}
    ) 