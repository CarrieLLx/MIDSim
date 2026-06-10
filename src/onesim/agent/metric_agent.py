from typing import Dict, List, Any, Optional, Union, Callable
import re
import os
from loguru import logger
import textwrap
import json
import random
from collections import defaultdict

from onesim.models.core.message import Message
from onesim.models import JsonBlockParser
from onesim.models.parsers import CodeBlockParser
from onesim.monitor.utils import (
    safe_get, safe_number, safe_list, safe_sum, 
    safe_avg, safe_max, safe_min, safe_count, log_metric_error
)
from .base import AgentBase

class MetricAgent(AgentBase):
    """Agent for generating monitoring metrics based on scenario, with enhanced data validation and error handling"""
    
    def __init__(
        self,
        model_config_name: str,
        sys_prompt: str = '',
    ):
        """
        Initialize metric generation agent
        """
        super().__init__(
            sys_prompt=sys_prompt or (
                "You are an AI assistant specialized in analyzing performance metrics for multi-agent systems. "
                "Your task is to generate meaningful monitoring metrics from system descriptions and data models, "
                "and to produce a calculation function for each metric. "
                "Generated functions must include robust error handling for None values, empty lists, type errors, and other edge cases."
            ),
            model_config_name=model_config_name,
        )
        self.parser = JsonBlockParser()
        self.code_parser = CodeBlockParser(language="python")
        self.visualization_types = ["line", "bar", "pie"]
        self.source_types = ["env", "agent"]

    def generate_metrics(self, scenario_description: str, agent_types: List[str], system_data_model: Dict = None, num_metrics: int = 3) -> List[Dict]:
        """
        Analyze scenario, generate applicable metrics list
        """
        if not scenario_description:
            logger.error("Scenario description cannot be empty")
            return []
            
        if not agent_types:
            logger.error("Agent types list cannot be empty")
            return []
            
        prompt = self._create_generation_prompt(scenario_description, agent_types, system_data_model, num_metrics)
        
        # Use model to get response
        prompt_message = self.model.format(
            Message("system", self.sys_prompt, role="system"),
            Message("user", prompt + self.parser.format_instruction, role="user")
        )
        
        response = self.model(prompt_message)
        
        # Parse response, extract metric definitions
        try:
            result = self.parser.parse(response)
            metrics = result.parsed.get("metrics", [])
            logger.info(f"Generated {len(metrics)} metrics for the scenario")
            
            # Validate metrics
            if system_data_model:
                metrics = self.validate_metrics(metrics, system_data_model)
                
            return metrics
        except Exception as e:
            logger.error(f"Error parsing metric generation response: {str(e)}")
            # Try using regex to extract JSON block
            try:
                import re
                json_pattern = r'```json\s*([\s\S]*?)\s*```'
                matches = re.findall(json_pattern, response)
                if matches:
                    import json
                    metrics_data = json.loads(matches[0])
                    metrics = metrics_data.get("metrics", [])
                    logger.info(f"Extracted {len(metrics)} metrics using backup method")
                    
                    # Validate metrics
                    if system_data_model:
                        metrics = self.validate_metrics(metrics, system_data_model)
                        
                    return metrics
            except Exception as backup_error:
                logger.error(f"Backup extraction method also failed: {str(backup_error)}")
            
            return []
    
    def validate_metrics(self, metrics: List[Dict], system_data_model: Dict = None) -> List[Dict]:
        """
        Validate metric definitions, ensure all referenced variables exist
        """
        if system_data_model is None:
            logger.warning("Cannot validate metrics, system_data_model not provided")
            return metrics
        
        available_variables = set()
        
        # Collect all available variables
        if "environment" in system_data_model and "variables" in system_data_model["environment"]:
            for var in system_data_model["environment"]["variables"]:
                available_variables.add(var["name"])
        
        if "agents" in system_data_model:
            for agent_type, agent_data in system_data_model["agents"].items():
                if "variables" in agent_data:
                    for var in agent_data["variables"]:
                        available_variables.add(var["name"])
        
        valid_metrics = []
        for metric in metrics:
            is_valid = True
            invalid_vars = []
            
            # Validate visualization type
            if "visualization_type" not in metric or metric["visualization_type"] not in self.visualization_types:
                logger.warning(f"Metric '{metric.get('name', 'unknown')}' uses invalid visualization type: {metric.get('visualization_type', 'missing')}, set to default 'line'")
                metric["visualization_type"] = "line"
            
            # Ensure variables referenced by the metric exist
            for var in metric.get("variables", []):
                if var.get("name") not in available_variables:
                    is_valid = False
                    invalid_vars.append(var.get("name", "unknown"))
                
                # Validate source_type
                if "source_type" not in var or var["source_type"] not in self.source_types:
                    logger.warning(f"指标 '{metric.get('name', 'unknown')}' 的变量 '{var.get('name', 'unknown')}' 使用了无效的source_type: {var.get('source_type', 'missing')}，已设为默认值'env'")
                    var["source_type"] = "env"
            
            if not is_valid:
                logger.warning(f"指标 '{metric.get('name', 'unknown')}' 使用了不存在的变量: {', '.join(invalid_vars)}，将跳过")
            else:
                valid_metrics.append(metric)
        
        logger.info(f"验证完成: {len(valid_metrics)}/{len(metrics)} 个指标有效")
        return valid_metrics
    
    def _create_generation_prompt(self, scenario_description: str, agent_types: List[str], system_data_model: Dict = None, num_metrics: int = 3) -> str:
        """
        Build the prompt used to generate metrics.
        
        Args:
            scenario_description: Scenario description
            agent_types: List of agent types
            system_data_model: System data model with environment and agent variables
            num_metrics: Number of metrics to generate
            
        Returns:
            Prompt string
        """
        system_data_model_str = ""
        available_variables = []
        
        if system_data_model:
            # Format system data model as a string
            system_data_model_str = json.dumps(system_data_model, indent=2)
            
            # Extract environment variables
            if "environment" in system_data_model and "variables" in system_data_model["environment"]:
                for var in system_data_model["environment"]["variables"]:
                    available_variables.append({
                        "name": var["name"],
                        "type": var["type"],
                        "source_type": "env",
                        "path": var["name"]  # Path can be simple or nested e.g., "stats.value"
                    })
            
            # Extract agent variables
            if "agents" in system_data_model:
                for agent_type, agent_data in system_data_model["agents"].items():
                    if "variables" in agent_data:
                        for var in agent_data["variables"]:
                            available_variables.append({
                                "name": var["name"],
                                "type": var["type"],
                                "source_type": "agent",
                                "agent_type": agent_type,
                                "path": var["name"],  # Path can be simple or nested e.g., "group.name"
                                "is_list": True  # Mark as list type
                            })
        
        # Format available variables list
        available_variables_str = json.dumps(available_variables, indent=4)

        return f"""
Metric Generation Task

Scenario Description:
```
{scenario_description}
```

Agent Types:
```
{", ".join(agent_types)}
```

System Data Model:
```json
{system_data_model_str}
```

Available Variables:
```json
{available_variables_str}
```

Task: Generate monitoring metrics that would be valuable for analyzing this multi-agent system.

Requirements:
1. Consider key performance indicators that would provide insights into:
   - System-level outcomes
   - Agent-specific behaviors
   - Resource utilization
   - Interaction patterns
   - Emergent phenomena

2. For each metric, specify:
   - Descriptive name
   - Clear explanation of what it measures
   - Variables needed from environment or agents
   - Calculation logic
   - Appropriate visualization type

3. Use only available data sources from the system data model:
   - Environment variables (via "env" source_type) - these are single values accessed directly from data dictionary
   - Agent variables (via "agent" source_type) - these are lists of values, one for each agent of that type

4. Support these visualization types:
   - "line": For time-series data plotting changes over time (can have multiple lines)
   - "bar": For comparing values across categories
   - "pie": For showing proportions of a whole

5. IMPORTANT: Remember that agent data comes as lists of values (one per agent of that type), not as single values.
   For agent variables, you will need to include appropriate aggregation methods (sum, average, max, min, etc.).

6. CRITICAL: Be aware that data might have None values, empty lists, or unexpected types. Your calculation logic
   must describe how to handle these edge cases safely (using default values, skipping calculations, etc.)

7. MULTI-SERIES SUPPORT: For appropriate metrics, consider returning multiple series of data:
   - For "line" charts: Return a dictionary where each key is a series name and value is the data point
   - For "bar" charts: Return a dictionary where each key is a category name and value is the bar height
   - For "pie" charts: Return a dictionary where each key is a slice name and value is the proportion

Output Format:
```json
{{
  "metrics": [
    {{
      "name": "metric_name",
      "description": "What this metric measures",
      "visualization_type": "line|bar|pie",
      "update_interval": 5,
      "variables": [
        {{
          "name": "variable_name",
          "source_type": "env|agent",
          "agent_type": "AgentType",  // Only when source_type is "agent"
          "path": "variable_name_or_nested.path",    // Variable name or dot-separated path
          "required": true,
          "is_list": true  // Set to true for agent variables which are lists
        }}
      ],
      "calculation_logic": "Explanation of how this metric is calculated from the variables, including how list data is aggregated, how edge cases are handled, and how multiple series are formed (if applicable)"
    }}
  ]
}}
```

Generate {num_metrics} metrics that would be most valuable for monitoring this scenario, making sure to use ONLY the variables defined in the system data model.
"""
    
    def generate_calculation_function(self, metric_def: Dict, system_data_model: Dict = None) -> str:
        """
        Generate calculation function code for a metric.
        """
        # Validate input
        if not metric_def or "name" not in metric_def:
            logger.error("Invalid metric definition, cannot generate calculation function")
            return "def invalid_metric(data: Dict[str, Any]) -> Any:\n    return 0"
        
        function_name = re.sub(r'[^\w\-_]', '_', metric_def["name"])
        prompt = self._create_function_prompt(metric_def, system_data_model, function_name)
        
        # Get response from the model
        prompt_message = self.model.format(
            Message("system", self.sys_prompt, role="system"),
            Message("user", prompt, role="user")
        )
        response = self.model(prompt_message)
        
        # Extract Python code from the response
        try:
            # Try code_parser first
            code_block = self.code_parser.parse(response)
            function_code = code_block.parsed
            
            # Ensure the code uses the correct function name
            expected_def = f"def {function_name}"
            if expected_def not in function_code:
                # If the name mismatches, try to replace it
                function_code = re.sub(r'def\s+([a-zA-Z0-9_]+)', f'def {function_name}', function_code)
                
                # If still missing, fall back to our template
                if expected_def not in function_code:
                    logger.warning(f"Cannot find correct function definition: {expected_def}, creating a basic function")
                    return self._create_default_function_code(function_name, metric_def)
            
            # Ensure the function has a docstring
            if '"""' not in function_code:
                # Insert docstring after the function definition
                docstring = self._generate_function_docstring(metric_def)
                function_code = re.sub(
                    f'def {function_name}\\([^)]*\\)[^:]*:',
                    f'def {function_name}(data: Dict[str, Any]) -> Any:\n    """{docstring}"""',
                    function_code
                )
            
            # Check whether safe utility functions are used
            safe_functions = ["safe_get", "safe_list", "safe_avg", "safe_sum", "safe_max", "safe_min", "safe_count"]
            has_safe_functions = any(func in function_code for func in safe_functions)
            
            # If not, additional error handling may be needed
            if not has_safe_functions and "try:" not in function_code:
                logger.warning(f"Generated function {function_name} lacks adequate error handling")
                # Could enhance error handling here, but parsing function structure is complex
            
            return function_code
                
        except Exception as e:
            logger.error(f"Error parsing calculation function code: {str(e)}")
            return self._create_default_function_code(function_name, metric_def)
    
    def _generate_function_docstring(self, metric_def: Dict) -> str:
        """
        Generate a docstring for a metric function.
        """
        return f"""
    Metric: {metric_def.get('name', 'unknown')}
    Description: {metric_def.get('description', 'No description')}
    Visualization type: {metric_def.get('visualization_type', 'line')}
    Update interval: {metric_def.get('update_interval', 5)} seconds
    
    Args:
        data: Dict of all variables; agent variables are list-valued
        
    Returns:
        Result format depends on visualization type:
        - line: single numeric value
        - bar/pie: dict mapping categories to values
        
    Notes:
        Handles edge cases such as None, empty lists, and type errors
    """
    
    def _create_default_function_code(self, function_name: str, metric_def: Dict) -> str:
        """
        Create default function code.
        """
        docstring = self._generate_function_docstring(metric_def)
        
        # Required variables
        required_vars = [v["name"] for v in metric_def.get("variables", []) if v.get("required", True)]
        required_vars_check = "\n        ".join([
            f"# Check required variable {var}",
            f"if '{var}' not in data:",
            f"    log_metric_error('{metric_def.get('name', 'unknown')}', ValueError(f'Missing required variable: {var}'))",
            f"    return 0 if '{metric_def.get('visualization_type', 'line')}' == 'line' else {{}}"
        ] for var in required_vars)
        
        # Split variables by source
        env_vars = [v for v in metric_def.get("variables", []) if v.get("source_type") == "env"]
        agent_vars = [v for v in metric_def.get("variables", []) if v.get("source_type") == "agent"]
        
        # Variable access code
        env_vars_code = "\n        ".join([
            f"# Access environment variable {v['name']}",
            f"{v['name']}_value = safe_get(data, '{v['name']}')"
        ] for v in env_vars)
        
        agent_vars_code = "\n        ".join([
            f"# Safely handle agent variable {v['name']} (list form)",
            f"{v['name']}_data = safe_list(safe_get(data, '{v['name']}'))",
            f"# Aggregate list data safely",
            f"{v['name']}_avg = safe_avg({v['name']}_data)",
            f"{v['name']}_sum = safe_sum({v['name']}_data)",
            f"{v['name']}_max = safe_max({v['name']}_data)",
            f"{v['name']}_min = safe_min({v['name']}_data)",
            f"{v['name']}_count = safe_count({v['name']}_data)"
        ] for v in agent_vars)
        
        # Default return value by visualization type
        vis_type = metric_def.get("visualization_type", "line")
        if vis_type == "line":
            result_code = "result = 0  # Default value; adjust per actual logic"
        elif vis_type in ["bar", "pie"]:
            result_code = "result = {'category1': 0, 'category2': 0}  # Example default; adjust per actual logic"
        else:
            result_code = "result = 0  # Default value; adjust per actual logic"
        
        return f"""def {function_name}(data: Dict[str, Any]) -> Any:
    \"\"\"{docstring}\"\"\"
    try:
        # Validate input data
        if not data or not isinstance(data, dict):
            log_metric_error('{metric_def.get('name', 'unknown')}', ValueError('Invalid data input'), {{'data': data}})
            return 0 if '{vis_type}' == 'line' else {{}} 
        
        {required_vars_check}
        
        # Environment variables (scalar)
        {env_vars_code}
        
        # Agent variables (list form)
        {agent_vars_code}
        
        # Metric result
        # TODO: Implement calculation_logic from the metric definition
        {result_code}
        
        return result
    except Exception as e:
        # Log errors instead of printing
        log_metric_error('{metric_def.get('name', 'unknown')}', e, {{'data_keys': list(data.keys()) if isinstance(data, dict) else None}})
        # Default by visualization type
        return 0 if '{vis_type}' == 'line' else {{}}
"""
    
    def _create_function_prompt(self, metric_def: Dict, system_data_model: Dict = None, function_name: str = None) -> str:
        """
        Build the prompt used to generate a calculation function.
        
        Args:
            metric_def: Metric definition dict
            system_data_model: System data model with environment and agent variables
            function_name: Function name; defaults from metric name
            
        Returns:
            Prompt string
        """
        if function_name is None:
            function_name = re.sub(r'[^\w\-_]', '_', metric_def["name"])
            
        variables_str = "\n".join([
            f"- {v.get('name', 'unknown')}: from {'environment' if v.get('source_type') == 'env' else v.get('agent_type', 'unknown')}" +
            (f" (optional)" if not v.get('required', True) else "") +
            (f" (LIST OF VALUES)" if v.get('source_type') == 'agent' or v.get('is_list', False) else "") +
            (f" (Path: {v.get('path', v.get('name', 'unknown'))})" if v.get('path') != v.get('name') else "") # Show path if different from name
            for v in metric_def.get("variables", [])
        ])
        
        system_data_model_str = ""
        if system_data_model:
            system_data_model_str = json.dumps(system_data_model, indent=2)
        
        return f"""
Metric Calculation Function Generation Task

Metric Definition:
- Name: {metric_def.get('name', 'unknown')}
- Description: {metric_def.get('description', 'No description')}
- Visualization Type: {metric_def.get('visualization_type', 'line')}

Available Variables:
{variables_str}

System Data Model:
```json
{system_data_model_str}
```

Calculation Logic:
{metric_def.get('calculation_logic', 'No calculation logic provided')}

Task: Write a Python function named "{function_name}" to calculate this metric.

Requirements:
1. Function name MUST be "{function_name}" (not "calculate")
2. Takes a single parameter: data (dict containing all variables collected by the monitor)
3. Return appropriate data structure for the visualization type:
   - For "line": Return a dict with series names as keys and values as data points
   - For "bar": Return a dict where keys are categories and values are measurements
   - For "pie": Return a dict where keys are categories and values are proportions

4. DATA VALIDATION AND ERROR HANDLING IS CRITICAL:
   - The 'data' dictionary contains the collected values using the 'name' specified in the metric definition's 'variables' list.
   - Check if required variables exist in the data dictionary using their 'name'.
   - Handle None values, empty lists, and invalid data types for the *values* within the 'data' dict.
   - Handle division by zero scenarios.
   - Use the utility functions imported from onesim.monitor.utils module.
   - Log errors with context using log_metric_error function.

5. Available Utility Functions:
   - safe_get(data, key, default=None): Safely gets a value from a dict. IMPORTANT: This function DOES NOT support dot notation directly on the input 'data' dictionary. Use it to get top-level variables by their 'name'.
   - safe_number(value, default=0): Safely converts a value to a number
   - safe_list(value): Ensures a value is a list
   - safe_sum(values, default=0): Safely sums a list of values
   - safe_avg(values, default=0): Safely calculates the average of a list
   - safe_max(values, default=0): Safely finds the maximum value in a list
   - safe_min(values, default=0): Safely finds the minimum value in a list
   - safe_count(values, predicate=None): Safely counts elements in a list
   - log_metric_error(metric_name, error, context=None): Logs metric calculation errors

6. IMPORTANT: Agent data is provided as LISTS of values, one value per agent of that type. The path definition was handled during data collection. The function receives a dictionary where keys are the variable 'name' and values are either single environment values or lists of agent values.
   - You MUST handle agent variables (which are lists) using appropriate aggregation (sum, average, max, min) using the safe_* helper functions.
   - Environment variables are single values.
   - ALWAYS check if list is empty before operations like sum() or calculating averages.
   - Use the safe_* helper functions available in the module.

7. MULTI-SERIES SUPPORT:
   - For "line" charts: Return a dictionary where each key represents a different line
     Example: {{'series1': value1, 'series2': value2}}
   - For "bar" charts with multiple series: Return a nested dictionary or appropriate format
     that represents multiple data series for grouped bar charts
   - For "pie" charts: Return a dictionary where each key represents a slice

8. EXTREMELY IMPORTANT: Make your function robust to all of these edge cases:
   - Variable 'name' might be missing from the data dictionary
   - Values associated with a 'name' might be None
   - Agent variables (lists) might be empty
   - Lists might contain None values
   - Values might be of unexpected types
   - Division by zero scenarios
   - Other potential errors or exceptions

9. VARIABLE ACCESS PATTERN:
   - Access variables from the input 'data' dictionary using the 'name' defined in the metric's 'variables' list.
   - Example: `value = safe_get(data, 'variable_name_from_metric_def')`
   - The monitor system handles data collection based on the variable's 'path':
     * For most variables, 'path' is the same as 'name' (simple key access)
     * For nested data structures, 'path' uses dot notation (e.g., "stats.value")
   - IMPORTANT: Regardless of how complex the original 'path' is, inside this function
     you ALWAYS access data using ONLY the variable 'name' with safe_get(data, 'name')
   - The system has already traversed any complex paths during data collection and placed
     the values in the 'data' dictionary under the corresponding 'name' keys.

Return the function in a Python code block:

```python
from typing import Dict, Any

def {function_name}(data: Dict[str, Any]) -> Any:
    \"\"\"
    Metric: {metric_def.get('name', 'unknown')}
    Description: {metric_def.get('description', 'No description')}
    Visualization type: {metric_def.get('visualization_type', 'line')}
    Update interval: {metric_def.get('update_interval', 5)} seconds
    
    Args:
        data: Dict of all variables; agent variables are list-valued
        
    Returns:
        Result format depends on visualization type:
        - line: dict mapping series names to values
        - bar/pie: dict mapping categories to values
        
    Notes:
        Handles edge cases such as None, empty lists, and type errors
    \"\"\"
    try:
        # Check if required variables exist and validate input data
        if not data or not isinstance(data, dict):
            log_metric_error("{metric_def.get('name', 'unknown')}", ValueError("Invalid data input"), {{"data": data}})
            return {{}} if "{metric_def.get('visualization_type', 'line')}" != "line" else {{"default": 0}}

        # Example: Accessing a required environment variable named 'env_var_name'
        env_value = safe_get(data, 'env_var_name')
        if env_value is None: # Check if required value is missing or None
            log_metric_error("{metric_def.get('name', 'unknown')}", ValueError("Missing required variable: env_var_name"))
            return {{}} if "{metric_def.get('visualization_type', 'line')}" != "line" else {{"default": 0}}

        # Example: Accessing an agent variable list named 'agent_var_name'
        agent_data_list = safe_list(safe_get(data, 'agent_var_name', [])) # Use default [] if not present

        # Safe aggregation using helper functions
        agent_avg = safe_avg(agent_data_list)
        agent_sum = safe_sum(agent_data_list)
        
        # Implementation
        # [Your calculation logic here using safe_get(data, var_name) to access values]
        
        # Return result in appropriate format
        result = {{}} # Placeholder
        return result
    except Exception as e:
        log_metric_error("{metric_def.get('name', 'unknown')}", e, {{"data_keys": list(data.keys()) if isinstance(data, dict) else None}})
        return {{}} if "{metric_def.get('visualization_type', 'line')}" != "line" else {{"default": 0}}
```
"""

    def format_metrics_for_export(self, metrics: List[Dict]) -> List[Dict]:
        """
        Format generated metrics for export (e.g. scene_info.json).
        """
        formatted_metrics = []
        
        for metric in metrics:
            # Safe function name
            function_name = re.sub(r'[^\w\-_]', '_', metric.get("name", "unknown_metric"))
            
            # Normalize metric fields
            formatted_metric = {
                "id": function_name,
                "name": metric.get("name", "Unnamed metric"),
                "description": metric.get("description", "No description"),
                "visualization_type": metric.get("visualization_type", "line"),
                "update_interval": metric.get("update_interval", 60),
                "variables": metric.get("variables", []),
                "calculation_logic": metric.get("calculation_logic", "No calculation logic"),
                "function_name": function_name
            }
            
            formatted_metrics.append(formatted_metric)
            
        return formatted_metrics

    def generate_metrics_code_file(self, metrics: List[Dict], output_dir: str) -> Dict[str, str]:
        """
        Generate a calculation code file for each metric.
        """
        os.makedirs(output_dir, exist_ok=True)
        file_paths = {}
        
        for metric in metrics:
            metric_name = metric.get('name', 'unknown_metric')
            function_code = self.generate_calculation_function(metric)
            
            # Sanitize filename
            safe_name = re.sub(r'[^\w\-_]', '_', metric_name)
            file_path = os.path.join(output_dir, f"metric_{safe_name}.py")
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"""# -*- coding: utf-8 -*-
\"\"\"
Metric: {metric.get('name', 'unknown_metric')}
Description: {metric.get('description', 'No description')}
Visualization type: {metric.get('visualization_type', 'line')}
Update interval: {metric.get('update_interval', 60)} seconds
\"\"\"

from typing import Dict, Any, List, Optional, Union
from loguru import logger
from onesim.monitor.utils import (
    safe_get, safe_number, safe_list, safe_sum, 
    safe_avg, safe_max, safe_min, safe_count, log_metric_error
)

{function_code}
""")
            file_paths[metric_name] = file_path
            logger.info(f"Generated metric calculation code: {file_path}")
            
        return file_paths

    def generate_metrics_module(self, metrics: List[Dict], output_dir: str, system_data_model: Dict = None) -> str:
        """
        Generate a single metrics module file for all metrics.
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create metrics.py
        module_path = os.path.join(output_dir, "metrics.py")
        with open(module_path, 'w', encoding='utf-8') as f:
            # File header and utility imports
            f.write("""# -*- coding: utf-8 -*-
\"\"\"
Auto-generated monitoring metric calculation module
\"\"\"

from typing import Dict, Any, List, Optional, Union, Callable
import math
from loguru import logger
from onesim.monitor.utils import (
    safe_get, safe_number, safe_list, safe_sum, 
    safe_avg, safe_max, safe_min, safe_count, log_metric_error
)

""")
            
            # One calculation function per metric
            for metric in metrics:
                metric_name = metric.get('name', 'unknown_metric')
                # Safe function name
                function_name = re.sub(r'[^\w\-_]', '_', metric_name)
                # Function body
                function_code = self.generate_calculation_function(metric, system_data_model)
                
                # Append function
                f.write(f"""
{function_code}
""")
            
            # Metric function registry
            f.write("""
# Metric function lookup table
METRIC_FUNCTIONS = {
""")
            for metric in metrics:
                function_name = re.sub(r'[^\w\-_]', '_', metric.get("name", "unknown_metric"))
                f.write(f"    '{function_name}': {function_name},\n")
            f.write("}\n\n")
            
            # Helper to resolve functions by name
            f.write('''
def get_metric_function(function_name: str) -> Optional[Callable]:
    """
    Return the metric calculation function for the given name.
    """
    return METRIC_FUNCTIONS.get(function_name)
''')
        
        logger.info(f"Generated metrics module: {module_path}")
        return module_path