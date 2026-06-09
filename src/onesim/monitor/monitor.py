from typing import Dict, List, Any, Callable, Optional
import logging
from collections import defaultdict
import os
import asyncio
import matplotlib.pyplot as plt
import seaborn as sns
import json
import csv
from datetime import datetime, timedelta, timezone

from matplotlib import dates as mdates
import time
import importlib.util
import re

# Import get_config to access global configuration
from onesim.config import get_config

from .metric import (
    MetricDefinition, 
    VariableSpec,
    MetricResult, 
    TimeSeriesMetricData,
    CategoryMetricData
)

from loguru import logger

from .utils import (
    create_line_chart_option,
    create_pie_chart_option,
    create_bar_chart_option,
    create_time_series_chart_option,
    summarize_metric_result_for_log,
)


class DataCollector:
    """数据收集器，负责从环境和Agent收集所需数据"""
    
    async def collect_env_data(self, env: Any, variables: List[VariableSpec]) -> Dict:
        """
        从环境中收集指定变量 (using var.path and env.get_data)
        
        Args:
            env: 环境对象 (assuming env has a get_data method)
            variables: 变量规范列表
            
        Returns:
            变量名到值的映射
        """
        result = {}
        if not hasattr(env, 'get_data') or not callable(env.get_data):
            logger.error("Environment object is missing a callable 'get_data' method.")
            # Provide default values (e.g., None) for all requested env vars
            for var in variables:
                if var.source_type == "env":
                    result[var.name] = None
            return result
            
        for var in variables:
            if var.source_type != "env":
                continue
            
            # Use env.get_data which supports dot notation for nested access
            value = await env.get_data(var.path) # Using var.path directly
            result[var.name] = value # Store result using the metric's variable name
            
        return result
        
    async def collect_agent_data(self, env: Any, agent_type: str, variables: List[VariableSpec]) -> Dict:
        """
        从特定类型的所有Agent收集数据 (using env.get_agent_data_by_type)
        
        Args:
            env: 环境对象 (must have get_agent_data_by_type method)
            agent_type: Agent类型
            variables: 变量规范列表
            
        Returns:
            变量名到值的映射，对于Agent变量，值是列表
        """
        result = defaultdict(list)
        
        # Filter variables relevant to this agent type
        agent_vars_for_type = [var for var in variables if var.source_type == "agent" and var.agent_type == agent_type]
        if not agent_vars_for_type:
            return {} # No relevant variables for this type

        # Check if environment has the required method
        if not hasattr(env, 'get_agent_data_by_type') or not callable(env.get_agent_data_by_type):
            logger.error(f"Environment is missing callable 'get_agent_data_by_type' method.")
            # Return empty lists for all variables of this type
            for var in agent_vars_for_type:
                result[var.name] = []
            return result

        # Iterate through each relevant variable definition
        for var in agent_vars_for_type:
            try:
                # Use the environment's method to get data for this variable from all agents of the type
                # This method should handle dot notation in var.path and potential distribution
                agent_data_dict = await env.get_agent_data_by_type(agent_type, var.path)
                
                # The expected return format for collect_agent_data is a list of values.
                # Extract values from the dictionary returned by get_agent_data_by_type.
                # The order might not be guaranteed, but for aggregation it often doesn't matter.
                if isinstance(agent_data_dict, dict):
                    collected_values = list(agent_data_dict.values())
                elif agent_data_dict is None:
                    # If the method returns None (e.g., error), provide an empty list
                    collected_values = []
                    logger.warning(f"Received None from get_agent_data_by_type for {agent_type}.{var.path}")
                else:
                    # Handle unexpected return types
                    logger.warning(f"Unexpected return type {type(agent_data_dict)} from get_agent_data_by_type for {agent_type}.{var.path}")
                    collected_values = []
                    
            except Exception as e:
                logger.error(f"Error calling env.get_agent_data_by_type for '{agent_type}', path '{var.path}': {e}")
                collected_values = [] # Provide empty list on error
                
            # Store the list of collected values under the metric variable name
            result[var.name] = collected_values
            
        return result
    
    async def collect_for_metric(self, env: Any, metric_def: MetricDefinition) -> Dict:
        """
        收集特定指标所需的所有数据
        
        Args:
            env: 环境对象
            metric_def: 指标定义
            
        Returns:
            变量名到值的映射
        """
        result = {}
        
        # 按变量来源分组
        env_vars = []
        agent_vars_by_type = defaultdict(list)
        
        for var in metric_def.variables:
            if var.source_type == "env":
                env_vars.append(var)
            elif var.source_type == "agent" and var.agent_type:
                agent_vars_by_type[var.agent_type].append(var)
        
        # 收集环境变量
        if env_vars:
            env_data = await self.collect_env_data(env, env_vars)
            result.update(env_data)
        
        # 收集每种类型的Agent变量
        for agent_type, vars_list in agent_vars_by_type.items():
            agent_data = await self.collect_agent_data(env, agent_type, vars_list)
            result.update(agent_data)
        
        return result


class MetricProcessor:
    """指标处理器，执行计算并格式化结果"""
    
    def calculate(self, metric_def: MetricDefinition, data: Dict) -> Any:
        """
        执行指标计算
        
        Args:
            metric_def: 指标定义
            data: 所需的数据
            
        Returns:
            计算结果
        """
        try:
            # 检查是否所有必需的变量都有值
            missing_vars = []
            for var in metric_def.variables:
                # Also check for None if required, as None might break calculations
                if var.required and (var.name not in data or data[var.name] is None):
                    missing_vars.append(var.name)
                    
            if missing_vars:
                logger.warning(f"指标 {metric_def.name} 缺少必要变量或其值为None: {', '.join(missing_vars)}. Returning None.")
                return None # Calculation function should handle None input if possible
                
            # 执行计算函数
            result = metric_def.calculation_function(data)
            return result
        except Exception as e:
            logger.error(f"计算指标 {metric_def.name} 时发生错误: {str(e)}", exc_info=True)
            return None # Return None on calculation error
            
    def format_for_visualization(self, raw_result: Any, metric_def: MetricDefinition, ts_data: Optional[TimeSeriesMetricData] = None) -> Dict:
        """
        将原始结果转换为可视化格式
        
        Args:
            raw_result: 原始计算结果
            metric_def: 指标定义
            ts_data: 时间序列数据 (用于折线图)
            
        Returns:
            适用于ECharts的数据结构
        """
        if raw_result is None:
             logger.debug(f"Raw result for {metric_def.name} is None, returning empty viz data.")
             return {} # Return empty dict if calculation failed or returned None
            
        viz_type = metric_def.visualization_type
        
        if viz_type == "line":
            # 折线图使用时间序列数据 (ts_data handles formatting)
            if ts_data is None:
                logger.warning(f"TimeSeries data not provided for line chart: {metric_def.name}")
                return {"xAxis": [], "series": []} # Default empty structure
            try:
                return ts_data.get_echarts_data() # Delegate formatting
            except Exception as e:
                logger.error(f"Error getting ECharts data from TimeSeriesMetricData for {metric_def.name}: {e}")
                return {"xAxis": [], "series": []}
            
        elif viz_type == "bar":
            # 柱状图处理 - Expects dict {category: value} or list [(cat, val), ...]
            try:
                if isinstance(raw_result, dict):
                    categories = list(raw_result.keys())
                    values = list(raw_result.values())
                elif isinstance(raw_result, (list, tuple)) and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in raw_result):
                    # 处理 [(key, value), ...] 结构
                    categories = [item[0] for item in raw_result]
                    values = [item[1] for item in raw_result]
                elif isinstance(raw_result, (int, float)): # Handle single value case for bar? Maybe should be pie?
                     logger.warning(f"Single value {raw_result} received for bar chart {metric_def.name}. Formatting as single bar.")
                     categories = [metric_def.name] # Use metric name as category
                     values = [raw_result]
                else:
                    logger.error(f"无法将结果格式化为柱状图 (不支持的类型 {type(raw_result)}): {metric_def.name}")
                    return {"xAxis": [], "series": []}
                    
                return {"xAxis": categories, "series": values}
            except Exception as e:
                 logger.error(f"Error formatting bar chart data for {metric_def.name}: {e}")
                 return {"xAxis": [], "series": []}
            
        elif viz_type == "pie":
            # 饼图处理 - Expects dict {category: value} or list [(cat, val), ...]
            # 饼图：若结果为 {"total_comments", "counts", "ratios"}，用 counts 画扇区
            try:
                series_data = []
                if isinstance(raw_result, dict):
                    pie_dict = raw_result
                    if (
                        "counts" in raw_result
                        and isinstance(raw_result.get("counts"), dict)
                        and raw_result.get("counts")
                    ):
                        pie_dict = raw_result["counts"]
                    series_data = [
                        {"name": k, "value": v}
                        for k, v in pie_dict.items()
                        if isinstance(v, (int, float)) and v >= 0
                    ]
                elif isinstance(raw_result, (list, tuple)) and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in raw_result):
                    # 处理 [(key, value), ...] 结构
                    series_data = [{"name": item[0], "value": item[1]} for item in raw_result if isinstance(item[1], (int, float)) and item[1] >= 0]
                else:
                    logger.error(f"无法将结果格式化为饼图 (不支持的类型 {type(raw_result)}): {metric_def.name}")
                    return {"series": []}
                
                # Filter out zero/negative values as they don't make sense in pie charts
                series_data = [item for item in series_data if item['value'] > 0]
                
                return {"series": series_data}
            except Exception as e:
                 logger.error(f"Error formatting pie chart data for {metric_def.name}: {e}")
                 return {"series": []}
            
        # Fallback for unknown viz_type or if logic missed a case
        logger.warning(f"Unknown visualization type '{viz_type}' or unhandled format for metric {metric_def.name}. Returning raw result.")
        return {"raw": raw_result} # Or return empty dict {}?
 

class MonitorScheduler:
    """指标更新调度器"""
    
    def __init__(self):
        self.tasks = {}  # 存储每个指标的调度任务 {metric_name: task}
        self.lock = asyncio.Lock()  # 使用asyncio.Lock而不是threading.Lock
        
    async def schedule_metric(self, metric_name: str, env: Any, monitor_manager: 'MonitorManager', frequency: int):
        """异步调度指标定期更新"""
        # 如果该指标已在调度中，先停止
        if metric_name in self.tasks:
            await self.pause_metric(metric_name)
        
        # 创建异步任务
        task = asyncio.create_task(
            self._update_loop(metric_name, env, monitor_manager, frequency)
        )
        
        # 保存任务
        async with self.lock:
            self.tasks[metric_name] = task
        
        logger.debug(f"指标 {metric_name} 调度已启动，更新频率: {frequency}秒")
    
    async def _update_loop(self, metric_name: str, env: Any, monitor_manager: 'MonitorManager', frequency: int):
        """异步指标更新循环"""
        try:
            while True:
                # 执行更新
                await monitor_manager.update_metric(metric_name, env)
                
                # 异步等待
                await asyncio.sleep(frequency)
        except asyncio.CancelledError:
            logger.debug(f"指标 {metric_name} 更新任务已取消")
        except Exception as e:
            logger.error(f"指标 {metric_name} 更新循环出错: {e}")
    
    async def pause_metric(self, metric_name: str):
        """
        暂停指标更新
        
        Args:
            metric_name: 指标名称
        """
        async with self.lock:
            if metric_name in self.tasks:
                task = self.tasks[metric_name]
                task.cancel()
                # Wait for the task to actually finish cancellation
                try:
                    await task 
                except asyncio.CancelledError:
                    pass # Expected
                self.tasks.pop(metric_name, None)
                logger.debug(f"指标 {metric_name} 调度已暂停")
    
    async def update_interval(self, metric_name: str, env: Any, monitor_manager: 'MonitorManager', new_frequency: int):
        """
        更新指标的更新频率
        
        Args:
            metric_name: 指标名称
            env: 环境对象
            monitor_manager: 监控管理器
            frequency: 更新频率(秒)，如果不提供则使用指标定义中的频率或全局配置
        """
        # Fetch the global update interval from MonitorConfig
        global_update_interval = get_config().monitor_config.update_interval
        
        metric_def = monitor_manager.metrics.get(metric_name)
        if not metric_def:
             logger.error(f"Cannot resume metric {metric_name}: definition not found.")
             return

        # Determine the frequency to use
        effective_frequency = global_update_interval if global_update_interval is not None else metric_def.update_interval
        
        # Use the provided frequency if it was explicitly passed (overrides default logic)
        if new_frequency is not None:
            effective_frequency = new_frequency

        await self.schedule_metric(metric_name, env, monitor_manager, effective_frequency)
        logger.debug(f"指标 {metric_name} 调度已恢复，更新频率: {effective_frequency}秒")
    
    async def resume_metric(self, metric_name: str, env: Any, monitor_manager: 'MonitorManager', frequency: int = None):
        """
        恢复指标更新
        
        Args:
            metric_name: 指标名称
            env: 环境对象
            monitor_manager: 监控管理器
            frequency: 更新频率(秒)，如果不提供则使用指标定义中的频率
        """
        if frequency is None:
            frequency = monitor_manager.metrics[metric_name].update_interval
        await self.schedule_metric(metric_name, env, monitor_manager, frequency)
        logger.debug(f"指标 {metric_name} 调度已恢复")
    
    async def stop_all(self):
        """停止所有指标更新"""
        for metric_name in list(self.tasks.keys()):
            await self.pause_metric(metric_name)


class MonitorManager:
    """监控系统总控制器"""
    
    def __init__(self):
        # 存储所有注册的指标定义
        self.metrics: Dict[str, MetricDefinition] = {}
        
        # 存储指标计算结果
        self.results: Dict[str, MetricResult] = {}
        
        # 存储时间序列数据(用于折线图)
        self.time_series_data: Dict[str, TimeSeriesMetricData] = {}
        
        # 存储类别数据(用于柱状图和饼图)
        self.category_data: Dict[str, CategoryMetricData] = {}
        
        # 数据收集器和处理器
        self.collector = DataCollector()
        self.processor = MetricProcessor()
        
        # 调度器
        self.scheduler = MonitorScheduler()
        
        # 线程安全锁
        self.lock = asyncio.Lock()  # 使用asyncio.Lock而不是threading.Lock
        
        # 环境对象引用
        self.env = None
        
        # 监控状态
        self.is_monitoring = False
        self.metric_index = 0
        
    def setup(self, env: Any):
        """
        设置监控系统，关联环境对象
        
        Args:
            env: 环境对象
        """
        self.env = env
        logger.info(f"监控系统已关联环境对象")
        return self
    
    @staticmethod
    async def setup_metrics(env: Any):
        """
        在环境中设置和启动监控系统
        
        Args:
            env: 环境对象

        Returns:
            MonitorManager实例
        """
        from onesim.config import get_component_registry
        
        # 尝试从注册表获取监控管理器
        registry = get_component_registry()
        monitor_manager = registry.get_instance("monitor")
        
        # 如果监控管理器不存在，创建一个新的
        if not monitor_manager:
            logger.warning("监控组件未初始化，正在创建新的监控管理器")
            monitor_manager = MonitorManager()
            registry.register("monitor", monitor_manager)
        
        # 设置环境
        monitor_manager.setup(env)

      
            
        env_path = env.env_path
        try:
            # 导入指标计算模块
            metrics_path = os.path.join(env_path, "code", "metrics")
            metrics_module = None
            if os.path.exists(metrics_path):
                if os.path.isdir(metrics_path):
                    module_path = os.path.join(metrics_path, "metrics.py")
                else:
                    module_path = metrics_path
                
                if os.path.exists(module_path):
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("metrics_module", module_path)
                    metrics_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(metrics_module)
                    logger.info(f"成功导入指标计算模块: {module_path}")
            
            # 加载scene_info.json中的指标定义
            scene_info_path = os.path.join(env_path, "scene_info.json")
            if os.path.exists(scene_info_path):
                import json
                import re
                from .metric import VariableSpec, MetricDefinition
                
                with open(scene_info_path, 'r', encoding='utf-8') as f:
                    scene_info = json.load(f)
                
                if "metrics" in scene_info and isinstance(scene_info["metrics"], list):
                    metrics_loaded = 0
                    for metric_def in scene_info["metrics"]:
                        try:
                            # 创建变量规格
                            variables = []
                            for var in metric_def.get("variables", []):
                                variables.append(VariableSpec(
                                    name=var["name"],
                                    source_type=var["source_type"],
                                    path=var["path"],
                                    agent_type=var.get("agent_type"),
                                    required=var.get("required", True)
                                ))
                            
                            # 获取函数名
                            function_name = metric_def.get("function_name") or metric_def.get("id")
                            if not function_name:
                                function_name = re.sub(r'[^\w\-_]', '_', metric_def["name"])
                            
                            # 创建指标定义
                            metric_definition = MetricDefinition(
                                name=metric_def["name"],
                                description=metric_def["description"],
                                variables=variables,
                                visualization_type=metric_def["visualization_type"],
                                update_interval=metric_def.get("update_interval", 60),
                                calculation_function=function_name
                            )
                            
                            # 尝试从metrics模块查找计算函数
                            calculation_function = None
                            if metrics_module:
                                # 首先尝试通过get_metric_function获取
                                if hasattr(metrics_module, 'get_metric_function'):
                                    calculation_function = metrics_module.get_metric_function(function_name)
                                
                                # 如果没有找到，直接尝试获取同名函数
                                if calculation_function is None and hasattr(metrics_module, function_name):
                                    calculation_function = getattr(metrics_module, function_name)
                            
                            if calculation_function:
                                # 注册指标
                                monitor_manager.register_metric(
                                    metric_definition, 
                                    calculation_function=calculation_function
                                )
                                metrics_loaded += 1
                            else:
                                logger.warning(f"未找到指标 '{metric_def['name']}' 的计算函数 '{function_name}'")
                        except Exception as e:
                            logger.error(f"加载指标 '{metric_def.get('name', 'unknown')}' 失败: {e}")
                    
                    logger.info(f"从scene_info.json中加载了 {metrics_loaded} 个指标")
            
        except Exception as e:
            logger.error(f"加载指标失败: {e}")

        # 等环境 load_initial_data 完成后再启动指标轮询，避免首轮 content_pool 等为 None
        wait_fn = getattr(env, "wait_until_initialized", None)
        if callable(wait_fn):
            await wait_fn()
            logger.info("环境异步初始化已完成，启动指标监控")
        
        # 启动监控
        await monitor_manager.start_all_metrics()
        
        return monitor_manager
        
    def register_metric(self, metric_def: MetricDefinition, calculation_function: Callable = None):
        """
        注册新指标
        
        Args:
            metric_def: 指标定义
            calculation_function: 计算函数，如果提供则覆盖metric_def中的函数
        """
        # 检查指标名是否已存在
        if metric_def.name in self.metrics:
            logger.warning(f"指标 {metric_def.name} 已存在，将被覆盖")
        
        # 如果提供了计算函数，覆盖指标定义中的函数
        if calculation_function:
            metric_def.calculation_function = calculation_function
            
        # 存储指标定义
        self.metrics[metric_def.name] = metric_def
        
        # 初始化数据存储
        if metric_def.visualization_type == "line":
            self.time_series_data[metric_def.name] = TimeSeriesMetricData()
        else:  # "bar" or "pie"
            self.category_data[metric_def.name] = CategoryMetricData()
            
        logger.info(f"指标 {metric_def.name} 已注册")
    
    async def set_update_interval(self, metric_name: str, update_interval: int):
        """
        设置指标的更新频率
        
        Args:
            metric_name: 指标名称
            update_interval: 更新频率(秒)
        """
        async with self.lock:
            if metric_name not in self.metrics:
                logger.error(f"无法设置更新频率：指标 {metric_name} 未定义")
                return False
            
            # 更新指标定义中的更新频率
            self.metrics[metric_name].update_interval = update_interval
            
            # 如果指标正在监控中，更新调度频率
            if self.is_monitoring and hasattr(self, 'env') and self.env:
                await self.scheduler.update_interval(metric_name, self.env, self, update_interval)
                logger.info(f"指标 {metric_name} 更新频率已设置为 {update_interval}秒")
            return True
        
    async def unregister_metric(self, metric_name: str):
        """
        注销指标
        
        Args:
            metric_name: 指标名称
        """
        async with self.lock:
            # 先停止调度
            await self.scheduler.pause_metric(metric_name)
            
            # 移除指标定义和相关数据
            self.metrics.pop(metric_name, None)
            self.results.pop(metric_name, None)
            self.time_series_data.pop(metric_name, None)
            self.category_data.pop(metric_name, None)
            
            logger.info(f"指标 {metric_name} 已注销")
            
    def get_metric_definition(self, metric_name: str) -> Optional[MetricDefinition]:
        """
        获取指标定义
        
        Args:
            metric_name: 指标名称
            
        Returns:
            指标定义，如果不存在则返回None
        """
        return self.metrics.get(metric_name)
            
    async def start_all_metrics(self, env: Any = None):
        """
        启动所有指标的监控
        
        Args:
            env: 环境对象，如果不提供则使用已设置的环境
        """
        if env:
            self.env = env
            
        if not self.env:
            logger.error("无法启动监控：未设置环境对象")
            return
        
        # Fetch the global update interval from MonitorConfig
        global_update_interval = get_config().monitor_config.update_interval
        if global_update_interval is not None:
            logger.info(f"使用全局监控更新间隔: {global_update_interval}秒")
        
        async with self.lock:
            for metric_name, metric_def in self.metrics.items():
                # Determine the frequency for this specific metric
                frequency = global_update_interval if global_update_interval is not None else metric_def.update_interval
                
                await self.scheduler.schedule_metric(
                    metric_name, 
                    self.env, 
                    self, 
                    frequency # Use the determined frequency
                )
            self.is_monitoring = True
            logger.info(f"已启动 {len(self.metrics)} 个指标的监控")
                
    async def stop_all_metrics(self):
        """停止所有指标的监控"""
        await self.scheduler.stop_all()
        self.is_monitoring = False
        logger.info("已停止所有指标的监控")
            
    async def update_metric(self, metric_name: str, env: Any = None):
        """
        异步更新指定指标的值
        
        Args:
            metric_name: 指标名称
            env: 环境对象，如果不提供则使用已设置的环境
        """
        if not env and not self.env:
            logger.error(f"无法更新指标 {metric_name}：未提供环境对象")
            return
            
        env = env or self.env
        logger.info(f"更新指标 {metric_name}")
        async with self.lock:
            # 获取指标定义
            metric_def = self.metrics.get(metric_name)
            if not metric_def:
                logger.error(f"无法更新指标 {metric_name}：指标未定义")
                return
                
            # 收集数据
            data = await self.collector.collect_for_metric(env, metric_def)
            # 计算指标值
            raw_result = self.processor.calculate(metric_def, data)
            logger.info(summarize_metric_result_for_log(raw_result))
            if raw_result is None:
                return

            # 发帖用户行为 / 根帖作者自转发：每次指标更新追加一行 JSONL（含 env.current_step）
            if metric_name in (
                "Posting User Diffusion Behavior",
                "Posting User Comment Behavior",
            ) and isinstance(raw_result, dict):
                self._append_posting_user_behavior_snapshot(env, raw_result)
            if metric_name == "Root Author Self-Repost Behavior" and isinstance(
                raw_result, dict
            ):
                self._append_root_author_self_repost_snapshot(env, raw_result)
            
            # 根据可视化类型处理数据
            if metric_def.visualization_type == "line":
                # 规范化折线图数据
                normalized_result = self._normalize_line_data(raw_result)
                # 时间序列数据(折线图)
                ts_data = self.time_series_data[metric_name]
                ts_data.add_point(normalized_result)
                viz_data = ts_data.get_echarts_data()
            else:
                # 类别数据(柱状图或饼图)
                viz_data = self.processor.format_for_visualization(raw_result, metric_def)
                
                # 更新类别数据存储
                cat_data = self.category_data[metric_name]
                if metric_def.visualization_type == "bar":
                    cat_data.update_data(viz_data["xAxis"], viz_data["series"])
                elif metric_def.visualization_type == "pie" and "series" in viz_data:
                    categories = [item["name"] for item in viz_data["series"]]
                    values = [item["value"] for item in viz_data["series"]]
                    cat_data.update_data(categories, values)
            
            # 保存结果
            result = MetricResult(
                metric_name=metric_name,
                raw_data=raw_result,
                visualization_data=viz_data
            )
            self.results[metric_name] = result
            
            logger.debug(f"指标 {metric_name} 已更新")

    async def refresh_all_metrics(self, env: Any = None) -> None:
        """在导出前依次更新所有已注册指标，使 results 与当前 env.data 一致（如刚写入 user_recommended_note_ids_by_channel）。"""
        env = env or self.env
        if not env:
            logger.warning("refresh_all_metrics: 未设置环境对象")
            return
        if not self.metrics:
            return
        for metric_name in list(self.metrics.keys()):
            try:
                await self.update_metric(metric_name, env)
            except Exception as e:
                logger.error(f"refresh_all_metrics: 更新 {metric_name} 失败: {e}")

    def _append_posting_user_behavior_snapshot(self, env: Any, raw_result: Dict[str, Any]) -> None:
        """Append posting-user diffusion snapshots to metrics_save_dir/posting_user_diffusion_behavior/snapshots.jsonl."""
        users = raw_result.get("users")
        if not isinstance(users, list):
            return
        mdir = getattr(env, "metrics_save_dir", None) if env is not None else None
        if not mdir:
            logger.debug("metrics_save_dir unset; skip posting_user_diffusion_behavior JSONL write")
            return
        sub = os.path.join(mdir, "posting_user_diffusion_behavior")
        try:
            os.makedirs(sub, exist_ok=True)
            step = getattr(env, "current_step", None)
            record = {
                "simulation_step": step,
                "recorded_at": time.time(),
                "users": users,
            }
            path = os.path.join(sub, "snapshots.jsonl")
            with open(path, "a", encoding="utf-8") as af:
                af.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write posting_user_diffusion_behavior snapshot: {e}")

    def _append_root_author_self_repost_snapshot(self, env: Any, raw_result: Dict[str, Any]) -> None:
        """根作者在自己链路上的各跳自转发，追加到 root_author_self_repost_behavior/snapshots.jsonl。"""
        users = raw_result.get("users")
        if not isinstance(users, list):
            return
        mdir = getattr(env, "metrics_save_dir", None) if env is not None else None
        if not mdir:
            logger.debug("metrics_save_dir 未设置，跳过 root_author_self_repost 快照")
            return
        sub = os.path.join(mdir, "root_author_self_repost_behavior")
        try:
            os.makedirs(sub, exist_ok=True)
            step = getattr(env, "current_step", None)
            record = {
                "simulation_step": step,
                "recorded_at": time.time(),
                "users": users,
            }
            path = os.path.join(sub, "snapshots.jsonl")
            with open(path, "a", encoding="utf-8") as af:
                af.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"写入 root_author_self_repost 快照失败: {e}")

    @staticmethod
    def _write_root_author_self_repost_csv(path: str, raw: Dict[str, Any]) -> None:
        """根作者自转发行为表：user_id, nickname, root_post_count, repost_on_others_count, self_repost_hop_1, ..."""
        users = raw.get("users")
        if not isinstance(users, list):
            users = []
        hop_keys: set = set()
        for row in users:
            if isinstance(row, dict):
                for k in row:
                    if k.startswith("self_repost_hop_"):
                        hop_keys.add(k)
        hop_sorted = sorted(
            hop_keys,
            key=lambda x: int(x.replace("self_repost_hop_", "")),
        )
        fieldnames = [
            "user_id",
            "nickname",
            "root_post_count",
            "repost_on_others_count",
            "self_propagation_one_hop",
            "self_propagation_multi_hop",
        ] + hop_sorted
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in users:
                if not isinstance(row, dict):
                    continue
                writer.writerow({k: row.get(k, "") for k in fieldnames})

    @staticmethod
    def _write_posting_user_behavior_csv(path: str, raw: Dict[str, Any]) -> None:
        """Write calculate_posting_user_diffusion_behavior users table to CSV (UTF-8 BOM for Excel)."""
        users = raw.get("users")
        if not isinstance(users, list):
            users = []
        fieldnames = [
            "user_id",
            "nickname",
            "发帖数",
            "本帖一跳评论数",
            "本帖回复评论数",
            "他帖评论数",
            "发表评论总数",
        ]
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in users:
                if not isinstance(row, dict):
                    continue
                writer.writerow({k: row.get(k, "") for k in fieldnames})

    def get_result(self, metric_name: str) -> Optional[MetricResult]:
        """
        获取指标结果
        
        Args:
            metric_name: 指标名称
            
        Returns:
            指标结果，如果不存在则返回None
        """
        return self.results.get(metric_name)
        
    def get_all_results(self) -> Dict[str, MetricResult]:
        """
        获取所有指标结果
        
        Returns:
            指标名称到结果的映射
        """
        return self.results.copy()
        
    def get_results_by_type(self, visualization_type: str) -> Dict[str, MetricResult]:
        """
        获取特定可视化类型的所有指标结果
        
        Args:
            visualization_type: 可视化类型 ("bar", "pie", "line")
            
        Returns:
            指标名称到结果的映射
        """
        results = {}
        for name, metric_def in self.metrics.items():
            if metric_def.visualization_type == visualization_type:
                result = self.results.get(name)
                if result:
                    results[name] = result
        return results
        
    def get_time_series_data(self, metric_name: str, last_n: Optional[int] = None) -> Dict:
        """
        获取指标的时间序列数据
        
        Args:
            metric_name: 指标名称
            last_n: 如果指定，获取最近n个数据点
            
        Returns:
            格式化的时间序列数据
        """
        ts_data = self.time_series_data.get(metric_name)
        if not ts_data:
            return {"xAxis": [], "series": []}
            
        if last_n:
            return ts_data.get_last_n_points(last_n)
        return ts_data.get_echarts_data()
    
    def get_metric_data(self, metric_name: str, format: str = "echarts") -> Dict:
        """
        获取指标数据，支持多种输出格式
        
        Args:
            metric_name: 指标名称
            format: 输出格式，可选 "echarts" 或 "matplotlib"
        
        Returns:
            适合指定格式的数据结构
        """
        metric_def = self.metrics.get(metric_name)
        if not metric_def:
            return {}
            
        viz_type = metric_def.visualization_type
        
        if viz_type == "line":
            ts_data = self.time_series_data.get(metric_name)
            if not ts_data:
                return {"xAxis": [], "series": []}
                
            if format == "matplotlib":
                return ts_data.get_matplotlib_data()
            else:
                return ts_data.get_echarts_data()
        else:  # "bar" or "pie"
            cat_data = self.category_data.get(metric_name)
            if not cat_data:
                return {}
                
            return cat_data.get_data(format=format, viz_type=viz_type)
    
    def get_metrics_for_api(self) -> Dict[str, Any]:
        """返回适合API传输的所有指标数据, data字段为完整的ECharts Option"""
        metrics_data = {}
        for metric_name, metric_def in self.metrics.items():
            result = self.get_result(metric_name)
            if result and result.visualization_data is not None:
                viz_type = metric_def.visualization_type
                raw_viz_data = result.visualization_data # This holds the basic data, e.g. {"xAxis": [...], "series": [...]} or {"series": [...]} for pie
                echarts_option = {}

                try:
                    if viz_type == "line":
                        # Line charts use TimeSeriesMetricData which formats series data internally for type: time.
                        # raw_viz_data is expected to be {"series": [{name:..., type:'line', data:[[ts, val],...]}, ...]}}
                        series_list = raw_viz_data.get("series", []) # This contains the correctly formatted data
                        
                        # Use the dedicated time-series helper
                        echarts_option = create_time_series_chart_option(
                            title=metric_def.description or metric_name,
                            series_list=series_list
                        )
                        
                        # Removed old manual construction:
                        # x_axis = raw_viz_data.get("xAxis", []) 
                        # legend_data = [s.get('name', f'Series {i+1}') for i, s in enumerate(series_list)]
                        # echarts_option = {
                        #     "title": {"text": metric_def.description or metric_name, "left": "center"},
                        #     "tooltip": {"trigger": "axis"},
                        #     "legend": {"data": legend_data, "bottom": 10, "type": "scroll"}, 
                        #     "grid": {"left": '3%', "right": '4%', "bottom": '10%', "containLabel": True},
                        #     "xAxis": {"type": "category", "boundaryGap": False, "data": x_axis},
                        #     "yAxis": {"type": "value"},
                        #     "series": series_list
                        # }
                        
                    elif viz_type == "pie":
                        # Assuming raw_viz_data format: {"series": List[Dict[str, Any]]} with format {"name": ..., "value": ...}
                        series_data = raw_viz_data.get("series", [])
                        # Ensure it's in the correct format [{"name": ..., "value": ...}, ...]
                        if series_data and not isinstance(series_data[0], dict):
                             logger.warning(f"Pie chart data for {metric_name} has unexpected format. Attempting conversion.")
                             # Attempt conversion if it's just a list of values or simple list of lists/tuples
                             if isinstance(series_data[0], (int, float)): 
                                series_data = [{'name': f'Category {i}', 'value': v} for i, v in enumerate(series_data)]
                             else: # Give up if format is too weird
                                series_data = [] 
                                
                        echarts_option = create_pie_chart_option(
                            title=metric_def.description or metric_name,
                            series_data=series_data,
                            series_name=metric_name # Use metric name as series name for pie
                        )
                    elif viz_type == "bar":
                        # Assuming raw_viz_data format from CategoryMetricData.get_bar_chart_data: {"xAxis": List[str], "series": List[Any]}
                        x_axis = raw_viz_data.get("xAxis", [])
                        series_values = raw_viz_data.get("series", []) # Should be a list of values for a single series
                        
                        # The helper function create_bar_chart_option handles converting this to the ECharts series list format
                        echarts_option = create_bar_chart_option(
                             title=metric_def.description or metric_name,
                             x_axis_data=x_axis,
                             series_data=series_values, # Pass the list of values
                             series_name=metric_name # Default series name if only one
                        )
                        # Old fallback:
                        # logger.warning(f"Bar chart option generation not fully implemented for {metric_name}. Returning raw data structure.")
                        # echarts_option = self._format_for_api_display(raw_viz_data, viz_type)
                    else:
                        # For unknown types, return raw visualization data
                        echarts_option = raw_viz_data

                except Exception as e:
                    logger.error(f"Error generating ECharts option for {metric_name} ({viz_type}): {e}")
                    echarts_option = {"error": f"Failed to generate chart option: {e}"} 

                metrics_data[metric_name] = {
                    "name": metric_name,
                    "description": metric_def.description,
                    "visualization_type": viz_type,
                    "data": echarts_option, # Assign the generated ECharts option
                    "raw_data": result.raw_data, 
                    "timestamp": result.timestamp,
                    "formatted_time": result.formatted_time
                }
            elif metric_name in self.metrics: # Metric exists but no result yet
                 # Provide a default structure based on viz_type so frontend doesn't break
                 metric_def = self.metrics[metric_name]
                 viz_type = metric_def.visualization_type
                 default_option = {}
                 if viz_type == "line":
                     # Use the time series helper for the default option as well
                     default_option = create_time_series_chart_option(title=metric_def.description or metric_name, series_list=[])
                 elif viz_type == "pie":
                     default_option = create_pie_chart_option(title=metric_def.description or metric_name, series_data=[])
                 elif viz_type == "bar":
                     default_option = create_bar_chart_option(title=metric_def.description or metric_name, x_axis_data=[], series_data=[])
                 else:
                     default_option = {"message": "No data available yet."}
                     
                 metrics_data[metric_name] = {
                     "name": metric_name,
                     "description": metric_def.description,
                     "visualization_type": viz_type,
                     "data": default_option,
                     "raw_data": None,
                     "timestamp": int(time.time()),
                     "formatted_time": time.strftime('%Y-%m-%d %H:%M:%S')
                 }
                 
        return metrics_data
        
    def _format_for_api_display(self, viz_data: Dict, viz_type: str) -> Dict:
        """
        将可视化数据格式化为适合API传输的格式
        
        Args:
            viz_data: 可视化数据
            viz_type: 可视化类型
            
        Returns:
            格式化后的数据
        """
        if not viz_data:
            if viz_type == "line":
                return {"xAxis": [], "series": []}
            elif viz_type == "bar":
                return {"xAxis": [], "series": []}
            elif viz_type == "pie":
                return {"series": []}
            return {}
            
        # 对于折线图，确保series是数组格式
        if viz_type == "line" and "series" in viz_data:
            # 已经是数组格式，直接返回
            if isinstance(viz_data["series"], list):
                return viz_data
                
            # 将字典格式转换为数组格式
            if isinstance(viz_data["series"], dict):
                series_array = []
                for name, values in viz_data["series"].items():
                    series_array.append({
                        "name": name,
                        "type": "line",
                        "data": values
                    })
                return {
                    "xAxis": viz_data.get("xAxis", []),
                    "series": series_array
                }
        
        # 对于条形图，标准化格式
        elif viz_type == "bar" and "xAxis" in viz_data and "series" in viz_data:
            # 如果series是字典格式，转换为适合前端的格式
            if isinstance(viz_data["series"], dict):
                series_array = []
                for name, values in viz_data["series"].items():
                    series_array.append({
                        "name": name,
                        "type": "bar",
                        "data": values
                    })
                return {
                    "xAxis": viz_data["xAxis"],
                    "series": series_array
                }
            # 如果series是数组但不是对象数组，转换为对象数组
            elif isinstance(viz_data["series"], list) and (not viz_data["series"] or not isinstance(viz_data["series"][0], dict)):
                return {
                    "xAxis": viz_data["xAxis"],
                    "series": [{
                        "name": "Value",
                        "type": "bar",
                        "data": viz_data["series"]
                    }]
                }
                
        # 对于饼图，标准化格式
        elif viz_type == "pie" and "series" in viz_data:
            # 如果series已经是正确格式，直接返回
            if isinstance(viz_data["series"], list) and (not viz_data["series"] or isinstance(viz_data["series"][0], dict)):
                return viz_data
                
        # 默认返回原始数据
        return viz_data

    def plot_metrics(self, metrics_data: Dict[str, Any], save_dir: str, round_num: Optional[int] = None) -> None:
        """
        Plot metrics data and save as images.
        
        Args:
            metrics_data (Dict[str, Any]): Dictionary containing metrics data
            save_dir (str): Directory to save the plots
            round_num (Optional[int]): Current step number if applicable
        """
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        #plt.style.use('seaborn')
        
        # Plot step duration
        if 'round_duration' in metrics_data:
            plt.figure(figsize=(10, 6))
            plt.plot(metrics_data['round_duration'], marker='o')
            plt.title('Step Duration Over Time')
            plt.xlabel('Step')
            plt.ylabel('Duration (seconds)')
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, 'round_duration.png'))
            plt.close()
        
        if 'total_tokens' in metrics_data:
            fig = plt.figure(figsize=(10, 6))
            plt.plot(metrics_data['total_tokens'], marker='o')
            plt.title('Total Tokens Over Time')
            plt.xlabel('Step')
            plt.ylabel('Total Tokens')
            plt.savefig(os.path.join(save_dir, 'total_tokens.png'))
        
            plt.close(fig)
        
        # Save metrics data as JSON for reference
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        metrics_file = os.path.join(save_dir, f'metrics_{timestamp}.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=4)

    def collect_metrics(self, env_data: Dict[str, Any], round_num: Optional[int] = None) -> Dict[str, Any]:
        """
        Collect metrics from environment data.
        
        Args:
            env_data (Dict[str, Any]): Environment data dictionary
            round_num (Optional[int]): Current step number if applicable
            
        Returns:
            Dict[str, Any]: Dictionary containing collected metrics
        """
        metrics = {}
        
        # Extract completion rate
        if 'step_data' in env_data:
            step_data = env_data['step_data']

            metrics['round_duration'] = [
                step_data[r]['duration'] 
                for r in sorted(step_data.keys())
            ]

            
            metrics['total_tokens'] = [
                step_data[r]['token_usage']['total_tokens']
                for r in sorted(step_data.keys())
            ]
            metrics['total_prompt_tokens']=[
                step_data[r]['token_usage']['total_prompt_tokens']
                for r in sorted(step_data.keys())
            ]
            metrics['total_completion_tokens']=[
                step_data[r]['token_usage']['total_completion_tokens']
                for r in sorted(step_data.keys())
            ]
            metrics['request_count']=[
                step_data[r]['token_usage']['request_count']
                for r in sorted(step_data.keys())
            ]
        
        
            metrics['event_count'] = [
                step_data[r]['event_count']
                for r in sorted(step_data.keys())
            ]
        
        # # Extract agent participation
        # if 'agent_decisions' in env_data:
        #     metrics['agent_participation'] = env_data['agent_decisions']
        
        # # Extract decision distribution
        # if 'decision_distribution' in env_data:
        #     metrics['decision_distribution'] = env_data['decision_distribution']
        
        return metrics

    def _figure_comment_volume_realtime(
        self,
        raw: Dict[str, Any],
        metric_name: str,
        metric_def: Any,
    ):
        """
        上图：按评论真实时间排序的累计评论量（折线 + 标记；点极多时稀疏采样标记）。
        下图：各时间桶内评论条数柱状图；柱宽等于桶宽、align=edge，相邻柱左右相接无隙。
        """
        times_ms: List[float] = list(raw.get("timestamps_ms") or [])
        edges: List[float] = list(raw.get("hist_bin_edges_ms") or [])
        counts: List[int] = list(raw.get("hist_counts") or [])
        desc = str(raw.get("hist_bucket_description") or "")
        n_comments = int(raw.get("n_comments") or len(times_ms))

        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 8))

        def _ms_to_dt(ms: float) -> datetime:
            return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)

        def _date_fmt_for_span_ms(span_ms: float) -> str:
            """横轴格式：时间跨度很小时用秒，否则只到分钟会像『全在同一分钟』。"""
            if span_ms <= 0:
                return "%Y-%m-%d %H:%M:%S"
            if span_ms < 3_600_000:
                return "%H:%M:%S"
            if span_ms < 172800_000:
                return "%m-%d %H:%M"
            return "%Y-%m-%d %H:%M"

        span_ms = float(max(times_ms) - min(times_ms)) if len(times_ms) > 1 else 0.0
        _tfmt = _date_fmt_for_span_ms(span_ms)

        if times_ms:
            xs = [_ms_to_dt(t) for t in times_ms]
            ys = list(range(1, len(times_ms) + 1))
            npts = len(xs)
            markevery = max(1, npts // 50) if npts > 100 else None
            ax0.plot(
                xs,
                ys,
                color="tab:blue",
                linestyle="-",
                linewidth=1.5,
                marker="o",
                markersize=5 if npts <= 100 else 4,
                markevery=markevery,
                markeredgecolor="white",
                markeredgewidth=0.5,
            )
            ax0.set_ylabel("Cumulative comment count", fontsize=11)
            ax0.set_ylim(bottom=0)
            ax0.grid(True, linestyle="--", alpha=0.5)
            ax0.xaxis.set_major_formatter(mdates.DateFormatter(_tfmt))
        else:
            ax0.text(0.5, 0.5, "No valid comment timestamps", ha="center", va="center", transform=ax0.transAxes)
            ax0.set_ylabel("Cumulative comment count", fontsize=11)

        if edges and counts and len(edges) == len(counts) + 1:
            lefts = [_ms_to_dt(edges[i]) for i in range(len(counts))]
            widths = [
                timedelta(milliseconds=float(edges[i + 1] - edges[i]))
                for i in range(len(counts))
            ]
            ax1.bar(
                lefts,
                counts,
                width=widths,
                align="edge",
                color="steelblue",
                edgecolor="steelblue",
                linewidth=0,
            )
            ax1.set_ylabel("Comments per time bin", fontsize=11)
            ax1.set_xlabel("Comment time (UTC)", fontsize=11)
            ax1.set_ylim(bottom=0)
            ax1.grid(True, linestyle="--", alpha=0.5, axis="y")
            e_span = float(edges[-1] - edges[0]) if len(edges) >= 2 else span_ms
            ax1.xaxis.set_major_formatter(
                mdates.DateFormatter(_date_fmt_for_span_ms(e_span))
            )
        else:
            ax1.text(0.5, 0.5, "No histogram data", ha="center", va="center", transform=ax1.transAxes)
            ax1.set_xlabel("Comment time (UTC)", fontsize=11)

        fig.suptitle(metric_name, fontsize=12, fontweight="bold", y=0.98)
        note = f"Time bins: {desc} · total comments {n_comments}" if desc else f"Total comments {n_comments}"
        fig.text(0.5, 0.02, note, ha="center", fontsize=9, color="gray")
        for ax in (ax0, ax1):
            ax.tick_params(axis="x", rotation=25)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.12, hspace=0.28)
        return fig

    def _figure_repost_volume_realtime(
        self,
        raw: Dict[str, Any],
        metric_name: str,
        metric_def: Any,
    ):
        """
        Top: cumulative repost count vs repost time (sorted by timestamp).
        Bottom: reposts per equal-width time bin (bar width = bin width).
        """
        times_ms: List[float] = list(raw.get("timestamps_ms") or [])
        edges: List[float] = list(raw.get("hist_bin_edges_ms") or [])
        counts: List[int] = list(raw.get("hist_counts") or [])
        desc = str(raw.get("hist_bucket_description") or "")
        n_reposts = int(raw.get("n_reposts") or len(times_ms))

        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 8))

        def _ms_to_dt(ms: float) -> datetime:
            return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)

        if times_ms:
            xs = [_ms_to_dt(t) for t in times_ms]
            ys = list(range(1, len(times_ms) + 1))
            npts = len(xs)
            markevery = max(1, npts // 50) if npts > 100 else None
            ax0.plot(
                xs,
                ys,
                color="tab:green",
                linestyle="-",
                linewidth=1.5,
                marker="o",
                markersize=5 if npts <= 100 else 4,
                markevery=markevery,
                markeredgecolor="white",
                markeredgewidth=0.5,
            )
            ax0.set_ylabel("Cumulative repost count", fontsize=11)
            ax0.set_ylim(bottom=0)
            ax0.grid(True, linestyle="--", alpha=0.5)
            ax0.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
        else:
            ax0.text(0.5, 0.5, "No valid repost timestamps", ha="center", va="center", transform=ax0.transAxes)
            ax0.set_ylabel("Cumulative repost count", fontsize=11)

        if edges and counts and len(edges) == len(counts) + 1:
            lefts = [_ms_to_dt(edges[i]) for i in range(len(counts))]
            widths = [
                timedelta(milliseconds=float(edges[i + 1] - edges[i]))
                for i in range(len(counts))
            ]
            ax1.bar(
                lefts,
                counts,
                width=widths,
                align="edge",
                color="darkseagreen",
                edgecolor="darkseagreen",
                linewidth=0,
            )
            ax1.set_ylabel("Reposts per time bin", fontsize=11)
            ax1.set_xlabel("Repost time (UTC)", fontsize=11)
            ax1.set_ylim(bottom=0)
            ax1.grid(True, linestyle="--", alpha=0.5, axis="y")
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
        else:
            ax1.text(0.5, 0.5, "No histogram data", ha="center", va="center", transform=ax1.transAxes)
            ax1.set_xlabel("Repost time (UTC)", fontsize=11)

        fig.suptitle(metric_name, fontsize=12, fontweight="bold", y=0.98)
        note = f"Time bins: {desc} · total reposts {n_reposts}" if desc else f"Total reposts {n_reposts}"
        fig.text(0.5, 0.02, note, ha="center", fontsize=9, color="gray")
        for ax in (ax0, ax1):
            ax.tick_params(axis="x", rotation=25)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.12, hspace=0.28)
        return fig

    def _save_comment_count_frequency_pair(
        self,
        raw: Dict[str, Any],
        metric_dir: str,
        metric_name: str,
        round_info: str,
        timestamp: str,
    ) -> None:
        """按用户评论条数、按帖子评论条数的频率（百分比）各导出一张点线图。"""
        base = f"{metric_name}{round_info}_{timestamp}"

        def _save_one(
            suffix: str,
            subtitle: str,
            bins: List[Any],
            pct_vals: List[Any],
            xlabel: str,
            ylabel: str,
            color: str,
            empty_msg: str,
        ) -> None:
            n = len(bins)
            xlabels = [str(int(b)) if isinstance(b, (int, float)) and float(b) == int(b) else str(b) for b in bins]
            fig, ax = plt.subplots(figsize=(10, 6))
            if n == 0:
                ax.text(
                    0.5,
                    0.5,
                    empty_msg,
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            else:
                vals = list(pct_vals)
                while len(vals) < n:
                    vals.append(0.0)
                vals = vals[:n]
                xs = list(range(n))
                ax.plot(
                    xs,
                    vals,
                    color=color,
                    linestyle="-",
                    linewidth=1.8,
                    marker="o",
                    markersize=6,
                    markeredgecolor="white",
                    markeredgewidth=0.6,
                )
                ax.set_xticks(xs)
                ax.set_xticklabels(xlabels)
                ax.set_ylim(bottom=0)
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f"{metric_name}\n{subtitle}", fontsize=11, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            out = os.path.join(metric_dir, f"{base}_{suffix}.png")
            fig.savefig(out, dpi=300, bbox_inches="tight")
            plt.close(fig)

        ub = list(raw.get("user_comment_bins") or [])
        up = list(raw.get("user_frequency_pct") or [])
        nb = list(raw.get("note_comment_bins") or [])
        npct = list(raw.get("note_frequency_pct") or [])
        n_pool_u = int(raw.get("n_users_in_pool") or raw.get("n_commenting_users") or 0)
        nn = int(raw.get("n_notes") or 0)

        _save_one(
            "commenting_users",
            f"% of users in pool (N={n_pool_u}; note authors union commenters) — exact comment count per user",
            ub,
            up,
            "Comments per user (0 = none)",
            "Frequency (% of users in pool)",
            "tab:blue",
            "No users in content pool (no author/commenter user_id)",
        )
        _save_one(
            "notes",
            f"% of all notes (N={nn}) — exact comment count under each note",
            nb,
            npct,
            "Comments under note",
            "Frequency (% of all notes)",
            "tab:orange",
            "No notes in content pool",
        )

    def _save_single_comment_count_frequency(
        self,
        raw: Dict[str, Any],
        metric_dir: str,
        metric_name: str,
        round_info: str,
        timestamp: str,
    ) -> None:
        """Export one comment-count frequency line chart (user or note dimension)."""
        viz = raw.get("_viz_kind")
        base = f"{metric_name}{round_info}_{timestamp}"
        bins = list(raw.get("repost_bins") or raw.get("comment_bins") or [])
        pct = list(raw.get("frequency_pct") or [])

        if viz == "user_comment_count_freq_bar":
            n = int(raw.get("n_users_in_pool") or 0)
            subtitle = (
                f"% of users in pool (N={n}; note authors union commenters) "
                "— exact comment count per user"
            )
            xlabel = "Comments per user (0 = none)"
            ylabel = "Frequency (% of users in pool)"
            color = "tab:blue"
            empty_msg = "No users in content pool (no author/commenter user_id)"
            suffix = "commenting_users"
        elif viz == "note_comment_count_freq_bar":
            n = int(raw.get("n_notes") or 0)
            subtitle = f"% of all notes (N={n}) — exact comment count under each note"
            xlabel = "Comments under note"
            ylabel = "Frequency (% of all notes)"
            color = "tab:orange"
            empty_msg = "No notes in content pool"
            suffix = "notes"
        elif viz == "user_repost_count_freq_bar":
            n = int(raw.get("n_users_in_pool") or 0)
            subtitle = (
                f"% of users in pool (N={n}; basis={raw.get('user_count_basis', 'content_pool')}) "
                "— exact repost count per user"
            )
            xlabel = "Reposts per user (0 = none)"
            ylabel = "Frequency (% of users in pool)"
            color = "tab:blue"
            empty_msg = "No users in content pool"
            suffix = "reposting_users"
        elif viz == "root_repost_count_freq_bar":
            n = int(raw.get("n_root_tweets") or 0)
            subtitle = f"% of root blogs (N={n}) — repost nodes under each root"
            xlabel = "Repost nodes under root blog"
            ylabel = "Frequency (% of root blogs)"
            color = "tab:orange"
            empty_msg = "No root blogs in content pool"
            suffix = "root_blogs"
        else:
            return

        n_bins = len(bins)
        xlabels = [
            str(int(b)) if isinstance(b, (int, float)) and float(b) == int(b) else str(b)
            for b in bins
        ]
        fig, ax = plt.subplots(figsize=(10, 6))
        if n_bins == 0:
            ax.text(
                0.5,
                0.5,
                empty_msg,
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        else:
            vals = list(pct)
            while len(vals) < n_bins:
                vals.append(0.0)
            vals = vals[:n_bins]
            xs = list(range(n_bins))
            ax.plot(
                xs,
                vals,
                color=color,
                linestyle="-",
                linewidth=1.8,
                marker="o",
                markersize=6,
                markeredgecolor="white",
                markeredgewidth=0.6,
            )
            ax.set_xticks(xs)
            ax.set_xticklabels(xlabels)
            ax.set_ylim(bottom=0)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{metric_name}\n{subtitle}", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        out = os.path.join(metric_dir, f"{base}_{suffix}.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _save_repost_count_frequency_pair(
        self,
        raw: Dict[str, Any],
        metric_dir: str,
        metric_name: str,
        round_info: str,
        timestamp: str,
    ) -> None:
        """Root-tweet repost totals vs user repost totals: line+marker, English labels, Y = raw counts."""
        base = f"{metric_name}{round_info}_{timestamp}"

        def _save_one(
            suffix: str,
            subtitle: str,
            bins: List[Any],
            count_vals: List[Any],
            xlabel: str,
            ylabel: str,
            color: str,
            empty_msg: str,
        ) -> None:
            n = len(bins)
            xlabels = [str(int(b)) if isinstance(b, (int, float)) and float(b) == int(b) else str(b) for b in bins]
            fig, ax = plt.subplots(figsize=(10, 6))
            if n == 0:
                ax.text(
                    0.5,
                    0.5,
                    empty_msg,
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            else:
                vals = list(count_vals)
                while len(vals) < n:
                    vals.append(0)
                vals = vals[:n]
                xs = list(range(n))
                ax.plot(
                    xs,
                    vals,
                    color=color,
                    linestyle="-",
                    linewidth=1.8,
                    marker="o",
                    markersize=6,
                    markeredgecolor="white",
                    markeredgewidth=0.6,
                )
                ax.set_xticks(xs)
                ax.set_xticklabels(xlabels)
                ax.set_ylim(bottom=0)
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f"{metric_name}\n{subtitle}", fontsize=11, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            out = os.path.join(metric_dir, f"{base}_{suffix}.png")
            fig.savefig(out, dpi=300, bbox_inches="tight")
            plt.close(fig)

        rb = list(raw.get("root_repost_bins") or [])
        rc = list(raw.get("root_repost_counts") or [])
        nr = int(raw.get("n_root_tweets") or 0)
        ub = list(raw.get("user_repost_bins") or [])
        uc = list(raw.get("user_repost_counts") or [])
        nu = int(raw.get("n_users_in_pool") or 0)

        viz_kind = raw.get("_viz_kind")
        if viz_kind == "received_propagation_count_freq_bar":
            user_subtitle = (
                f"N={nu} users (all authors in pool) — incoming propagation to their posts "
                "(seed-attributed, 1-hop parent)"
            )
            user_xlabel = (
                "Direct incoming propagation per user (seed trees, immediate parent only)"
            )
        else:
            user_subtitle = (
                "N={nu} users (all authors in pool) — repost entries per user (originals excluded)"
            )
            user_xlabel = "Reposts authored per user"

        _save_one(
            "root_tweets",
            f"N={nr} root tweets — reposts under each root (multi-level)",
            rb,
            rc,
            "Total reposts under root (all levels)",
            "Number of root tweets",
            "tab:blue",
            "No root tweets in content pool",
        )
        _save_one(
            "users",
            user_subtitle,
            ub,
            uc,
            user_xlabel,
            "Number of users",
            "tab:orange",
            "No users in content pool",
        )

    def _figure_comment_top_vs_reply_timeseries(
        self,
        data: Dict[str, Any],
        metric_name: str,
        metric_def: Any,
    ):
        """1-hop vs 多条（有 parent）评论条数：按监控采样序双子图折线（与 Comment Generation 同为按步采样）。"""
        series = data.get("series") or {}
        timestamps = data.get("xAxis") or []
        top_vals = list(series.get("top_level_comments") or [])
        reply_vals = list(series.get("reply_comments") or [])
        n = len(timestamps)
        if n == 0:
            fig, _ = plt.subplots(1, 1, figsize=(12, 4))
            plt.text(0.5, 0.5, "No time series data", ha="center", va="center", transform=plt.gca().transAxes)
            plt.title(metric_name)
            plt.tight_layout()
            return fig

        while len(top_vals) < n:
            top_vals.append(None)
        while len(reply_vals) < n:
            reply_vals.append(None)
        top_vals = top_vals[:n]
        reply_vals = reply_vals[:n]

        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        def _plot_ax(ax, vals: List[Any], ylabel: str, color: str) -> None:
            ax.plot(
                range(n),
                vals,
                marker="o",
                linestyle="-",
                markersize=4,
                color=color,
            )
            ax.set_ylabel(ylabel, fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.6)

        _plot_ax(ax0, top_vals, "1-hop comment count", "tab:blue")
        _plot_ax(ax1, reply_vals, "multi-hop comment count", "tab:orange")

        max_ticks = 10
        if n > 1:
            step = max(1, n // max_ticks)
            tick_indices = list(range(0, n, step))
            tick_labels = [
                timestamps[i].strftime("%H:%M:%S")
                if i < len(timestamps) and isinstance(timestamps[i], datetime)
                else (str(timestamps[i]) if i < len(timestamps) else "")
                for i in tick_indices
            ]
            ax1.set_xticks(tick_indices)
            ax1.set_xticklabels(tick_labels, rotation=30, ha="right")
        elif n == 1:
            tl = (
                timestamps[0].strftime("%H:%M:%S")
                if timestamps and isinstance(timestamps[0], datetime)
                else str(timestamps[0])
            )
            ax1.set_xticks([0])
            ax1.set_xticklabels([tl])

        ax1.set_xlabel("Sample index", fontsize=11)
        fig.suptitle(metric_name, fontsize=12, fontweight="bold", y=0.98)
        fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.14, hspace=0.22)
        return fig

    def _figure_comment_diversity_timeseries(
        self,
        data: Dict[str, Any],
        metric_name: str,
        metric_def: Any,
    ):
        """Multi-series diversity metrics on one chart (TTR, Distinct-2/3, 1-Self-BLEU, Div_sem)."""
        series = data.get("series") or {}
        timestamps = data.get("xAxis") or []
        keys = [
            ("ttr", "TTR", "tab:blue"),
            ("distinct_2", "Distinct-2", "tab:orange"),
            ("distinct_3", "Distinct-3", "tab:green"),
            ("one_minus_self_bleu", "1-Self-BLEU", "tab:red"),
            ("div_sem", "Div_sem", "tab:purple"),
        ]
        n = len(timestamps)
        fig, ax = plt.subplots(figsize=(12, 6))
        if n == 0:
            ax.text(0.5, 0.5, "No time series data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(metric_name)
            plt.tight_layout()
            return fig

        xs = list(range(n))
        for key, label, color in keys:
            vals = list(series.get(key) or [])
            while len(vals) < n:
                vals.append(None)
            vals = vals[:n]
            if not any(v is not None for v in vals):
                continue
            ax.plot(xs, vals, marker="o", linestyle="-", markersize=4, color=color, label=label)

        ax.set_xlabel("Sample index", fontsize=11)
        ax.set_ylabel("Diversity score", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="best", fontsize=9)
        fig.suptitle(metric_name, fontsize=12, fontweight="bold", y=0.98)
        plt.tight_layout()
        return fig

    def _save_repost_hop_depth_timeseries_figures(
        self,
        data: Dict[str, Any],
        metric_dir: str,
        metric_name: str,
        round_info: str,
        timestamp: str,
    ) -> None:
        """
        Multiple PNGs: up to 4 hop-depth series per figure (line+marker), English labels.
        Filenames: {metric}{round}_part{i}_hop{lo}-{hi}_{timestamp}.png
        """
        series = data.get("series") or {}
        timestamps = data.get("xAxis") or []
        hop_keys = [k for k in series if re.match(r"^hop_\d+$", str(k))]
        hop_keys.sort(key=lambda x: int(str(x).split("_")[1]))
        n = len(timestamps)
        stem = f"{metric_name}{round_info}_{timestamp}"

        if n == 0 or not hop_keys:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.text(
                0.5,
                0.5,
                "No time series data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(metric_name, fontsize=11, fontweight="bold")
            plt.tight_layout()
            fig.savefig(
                os.path.join(metric_dir, f"{stem}_part1_empty.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)
            return

        max_per_fig = 4
        part = 0
        for start in range(0, len(hop_keys), max_per_fig):
            chunk = hop_keys[start : start + max_per_fig]
            part += 1
            fig, ax = plt.subplots(figsize=(12, 5))
            for i, k in enumerate(chunk):
                vals = list(series.get(k) or [])
                while len(vals) < n:
                    vals.append(None)
                vals = vals[:n]
                hop_n = int(str(k).split("_")[1])
                label = f"{hop_n}-hop"
                ax.plot(
                    range(n),
                    vals,
                    marker="o",
                    linestyle="-",
                    markersize=4,
                    color=f"C{i % 10}",
                    label=label,
                )
            ax.set_ylabel("Cumulative repost count (in pool)", fontsize=11)
            ax.set_xlabel("Sample index", fontsize=11)
            lo = int(str(chunk[0]).split("_")[1])
            hi = int(str(chunk[-1]).split("_")[1])
            ax.set_title(
                f"{metric_name}\n(part {part}: hop depths {lo}-{hi})",
                fontsize=11,
                fontweight="bold",
            )
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="best", fontsize=9)
            max_ticks = 10
            if n > 1:
                step = max(1, n // max_ticks)
                tick_indices = list(range(0, n, step))
                tick_labels = [
                    timestamps[i].strftime("%H:%M:%S")
                    if i < len(timestamps) and isinstance(timestamps[i], datetime)
                    else (str(timestamps[i]) if i < len(timestamps) else "")
                    for i in tick_indices
                ]
                ax.set_xticks(tick_indices)
                ax.set_xticklabels(tick_labels, rotation=30, ha="right")
            elif n == 1:
                tl = (
                    timestamps[0].strftime("%H:%M:%S")
                    if timestamps and isinstance(timestamps[0], datetime)
                    else str(timestamps[0])
                )
                ax.set_xticks([0])
                ax.set_xticklabels([tl])
            plt.tight_layout()
            fn = f"{stem}_part{part}_hop{lo}-{hi}.png"
            fig.savefig(os.path.join(metric_dir, fn), dpi=300, bbox_inches="tight")
            plt.close(fig)

    @staticmethod
    def _plt_legend_if_labeled() -> None:
        """Only show legend when at least one artist has a label (avoids matplotlib UserWarning)."""
        _, labels = plt.gca().get_legend_handles_labels()
        if labels:
            plt.legend(
                loc="best",
                frameon=True,
                fancybox=True,
                framealpha=0.7,
            )

    def plot_registered_metrics(self, save_dir: str, round_num: Optional[int] = None) -> None:
        """
        Plot registered (scene-specific) metrics data and save them as images.
        
        Args:
            save_dir (str): Directory to save the plots
            round_num (Optional[int]): Current step number if applicable
        """
        if not self.results:
            logger.warning("No registered metrics data available to plot")
            return
            
        # Create scene-specific metrics directory
        scene_metrics_dir = os.path.join(save_dir, 'scene_metrics')
        os.makedirs(scene_metrics_dir, exist_ok=True)
        
        # Plot each registered metric
        for metric_name, result in self.results.items():
            fig = None
            try:
                metric_def = self.metrics.get(metric_name)
                if not metric_def:
                    continue
                    
                viz_type = metric_def.visualization_type
                fig = plt.figure(figsize=(12, 7))
                
                # Create metric-specific directory
                metric_dir = os.path.join(scene_metrics_dir, metric_name)
                os.makedirs(metric_dir, exist_ok=True)
                data = self.get_metric_data(metric_name, format="matplotlib")
                
                line_scalar_plotted = False
                line_wrote_distribution_snapshots = False
                save_comment_count_freq_pair = False
                save_single_comment_count_freq = False
                save_repost_count_freq_pair = False
                save_repost_hop_depth_figures = False
                if (
                    viz_type == "line"
                    and isinstance(result.raw_data, dict)
                    and result.raw_data.get("_viz_kind") == "comment_realtime"
                ):
                    plt.close(fig)
                    fig = self._figure_comment_volume_realtime(
                        result.raw_data, metric_name, metric_def
                    )
                    line_scalar_plotted = True
                elif (
                    viz_type == "line"
                    and isinstance(result.raw_data, dict)
                    and result.raw_data.get("_viz_kind") == "repost_realtime"
                ):
                    plt.close(fig)
                    fig = self._figure_repost_volume_realtime(
                        result.raw_data, metric_name, metric_def
                    )
                    line_scalar_plotted = True
                elif (
                    viz_type == "line"
                    and isinstance(result.raw_data, dict)
                    and result.raw_data.get("_viz_kind")
                    in ("user_comment_count_freq_bar", "note_comment_count_freq_bar",
                        "user_repost_count_freq_bar", "root_repost_count_freq_bar")
                ):
                    plt.close(fig)
                    fig = None
                    save_single_comment_count_freq = True
                    line_scalar_plotted = True
                elif (
                    viz_type == "line"
                    and isinstance(result.raw_data, dict)
                    and result.raw_data.get("_viz_kind") == "comment_count_freq_bar"
                ):
                    plt.close(fig)
                    fig = None
                    save_comment_count_freq_pair = True
                    line_scalar_plotted = True
                elif (
                    viz_type == "line"
                    and isinstance(result.raw_data, dict)
                    and result.raw_data.get("_viz_kind")
                    in ("repost_count_freq_bar", "received_propagation_count_freq_bar")
                ):
                    plt.close(fig)
                    fig = None
                    save_repost_count_freq_pair = True
                    line_scalar_plotted = True
                elif (
                    viz_type == "line"
                    and metric_name in (
                        "Diffusion Top-Level vs Reply Over Time",
                    )
                    and data
                    and isinstance(data.get("series"), dict)
                    and "top_level_comments" in data["series"]
                    and "reply_comments" in data["series"]
                ):
                    plt.close(fig)
                    fig = self._figure_comment_top_vs_reply_timeseries(
                        data, metric_name, metric_def
                    )
                    line_scalar_plotted = True
                elif (
                    viz_type == "line"
                    and metric_name in ("Comment Diversity", "Text Diversity")
                    and data
                    and isinstance(data.get("series"), dict)
                    and "ttr" in data["series"]
                ):
                    plt.close(fig)
                    fig = self._figure_comment_diversity_timeseries(
                        data, metric_name, metric_def
                    )
                    line_scalar_plotted = True
                elif (
                    viz_type == "line"
                    and metric_name in (
                        "Diffusion Hop Depth Over Time",
                    )
                    and data
                    and isinstance(data.get("series"), dict)
                    and any(
                        re.match(r"^hop_\d+$", str(k))
                        for k in (data.get("series") or {})
                    )
                ):
                    plt.close(fig)
                    fig = None
                    save_repost_hop_depth_figures = True
                    line_scalar_plotted = True
                elif viz_type == "line":
                    if data and "series" in data and isinstance(data["series"], dict):
                        snap_dir = os.path.join(metric_dir, "distribution_snapshots")
                        timestamp_snap = datetime.now().strftime('%Y%m%d_%H%M%S')
                        round_info_snap = f"_round{round_num}" if round_num is not None else ""
                        snap_json_payload: Dict[str, Any] = {}

                        for series_name, series_values in data["series"].items():
                            if (
                                not series_values
                                or not isinstance(series_values[0], list)
                            ):
                                continue
                            os.makedirs(snap_dir, exist_ok=True)
                            line_wrote_distribution_snapshots = True
                            snap_json_payload[series_name] = series_values

                            if series_name == "sorted_note_comment_counts":
                                x_label = "Note Frequency"
                                y_label = "Comments on note"
                                title_extra = "Per-note comment count distribution"
                            elif series_name == "sorted_counts":
                                x_label = "User Frequency"
                                y_label = "Total comments by user"
                                title_extra = "Per-user total comment distribution"
                            else:
                                x_label = "Frequency"
                                y_label = "Value"
                                title_extra = series_name

                            for step_i, counts in enumerate(series_values):
                                if counts is None:
                                    continue
                                if not isinstance(counts, list):
                                    continue
                                if len(counts) == 0:
                                    continue
                                fig_s = plt.figure(figsize=(10, 5))
                                xs = list(range(len(counts)))
                                plt.plot(
                                    xs,
                                    counts,
                                    marker='o',
                                    linestyle='-',
                                    markersize=3,
                                    linewidth=1,
                                )
                                plt.xlabel(x_label, fontsize=11)
                                plt.ylabel(y_label, fontsize=11)
                                plt.title(
                                    f"{title_extra} | {metric_name} | sample #{step_i}",
                                    fontsize=11,
                                    fontweight='bold',
                                )
                                plt.grid(True, linestyle='--', alpha=0.5)
                                plt.tight_layout()
                                safe_sn = re.sub(r"[^\w\-.]+", "_", str(series_name))[:80]
                                snap_fn = (
                                    f"{metric_name}{round_info_snap}_{safe_sn}_step{step_i:04d}_{timestamp_snap}.png"
                                )
                                plt.savefig(
                                    os.path.join(snap_dir, snap_fn),
                                    dpi=200,
                                    bbox_inches='tight',
                                )
                                plt.close(fig_s)

                        if snap_json_payload:
                            try:
                                with open(
                                    os.path.join(
                                        snap_dir,
                                        f"{metric_name}{round_info_snap}_all_distribution_steps_{timestamp_snap}.json",
                                    ),
                                    "w",
                                    encoding="utf-8",
                                ) as jf:
                                    json.dump(snap_json_payload, jf, indent=2)
                            except Exception as ej:
                                logger.warning(
                                    f"Failed to write distribution snapshot JSON for {metric_name}: {ej}"
                                )

                        scalar_series = {
                            k: v
                            for k, v in data["series"].items()
                            if not (
                                v
                                and isinstance(v[0], list)
                            )
                        }

                        if not line_wrote_distribution_snapshots and data.get("xAxis"):
                            timestamps = data["xAxis"]
                            num_points = len(timestamps)
                            if num_points > 0:
                                if isinstance(scalar_series, dict) and scalar_series:
                                    for series_name, series_values in scalar_series.items():
                                        plt.plot(
                                            range(num_points),
                                            series_values,
                                            marker='o',
                                            linestyle='-',
                                            markersize=4,
                                            label=series_name,
                                        )
                                    MonitorManager._plt_legend_if_labeled()
                                elif not isinstance(data["series"], dict):
                                    plt.plot(
                                        range(num_points),
                                        data["series"],
                                        marker='o',
                                        linestyle='-',
                                        markersize=4,
                                    )
                                else:
                                    for series_name, series_values in data[
                                        "series"
                                    ].items():
                                        if (
                                            series_values
                                            and isinstance(series_values[0], list)
                                        ):
                                            continue
                                        plt.plot(
                                            range(num_points),
                                            series_values,
                                            marker='o',
                                            linestyle='-',
                                            markersize=4,
                                            label=series_name,
                                        )
                                    MonitorManager._plt_legend_if_labeled()

                                max_ticks = 10
                                if num_points > 1:
                                    step = max(1, num_points // max_ticks)
                                    tick_indices = range(0, num_points, step)
                                    tick_labels = [
                                        timestamps[i].strftime('%H:%M:%S')
                                        if isinstance(timestamps[i], datetime)
                                        else timestamps[i]
                                        for i in tick_indices
                                    ]
                                    plt.xticks(
                                        tick_indices,
                                        tick_labels,
                                        rotation=30,
                                        ha='right',
                                    )
                                elif num_points == 1:
                                    tick_labels = [
                                        timestamps[0].strftime('%H:%M:%S')
                                        if isinstance(timestamps[0], datetime)
                                        else timestamps[0]
                                    ]
                                    plt.xticks([0], tick_labels)

                                plt.title(
                                    f'{metric_name} Over Time',
                                    fontsize=14,
                                    fontweight='bold',
                                )
                                plt.xlabel('Step', fontsize=12)
                                plt.ylabel(metric_name, fontsize=12)
                                plt.grid(True, linestyle='--', alpha=0.6)
                                plt.tight_layout()
                                line_scalar_plotted = True
                        elif line_wrote_distribution_snapshots and data.get("xAxis") and scalar_series:
                            timestamps = data["xAxis"]
                            num_points = len(timestamps)
                            if num_points > 0:
                                for series_name, series_values in scalar_series.items():
                                    plt.plot(
                                        range(num_points),
                                        series_values,
                                        marker='o',
                                        linestyle='-',
                                        markersize=4,
                                        label=series_name,
                                    )
                                MonitorManager._plt_legend_if_labeled()
                                max_ticks = 10
                                if num_points > 1:
                                    step = max(1, num_points // max_ticks)
                                    tick_indices = range(0, num_points, step)
                                    tick_labels = [
                                        timestamps[i].strftime('%H:%M:%S')
                                        if isinstance(timestamps[i], datetime)
                                        else timestamps[i]
                                        for i in tick_indices
                                    ]
                                    plt.xticks(
                                        tick_indices,
                                        tick_labels,
                                        rotation=30,
                                        ha='right',
                                    )
                                elif num_points == 1:
                                    tick_labels = [
                                        timestamps[0].strftime('%H:%M:%S')
                                        if isinstance(timestamps[0], datetime)
                                        else timestamps[0]
                                    ]
                                    plt.xticks([0], tick_labels)
                                plt.title(
                                    f'{metric_name} Over Time',
                                    fontsize=14,
                                    fontweight='bold',
                                )
                                plt.xlabel('Step', fontsize=12)
                                plt.ylabel(metric_name, fontsize=12)
                                plt.grid(True, linestyle='--', alpha=0.6)
                                plt.tight_layout()
                                line_scalar_plotted = True

                elif viz_type == "bar":
                    if data and "xAxis" in data and "series" in data:
                        categories = data["xAxis"]
                        
                        if isinstance(data["series"], dict):
                            series_count = len(data["series"])
                            width = 0.8 / series_count  
                            
                            for i, (series_name, values) in enumerate(data["series"].items()):
                                positions = [j + (i - series_count/2 + 0.5) * width for j in range(len(categories))]
                                plt.bar(positions, values, width, label=series_name)
                            
                            plt.xticks(range(len(categories)), categories, rotation=45)
                            MonitorManager._plt_legend_if_labeled()
                        else:
                            plt.bar(categories, data["series"])
                        
                        plt.title(f'{metric_name} Distribution', fontsize=14, fontweight='bold')
                        plt.xlabel('Category', fontsize=12)
                        plt.ylabel('Value', fontsize=12)
                        plt.tight_layout()
                        
                elif viz_type == "pie":
                    if data:
                        if "categories" in data and "values" in data:
                            plt.pie(data["values"], labels=data["categories"], autopct='%1.1f%%', 
                                  shadow=True, startangle=90)
                        elif "series" in data and isinstance(data["series"], list):
                            values = [item["value"] for item in data["series"]]
                            labels = [item["name"] for item in data["series"]]
                            plt.pie(values, labels=labels, autopct='%1.1f%%', 
                                  shadow=True, startangle=90)
                        
                        plt.title(f'{metric_name} Distribution', fontsize=14, fontweight='bold')
                        plt.axis('equal')
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                round_info = f"_round{round_num}" if round_num is not None else ""
                filename = f"{metric_name}{round_info}_{timestamp}.png"
                skip_main_png = (
                    (
                        viz_type == "line"
                        and line_wrote_distribution_snapshots
                        and not line_scalar_plotted
                    )
                    or (
                        viz_type == "line"
                        and metric_name in (
                            "Posting User Diffusion Behavior",
                        )
                        and not line_scalar_plotted
                    )
                    or (
                        viz_type == "line"
                        and metric_name == "Root Author Self-Repost Behavior"
                        and not line_scalar_plotted
                    )
                    or save_comment_count_freq_pair
                    or save_single_comment_count_freq
                    or save_repost_count_freq_pair
                    or save_repost_hop_depth_figures
                )
                if save_comment_count_freq_pair and isinstance(result.raw_data, dict):
                    self._save_comment_count_frequency_pair(
                        result.raw_data,
                        metric_dir,
                        metric_name,
                        round_info,
                        timestamp,
                    )
                elif save_single_comment_count_freq and isinstance(result.raw_data, dict):
                    self._save_single_comment_count_frequency(
                        result.raw_data,
                        metric_dir,
                        metric_name,
                        round_info,
                        timestamp,
                    )
                elif save_repost_count_freq_pair and isinstance(result.raw_data, dict):
                    self._save_repost_count_frequency_pair(
                        result.raw_data,
                        metric_dir,
                        metric_name,
                        round_info,
                        timestamp,
                    )
                elif save_repost_hop_depth_figures and data and isinstance(data, dict):
                    self._save_repost_hop_depth_timeseries_figures(
                        data,
                        metric_dir,
                        metric_name,
                        round_info,
                        timestamp,
                    )
                elif not skip_main_png:
                    plt.savefig(os.path.join(metric_dir, filename), dpi=300, bbox_inches='tight')
                
                if (
                    metric_name in (
                        "Posting User Diffusion Behavior",
                    )
                    and isinstance(result.raw_data, dict)
                ):
                    data_filename = f"{metric_name}{round_info}_{timestamp}.csv"
                    self._write_posting_user_behavior_csv(
                        os.path.join(metric_dir, data_filename),
                        result.raw_data,
                    )
                elif (
                    metric_name == "Root Author Self-Repost Behavior"
                    and isinstance(result.raw_data, dict)
                ):
                    data_filename = f"{metric_name}{round_info}_{timestamp}.csv"
                    self._write_root_author_self_repost_csv(
                        os.path.join(metric_dir, data_filename),
                        result.raw_data,
                    )
                else:
                    data_filename = f"{metric_name}{round_info}_{timestamp}.json"
                    with open(os.path.join(metric_dir, data_filename), "w") as f:
                        if (
                            metric_name in (
                                "Diffusion Volume Real Time",
                            )
                            and isinstance(result.raw_data, dict)
                        ):
                            json.dump(result.raw_data, f, ensure_ascii=False, indent=2)
                        elif (
                            isinstance(result.raw_data, dict)
                            and result.raw_data.get("_viz_kind")
                            in (
                                "comment_count_freq_bar",
                                "user_comment_count_freq_bar",
                                "note_comment_count_freq_bar",
                                "user_repost_count_freq_bar",
                                "root_repost_count_freq_bar",
                                "comment_diversity",
                                "repost_diversity",
                                "comment_max_reference_similarity",
                                "repost_max_reference_similarity",
                            )
                        ):
                            json.dump(result.raw_data, f, ensure_ascii=False, indent=2)
                        else:
                            json.dump(data, f, indent=4)
                    
            except Exception as e:
                logger.error(f"Error plotting metric {metric_name}: {e}")
            finally:
                # Always close the figure, even if an exception occurs
                if fig is not None:
                    plt.close(fig)
                
    def export_metrics_as_images(self, save_dir: str, round_num: Optional[int] = None) -> None:
        """Save all metrics as local image files."""
        try:
            # Create save directory
            os.makedirs(save_dir, exist_ok=True)

            # Export general metric charts
            general_metrics = self.collect_metrics(self.env.data if hasattr(self, 'env') and self.env else {}, round_num)

            general_dir = os.path.join(save_dir, 'general')
            self.plot_metrics(general_metrics, general_dir, round_num)
            logger.info(f"Saved general metrics plots to {general_dir}")

            # Export registered scene-specific metric charts
            self.plot_registered_metrics(save_dir, round_num)
            logger.info(f"Saved registered metrics plots to {save_dir}")
        finally:
            # Make sure to close any remaining figures
            plt.close('all')

    def _normalize_line_data(self, raw_result: Any) -> Any:
        """Normalize line chart data format."""
        if isinstance(raw_result, dict) and raw_result.get("_viz_kind") == "comment_realtime":
            n = raw_result.get("n_comments", 0)
            try:
                nv = float(n) if n is not None else 0.0
            except (TypeError, ValueError):
                nv = 0.0
            return {"_comment_realtime_placeholder": nv}

        if isinstance(raw_result, dict) and raw_result.get("_viz_kind") in (
            "comment_count_freq_bar",
            "user_comment_count_freq_bar",
            "note_comment_count_freq_bar",
            "user_repost_count_freq_bar",
            "root_repost_count_freq_bar",
        ):
            return {"_comment_count_freq_placeholder": 0.0}

        if isinstance(raw_result, dict) and raw_result.get("_viz_kind") in (
            "comment_diversity",
            "repost_diversity",
        ):
            out: Dict[str, float] = {}
            for k in (
                "ttr",
                "distinct_2",
                "distinct_3",
                "one_minus_self_bleu",
                "div_sem",
                "n_comments",
                "n_reposts",
            ):
                v = raw_result.get(k)
                if isinstance(v, (int, float)) and v is not None:
                    out[k] = float(v)
            return out if out else {"_comment_diversity_placeholder": 0.0}

        if isinstance(raw_result, dict) and raw_result.get("_viz_kind") in (
            "comment_max_reference_similarity",
            "repost_max_reference_similarity",
        ):
            v = raw_result.get("mean_max_cosine_similarity")
            if isinstance(v, (int, float)):
                return {"mean_max_cosine_similarity": float(v)}
            return {"_comment_max_ref_sim_placeholder": 0.0}

        if isinstance(raw_result, dict) and raw_result.get("_viz_kind") == "repost_realtime":
            n = raw_result.get("n_reposts", 0)
            try:
                nv = float(n) if n is not None else 0.0
            except (TypeError, ValueError):
                nv = 0.0
            return {"_repost_realtime_placeholder": nv}

        if isinstance(raw_result, dict) and raw_result.get("_viz_kind") in (
            "repost_count_freq_bar",
            "received_propagation_count_freq_bar",
        ):
            return {"_repost_count_freq_placeholder": 0.0}

        if isinstance(raw_result, dict) and "users" in raw_result:
            users = raw_result.get("users")
            if isinstance(users, list) and (
                not users
                or (
                    isinstance(users[0], dict)
                    and "user_id" in users[0]
                )
            ):
                return {"_posting_behavior_placeholder": 0.0}

        if isinstance(raw_result, dict) and raw_result:
            if all(
                isinstance(v, list) and all(isinstance(x, (int, float)) for x in v)
                for v in raw_result.values()
            ):
                return {k: [int(x) for x in v] for k, v in raw_result.items()}

        if isinstance(raw_result, dict):
            for key, value in raw_result.items():
                if not isinstance(value, (int, float, bool)) and value is not None:
                    try:
                        return self._flatten_nested_dict(raw_result)
                    except:
                        pass
            return raw_result
            
        if isinstance(raw_result, (int, float, bool, str)) or raw_result is None:
            return {"default": raw_result}
            
        try:
            if isinstance(raw_result, (list, tuple)) and len(raw_result) > 0:
                if all(isinstance(item, (list, tuple)) and len(item) == 2 for item in raw_result):
                    return {str(item[0]): item[1] for item in raw_result}
                return {f"series_{i}": val for i, val in enumerate(raw_result) if val is not None}
        except Exception:
            pass
            
        return {"default": raw_result}
        
    def _flatten_nested_dict(self, nested_dict: Dict, prefix: str = "") -> Dict:
        """Flatten nested dictionary into a single level dictionary."""
        result = {}
        for key, value in nested_dict.items():
            new_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                result.update(self._flatten_nested_dict(value, new_key))
            elif isinstance(value, (int, float, bool)) or value is None:
                result[new_key] = value
        return result