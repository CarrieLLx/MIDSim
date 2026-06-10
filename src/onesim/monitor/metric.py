from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass, field
from datetime import datetime
import time

@dataclass
class VariableSpec:
    """Describe the source of variables required for metric calculation"""
    
    name: str  # Variable name
    source_type: str  # Source type: "env" or "agent"
    path: str  # Variable path in data/profile (supports dot notation, e.g. "economy.gdp")
    required: bool = True  # Whether required
    agent_type: Optional[str] = None  # If source_type is "agent", specify agent type

@dataclass
class MetricDefinition:
    """Metric definition class, describing all information about a monitoring metric"""
    
    name: str  # Unique metric name
    description: str  # Metric description
    visualization_type: str  # Visualization type: "bar", "pie", "line"
    variables: List[VariableSpec]  # Required variables list
    calculation_function: str  # Calculation function name
    update_interval: int = 60  # Update frequency (seconds)
    visualization_config: Dict = field(default_factory=dict)  # ECharts visualization configuration
    
    def __post_init__(self):
        # Validate visualization type
        valid_types = ["bar", "pie", "line"]
        if self.visualization_type not in valid_types:
            raise ValueError(f"Visualization type must be one of {', '.join(valid_types)}")

@dataclass
class MetricResult:
    """Metric calculation result structure"""
    
    metric_name: str  # Corresponding metric name
    raw_data: Any  # Original calculation result
    visualization_data: Dict  # Data format compatible with ECharts
    timestamp: float = field(default_factory=time.time)  # Calculation timestamp
    metadata: Dict = field(default_factory=dict)  # Additional metadata
    
    @property
    def formatted_time(self) -> str:
        """Return formatted time string"""
        return datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S')


class TimeSeriesMetricData:
    """Handle time series metric data, for line chart"""
    
    def __init__(self, max_points: int = 1000):
        """
        Initialize time series data storage
        
        Args:
            max_points: Maximum number of historical data points to save
        """
        self.timestamps: List[float] = []
        self.series_data: Dict[str, List[Any]] = {}  # Store data for each series
        self.max_points = max_points
    
    def add_point(self, value: Any, timestamp: Optional[float] = None):
        """
        Add a data point
        
        Args:
            value: Data value, can be a single value or a dictionary (multiple series)
            timestamp: Timestamp, default is current time
        """
        if timestamp is None:
            timestamp = time.time()
        
        self.timestamps.append(timestamp)
        
        # Handle multi-series data
        if isinstance(value, dict):
            for series_name, series_value in value.items():
                if series_name not in self.series_data:
                    self.series_data[series_name] = []
                    
                # Fill historical missing values
                while len(self.series_data[series_name]) < len(self.timestamps) - 1:
                    self.series_data[series_name].append(None)
                    
                self.series_data[series_name].append(series_value)
        else:
            # Single value case, use "default" as default series name
            if "default" not in self.series_data:
                self.series_data["default"] = []
                
            # Fill historical missing values
            while len(self.series_data["default"]) < len(self.timestamps) - 1:
                self.series_data["default"].append(None)
                
            self.series_data["default"].append(value)
        
        # Remove earliest point when exceeding maximum points
        if len(self.timestamps) > self.max_points:
            self.timestamps.pop(0)
            for series in self.series_data.values():
                if series:
                    series.pop(0)
    
    def has_multiple_series(self) -> bool:
        """Check if contains multiple series"""
        return len(self.series_data) > 1 or (len(self.series_data) == 1 and "default" not in self.series_data)
    
    def get_series_names(self) -> List[str]:
        """Get all series names"""
        return list(self.series_data.keys())
    
    def get_echarts_data(self) -> Dict:
        """Get data format suitable for ECharts time series line chart (type: 'time')"""
        # formatted_times = [datetime.fromtimestamp(ts).strftime('%H:%M:%S') 
        #                    for ts in self.timestamps] # No longer needed for type: time
        
        # If no data, return empty structure
        if not self.timestamps:
            # Return structure expected by the new helper (just series list)
            return {"series": []} 
        
        series = []
        for series_name, values in self.series_data.items():
            # Ensure values length matches timestamps
            adjusted_values = values.copy()
            while len(adjusted_values) < len(self.timestamps):
                adjusted_values.append(None)
            
            # Format data as [[timestamp_ms, value], ...]
            # Multiply timestamp by 1000 for milliseconds
            time_value_pairs = []
            for i, ts in enumerate(self.timestamps):
                 val = adjusted_values[i]
                 # Echarts time axis can handle nulls for gaps
                 time_value_pairs.append([datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'), val])
                
            series.append({
                "name": series_name,
                "type": "line", # Type might be adjusted in the helper
                "data": time_value_pairs # Use the time-value pair format
            })
        
        # Return only the series list, as xAxis is implicit in the data for type: time
        return {
            # "xAxis": formatted_times, # Removed
            "series": series
        }
    
    def get_matplotlib_data(self) -> Dict:
        """Get data format suitable for matplotlib line chart"""
        formatted_times = [datetime.fromtimestamp(ts).strftime('%H:%M:%S') 
                           for ts in self.timestamps]
        
        # If no data, return empty structure
        if not self.timestamps:
            return {"xAxis": [], "series": {}}
        
        # Ensure all series lengths are consistent
        series_data = {}
        for series_name, values in self.series_data.items():
            adjusted_values = values.copy()
            while len(adjusted_values) < len(self.timestamps):
                adjusted_values.append(None)
            series_data[series_name] = adjusted_values
        
        return {
            "xAxis": formatted_times,
            "series": series_data
        }
    
    def get_last_n_points(self, n: int, format: str = "echarts") -> Dict:
        """
        Get last n data points
        """
        if n >= len(self.timestamps):
            return self.get_echarts_data() if format == "echarts" else self.get_matplotlib_data()
        
        sliced_times = self.timestamps[-n:]
        sliced_series = {}
        for series_name, values in self.series_data.items():
            # Handle series length less than n
            if len(values) < n:
                # Fill None values
                padding = [None] * (n - len(values))
                sliced_series[series_name] = padding + values
            else:
                sliced_series[series_name] = values[-n:]
        
        formatted_times = [datetime.fromtimestamp(ts).strftime('%H:%M:%S') 
                           for ts in sliced_times]
        
        if format == "echarts":
            series = []
            for series_name, values in sliced_series.items():
                series.append({
                    "name": series_name,
                    "type": "line",
                    "data": values
                })
            return {
                "xAxis": formatted_times,
                "series": series
            }
        else:
            return {
                "xAxis": formatted_times,
                "series": sliced_series
            }
            
    def clear(self):
        """Clear all data"""
        self.timestamps = []
        self.series_data = {}
    
    def merge(self, other: 'TimeSeriesMetricData'):
        """
        Merge another time series data
        """
        if not other.timestamps:
            return
            
        # Merge timestamps and data
        for ts_idx, ts in enumerate(other.timestamps):
            if ts not in self.timestamps:
                self.timestamps.append(ts)
                
                # Add None values for existing series
                for series in self.series_data.values():
                    series.append(None)
            
            ts_position = self.timestamps.index(ts)
            
            # Update data for each series
            for series_name, values in other.series_data.items():
                if series_name not in self.series_data:
                    self.series_data[series_name] = [None] * len(self.timestamps)
                
                if ts_idx < len(values):
                    self.series_data[series_name][ts_position] = values[ts_idx]
        
        # Sort timestamps and corresponding data
        sorted_data = sorted(zip(self.timestamps, range(len(self.timestamps))))
        self.timestamps = [item[0] for item in sorted_data]
        sort_indices = [item[1] for item in sorted_data]
        
        for series_name in self.series_data:
            self.series_data[series_name] = [self.series_data[series_name][i] for i in sort_indices]
            
        # Crop parts exceeding max_points
        while len(self.timestamps) > self.max_points:
            self.timestamps.pop(0)
            for series in self.series_data.values():
                series.pop(0)


class CategoryMetricData:
    """Handle category type metric data, for bar chart and pie chart"""
    
    def __init__(self):
        """Initialize category data storage"""
        self.categories: List[str] = []
        self.values: List[float] = []
        self.timestamp: float = time.time()
    
    def update_data(self, categories: List[str], values: List[float], timestamp: Optional[float] = None):
        """
        Update category data
        """
        if len(categories) != len(values):
            raise ValueError("Category list and value list lengths must be the same")
        
        self.categories = categories
        self.values = values
        self.timestamp = timestamp or time.time()
    
    def get_data(self, format: str = "echarts", viz_type: str = "bar") -> Dict:
        """
        Get data, support multiple formats and visualization types
        """
        if format == "matplotlib":
            return self.get_matplotlib_data(viz_type)
        else:
            if viz_type == "pie":
                return self.get_pie_chart_data()
            else:
                return self.get_bar_chart_data()
    
    def get_bar_chart_data(self) -> Dict:
        """Get data format suitable for ECharts bar chart"""
        return {
            "xAxis": self.categories,
            "series": self.values
        }
    
    def get_pie_chart_data(self) -> Dict:
        """Get data format suitable for ECharts pie chart"""
        series_data = [{"name": cat, "value": val} 
                      for cat, val in zip(self.categories, self.values)]
        
        return {
            "series": series_data
        }
        
    def get_matplotlib_data(self, viz_type: str = "bar") -> Dict:
        """
        Get data format suitable for matplotlib
        """
        if viz_type == "pie":
            return {
                "categories": self.categories,
                "values": self.values
            }
        else:  # bar
            return {
                "xAxis": self.categories,
                "series": self.values
            } 