"""
Metrics Module

Prometheus-compatible metrics for production monitoring.
Tracks request latency, prediction distribution, and model health.
"""

import logging
import time
from typing import Dict, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class MetricValue:
    """Single metric value with timestamp."""
    value: float
    timestamp: str
    labels: Dict[str, str] = field(default_factory=dict)


class Counter:
    """Simple counter metric."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self._value = 0
    
    def inc(self, value: float = 1.0):
        """Increment counter."""
        self._value += value
    
    def get(self) -> float:
        return self._value
    
    def reset(self):
        self._value = 0


class Gauge:
    """Simple gauge metric."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self._value = 0.0
    
    def set(self, value: float):
        """Set gauge value."""
        self._value = value
    
    def inc(self, value: float = 1.0):
        self._value += value
    
    def dec(self, value: float = 1.0):
        self._value -= value
    
    def get(self) -> float:
        return self._value


class Histogram:
    """Simple histogram metric for latency tracking."""
    
    DEFAULT_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    
    def __init__(self, name: str, description: str, buckets: list = None):
        self.name = name
        self.description = description
        self.buckets = buckets or self.DEFAULT_BUCKETS
        self._values = []
    
    def observe(self, value: float):
        """Record an observation."""
        self._values.append(value)
    
    def get_histogram(self) -> Dict:
        """Get histogram data."""
        if not self._values:
            return {'count': 0, 'sum': 0, 'buckets': {}}
        
        values = sorted(self._values)
        bucket_counts = {}
        for bucket in self.buckets:
            bucket_counts[bucket] = sum(1 for v in values if v <= bucket)
        
        return {
            'count': len(values),
            'sum': sum(values),
            'mean': sum(values) / len(values),
            'p50': values[len(values) // 2],
            'p95': values[int(len(values) * 0.95)] if len(values) > 1 else values[-1],
            'p99': values[int(len(values) * 0.99)] if len(values) > 1 else values[-1],
            'buckets': bucket_counts
        }


class MetricsCollector:
    """
    Central metrics collection and export.
    
    Tracks:
    - Request counts and latencies
    - Prediction distributions
    - Model health indicators
    - Error rates
    """
    
    def __init__(self):
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}
        
        # Initialize default metrics
        self._init_default_metrics()
    
    def _init_default_metrics(self):
        """Initialize standard metrics."""
        # Request metrics
        self.register_counter('requests_total', 'Total number of requests')
        self.register_counter('requests_success', 'Successful requests')
        self.register_counter('requests_error', 'Failed requests')
        
        # Latency metrics
        self.register_histogram('request_latency_seconds', 'Request latency in seconds')
        self.register_histogram('inference_latency_seconds', 'Model inference latency')
        
        # Prediction metrics
        self.register_histogram('sales_predictions', 'Sales prediction distribution')
        self.register_histogram('risk_predictions', 'Risk prediction distribution')
        
        # Model health
        self.register_gauge('model_loaded_sales', 'Sales model loaded status')
        self.register_gauge('model_loaded_risk', 'Risk model loaded status')
        self.register_gauge('last_prediction_timestamp', 'Timestamp of last prediction')
    
    def register_counter(self, name: str, description: str) -> Counter:
        """Register a new counter."""
        if name not in self._counters:
            self._counters[name] = Counter(name, description)
        return self._counters[name]
    
    def register_gauge(self, name: str, description: str) -> Gauge:
        """Register a new gauge."""
        if name not in self._gauges:
            self._gauges[name] = Gauge(name, description)
        return self._gauges[name]
    
    def register_histogram(self, name: str, description: str) -> Histogram:
        """Register a new histogram."""
        if name not in self._histograms:
            self._histograms[name] = Histogram(name, description)
        return self._histograms[name]
    
    def inc_counter(self, name: str, value: float = 1.0):
        """Increment a counter."""
        if name in self._counters:
            self._counters[name].inc(value)
    
    def set_gauge(self, name: str, value: float):
        """Set a gauge value."""
        if name in self._gauges:
            self._gauges[name].set(value)
    
    def observe_histogram(self, name: str, value: float):
        """Record histogram observation."""
        if name in self._histograms:
            self._histograms[name].observe(value)
    
    def record_request(self, model_type: str, latency: float, success: bool):
        """Record a prediction request."""
        self.inc_counter('requests_total')
        if success:
            self.inc_counter('requests_success')
        else:
            self.inc_counter('requests_error')
        
        self.observe_histogram('request_latency_seconds', latency)
        self.set_gauge('last_prediction_timestamp', time.time())
    
    def record_prediction(self, model_type: str, value: float):
        """Record a prediction value."""
        if model_type == 'sales':
            self.observe_histogram('sales_predictions', value)
        else:
            self.observe_histogram('risk_predictions', value)
    
    def get_all_metrics(self) -> Dict:
        """Export all metrics."""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'counters': {name: c.get() for name, c in self._counters.items()},
            'gauges': {name: g.get() for name, g in self._gauges.items()},
            'histograms': {name: h.get_histogram() for name, h in self._histograms.items()}
        }
        return metrics
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        for name, counter in self._counters.items():
            lines.append(f"# HELP {name} {counter.description}")
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {counter.get()}")
        
        for name, gauge in self._gauges.items():
            lines.append(f"# HELP {name} {gauge.description}")
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {gauge.get()}")
        
        return "\n".join(lines)


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def record_request_metrics(model_type: str, latency: float, success: bool):
    """Record request metrics to global collector."""
    collector = get_metrics_collector()
    collector.record_request(model_type, latency, success)


def record_prediction(model_type: str, value: float):
    """Record prediction to global collector."""
    collector = get_metrics_collector()
    collector.record_prediction(model_type, value)
