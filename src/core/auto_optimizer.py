"""
Auto-Optimization Engine for OpenHands Enhanced
Continuously improves system performance, accuracy, and efficiency.
"""

import asyncio
import logging
import os
import time
import json
import psutil
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

@dataclass
class OptimizationMetrics:
    """Metrics for tracking optimization performance."""
    timestamp: float
    response_time: float
    success_rate: float
    cost_per_request: float
    cpu_usage: float
    memory_usage: float
    error_rate: float
    user_satisfaction: float = 0.0

@dataclass
class OptimizationResult:
    """Result of an optimization attempt."""
    optimization_id: str
    timestamp: float
    optimization_type: str
    target_area: str
    improvement_percentage: float
    success: bool
    rollback_required: bool = False
    error_message: Optional[str] = None

class AutoOptimizer:
    """Main auto-optimization engine."""
    
    def __init__(self):
        self.running = False
        self.optimization_task: Optional[asyncio.Task] = None
        
        # Configuration
        self.optimization_level = os.getenv("OPTIMIZATION_LEVEL", "balanced")
        self.optimization_frequency = int(os.getenv("OPTIMIZATION_FREQUENCY", "3600"))  # seconds
        self.rollback_sensitivity = os.getenv("ROLLBACK_SENSITIVITY", "medium")
        
        # Performance targets
        self.target_response_time = float(os.getenv("TARGET_RESPONSE_TIME", "2000"))  # ms
        self.target_throughput = int(os.getenv("TARGET_THROUGHPUT", "100"))  # req/min
        self.target_error_rate = float(os.getenv("TARGET_ERROR_RATE", "1"))  # percentage
        self.target_cost_reduction = float(os.getenv("TARGET_COST_REDUCTION", "40"))  # percentage
        
        # Data storage
        self.metrics_history = deque(maxlen=1000)
        self.optimization_history = deque(maxlen=100)
        self.provider_performance = defaultdict(lambda: deque(maxlen=100))
        self.model_performance = defaultdict(lambda: deque(maxlen=100))
        
        # Optimization state
        self.current_baseline = None
        self.optimization_in_progress = False
        self.last_optimization_time = 0
        
        # ML-based optimization
        self.performance_predictor = None
        self.cost_optimizer = None
        
    async def start(self):
        """Start the auto-optimization engine."""
        if self.running:
            return
            
        self.running = True
        self.optimization_task = asyncio.create_task(self._optimization_loop())
        logger.info("Auto-optimization engine started")
        
    async def stop(self):
        """Stop the auto-optimization engine."""
        self.running = False
        if self.optimization_task:
            self.optimization_task.cancel()
            try:
                await self.optimization_task
            except asyncio.CancelledError:
                pass
        logger.info("Auto-optimization engine stopped")
        
    def record_metrics(self, metrics: OptimizationMetrics):
        """Record performance metrics for analysis."""
        self.metrics_history.append(metrics)
        
        # Update baseline if this is the first measurement
        if self.current_baseline is None:
            self.current_baseline = metrics
            
    def record_provider_performance(self, provider: str, response_time: float, 
                                  success: bool, cost: float):
        """Record provider-specific performance metrics."""
        self.provider_performance[provider].append({
            'timestamp': time.time(),
            'response_time': response_time,
            'success': success,
            'cost': cost
        })
        
    def record_model_performance(self, model: str, accuracy: float, 
                               response_time: float, cost: float):
        """Record model-specific performance metrics."""
        self.model_performance[model].append({
            'timestamp': time.time(),
            'accuracy': accuracy,
            'response_time': response_time,
            'cost': cost
        })
        
    async def _optimization_loop(self):
        """Main optimization loop."""
        while self.running:
            try:
                if self._should_optimize():
                    await self._run_optimization_cycle()
                    
                # Wait for next optimization cycle
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
                
    def _should_optimize(self) -> bool:
        """Determine if optimization should run."""
        # Check if enough time has passed
        time_since_last = time.time() - self.last_optimization_time
        if time_since_last < self.optimization_frequency:
            return False
            
        # Check if optimization is already in progress
        if self.optimization_in_progress:
            return False
            
        # Check if we have enough data
        if len(self.metrics_history) < 10:
            return False
            
        # Check if performance has degraded
        if self._performance_degraded():
            return True
            
        # Regular optimization cycle
        return True
        
    def _performance_degraded(self) -> bool:
        """Check if performance has degraded significantly."""
        if len(self.metrics_history) < 10:
            return False
            
        recent_metrics = list(self.metrics_history)[-10:]
        older_metrics = list(self.metrics_history)[-20:-10] if len(self.metrics_history) >= 20 else []
        
        if not older_metrics:
            return False
            
        # Calculate average performance
        recent_avg_response = np.mean([m.response_time for m in recent_metrics])
        older_avg_response = np.mean([m.response_time for m in older_metrics])
        
        recent_avg_success = np.mean([m.success_rate for m in recent_metrics])
        older_avg_success = np.mean([m.success_rate for m in older_metrics])
        
        # Check for degradation
        response_degradation = (recent_avg_response - older_avg_response) / older_avg_response
        success_degradation = (older_avg_success - recent_avg_success) / older_avg_success
        
        return response_degradation > 0.2 or success_degradation > 0.1
        
    async def _run_optimization_cycle(self):
        """Run a complete optimization cycle."""
        self.optimization_in_progress = True
        self.last_optimization_time = time.time()
        
        try:
            logger.info("Starting optimization cycle")
            
            # Analyze current performance
            analysis = await self._analyze_performance()
            
            # Generate optimization strategies
            strategies = await self._generate_optimization_strategies(analysis)
            
            # Execute optimizations
            results = []
            for strategy in strategies:
                result = await self._execute_optimization(strategy)
                results.append(result)
                
                # Check if rollback is needed
                if result.rollback_required:
                    await self._rollback_optimization(result)
                    
            # Update optimization history
            for result in results:
                self.optimization_history.append(result)
                
            logger.info(f"Optimization cycle completed. {len(results)} optimizations applied.")
            
        except Exception as e:
            logger.error(f"Error in optimization cycle: {e}")
        finally:
            self.optimization_in_progress = False
            
    async def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze current system performance."""
        if not self.metrics_history:
            return {}
            
        recent_metrics = list(self.metrics_history)[-50:]  # Last 50 measurements
        
        analysis = {
            'avg_response_time': np.mean([m.response_time for m in recent_metrics]),
            'avg_success_rate': np.mean([m.success_rate for m in recent_metrics]),
            'avg_cost_per_request': np.mean([m.cost_per_request for m in recent_metrics]),
            'avg_cpu_usage': np.mean([m.cpu_usage for m in recent_metrics]),
            'avg_memory_usage': np.mean([m.memory_usage for m in recent_metrics]),
            'avg_error_rate': np.mean([m.error_rate for m in recent_metrics]),
            'response_time_trend': self._calculate_trend([m.response_time for m in recent_metrics]),
            'success_rate_trend': self._calculate_trend([m.success_rate for m in recent_metrics]),
            'cost_trend': self._calculate_trend([m.cost_per_request for m in recent_metrics]),
        }
        
        # Analyze provider performance
        analysis['provider_analysis'] = self._analyze_provider_performance()
        
        # Analyze model performance
        analysis['model_analysis'] = self._analyze_model_performance()
        
        # Identify bottlenecks
        analysis['bottlenecks'] = self._identify_bottlenecks(analysis)
        
        return analysis
        
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values."""
        if len(values) < 2:
            return "stable"
            
        # Simple linear regression
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.05:
            return "increasing"
        elif slope < -0.05:
            return "decreasing"
        else:
            return "stable"
            
    def _analyze_provider_performance(self) -> Dict[str, Any]:
        """Analyze performance of different providers."""
        provider_analysis = {}
        
        for provider, metrics in self.provider_performance.items():
            if not metrics:
                continue
                
            recent_metrics = list(metrics)[-20:]  # Last 20 requests
            
            avg_response_time = np.mean([m['response_time'] for m in recent_metrics])
            success_rate = np.mean([m['success'] for m in recent_metrics])
            avg_cost = np.mean([m['cost'] for m in recent_metrics])
            
            provider_analysis[provider] = {
                'avg_response_time': avg_response_time,
                'success_rate': success_rate,
                'avg_cost': avg_cost,
                'request_count': len(recent_metrics),
                'performance_score': self._calculate_performance_score(
                    avg_response_time, success_rate, avg_cost
                )
            }
            
        return provider_analysis
        
    def _analyze_model_performance(self) -> Dict[str, Any]:
        """Analyze performance of different models."""
        model_analysis = {}
        
        for model, metrics in self.model_performance.items():
            if not metrics:
                continue
                
            recent_metrics = list(metrics)[-20:]  # Last 20 requests
            
            avg_accuracy = np.mean([m['accuracy'] for m in recent_metrics])
            avg_response_time = np.mean([m['response_time'] for m in recent_metrics])
            avg_cost = np.mean([m['cost'] for m in recent_metrics])
            
            model_analysis[model] = {
                'avg_accuracy': avg_accuracy,
                'avg_response_time': avg_response_time,
                'avg_cost': avg_cost,
                'request_count': len(recent_metrics),
                'quality_score': self._calculate_quality_score(
                    avg_accuracy, avg_response_time, avg_cost
                )
            }
            
        return model_analysis
        
    def _calculate_performance_score(self, response_time: float, 
                                   success_rate: float, cost: float) -> float:
        """Calculate overall performance score for a provider."""
        # Normalize metrics (lower is better for response_time and cost)
        response_score = max(0, 1 - (response_time / 5000))  # 5s max
        success_score = success_rate
        cost_score = max(0, 1 - (cost / 0.01))  # $0.01 max per request
        
        # Weighted average
        return (response_score * 0.4 + success_score * 0.4 + cost_score * 0.2)
        
    def _calculate_quality_score(self, accuracy: float, response_time: float, 
                               cost: float) -> float:
        """Calculate overall quality score for a model."""
        # Normalize metrics
        accuracy_score = accuracy
        response_score = max(0, 1 - (response_time / 5000))  # 5s max
        cost_score = max(0, 1 - (cost / 0.01))  # $0.01 max per request
        
        # Weighted average (accuracy is most important)
        return (accuracy_score * 0.5 + response_score * 0.3 + cost_score * 0.2)
        
    def _identify_bottlenecks(self, analysis: Dict[str, Any]) -> List[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        # Response time bottleneck
        if analysis['avg_response_time'] > self.target_response_time:
            bottlenecks.append("response_time")
            
        # Error rate bottleneck
        if analysis['avg_error_rate'] > self.target_error_rate:
            bottlenecks.append("error_rate")
            
        # CPU bottleneck
        if analysis['avg_cpu_usage'] > 80:
            bottlenecks.append("cpu_usage")
            
        # Memory bottleneck
        if analysis['avg_memory_usage'] > 85:
            bottlenecks.append("memory_usage")
            
        # Cost bottleneck
        if analysis['cost_trend'] == "increasing":
            bottlenecks.append("cost_increase")
            
        return bottlenecks
        
    async def _generate_optimization_strategies(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate optimization strategies based on analysis."""
        strategies = []
        
        # Provider optimization
        if 'provider_analysis' in analysis:
            provider_strategy = self._generate_provider_optimization(analysis['provider_analysis'])
            if provider_strategy:
                strategies.append(provider_strategy)
                
        # Model optimization
        if 'model_analysis' in analysis:
            model_strategy = self._generate_model_optimization(analysis['model_analysis'])
            if model_strategy:
                strategies.append(model_strategy)
                
        # Resource optimization
        if 'cpu_usage' in analysis.get('bottlenecks', []):
            strategies.append({
                'type': 'resource_optimization',
                'target': 'cpu',
                'action': 'reduce_concurrent_requests',
                'parameters': {'reduction_percentage': 20}
            })
            
        # Cache optimization
        if analysis['avg_response_time'] > self.target_response_time:
            strategies.append({
                'type': 'cache_optimization',
                'target': 'response_time',
                'action': 'increase_cache_size',
                'parameters': {'size_increase_percentage': 25}
            })
            
        # Cost optimization
        if 'cost_increase' in analysis.get('bottlenecks', []):
            strategies.append({
                'type': 'cost_optimization',
                'target': 'cost',
                'action': 'optimize_provider_selection',
                'parameters': {'cost_weight_increase': 0.1}
            })
            
        return strategies
        
    def _generate_provider_optimization(self, provider_analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate provider optimization strategy."""
        if not provider_analysis:
            return None
            
        # Find best performing provider
        best_provider = max(provider_analysis.items(), 
                          key=lambda x: x[1]['performance_score'])
        
        # Find worst performing provider
        worst_provider = min(provider_analysis.items(), 
                           key=lambda x: x[1]['performance_score'])
        
        # If there's a significant difference, optimize routing
        score_diff = best_provider[1]['performance_score'] - worst_provider[1]['performance_score']
        
        if score_diff > 0.2:
            return {
                'type': 'provider_optimization',
                'target': 'routing',
                'action': 'adjust_provider_weights',
                'parameters': {
                    'increase_provider': best_provider[0],
                    'decrease_provider': worst_provider[0],
                    'weight_adjustment': min(0.3, score_diff)
                }
            }
            
        return None
        
    def _generate_model_optimization(self, model_analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate model optimization strategy."""
        if not model_analysis:
            return None
            
        # Find best performing model
        best_model = max(model_analysis.items(), 
                        key=lambda x: x[1]['quality_score'])
        
        # Find worst performing model
        worst_model = min(model_analysis.items(), 
                         key=lambda x: x[1]['quality_score'])
        
        # If there's a significant difference, optimize model selection
        score_diff = best_model[1]['quality_score'] - worst_model[1]['quality_score']
        
        if score_diff > 0.2:
            return {
                'type': 'model_optimization',
                'target': 'selection',
                'action': 'adjust_model_preferences',
                'parameters': {
                    'prefer_model': best_model[0],
                    'avoid_model': worst_model[0],
                    'preference_strength': min(0.3, score_diff)
                }
            }
            
        return None
        
    async def _execute_optimization(self, strategy: Dict[str, Any]) -> OptimizationResult:
        """Execute a specific optimization strategy."""
        optimization_id = f"opt_{int(time.time())}_{strategy['type']}"
        
        try:
            # Record baseline performance
            baseline = await self._measure_current_performance()
            
            # Apply optimization
            success = await self._apply_optimization(strategy)
            
            if not success:
                return OptimizationResult(
                    optimization_id=optimization_id,
                    timestamp=time.time(),
                    optimization_type=strategy['type'],
                    target_area=strategy['target'],
                    improvement_percentage=0.0,
                    success=False,
                    error_message="Failed to apply optimization"
                )
                
            # Wait for changes to take effect
            await asyncio.sleep(30)
            
            # Measure new performance
            new_performance = await self._measure_current_performance()
            
            # Calculate improvement
            improvement = self._calculate_improvement(baseline, new_performance, strategy['target'])
            
            # Check if rollback is needed
            rollback_required = self._should_rollback(improvement, strategy)
            
            return OptimizationResult(
                optimization_id=optimization_id,
                timestamp=time.time(),
                optimization_type=strategy['type'],
                target_area=strategy['target'],
                improvement_percentage=improvement,
                success=True,
                rollback_required=rollback_required
            )
            
        except Exception as e:
            logger.error(f"Error executing optimization {optimization_id}: {e}")
            return OptimizationResult(
                optimization_id=optimization_id,
                timestamp=time.time(),
                optimization_type=strategy['type'],
                target_area=strategy['target'],
                improvement_percentage=0.0,
                success=False,
                error_message=str(e)
            )
            
    async def _apply_optimization(self, strategy: Dict[str, Any]) -> bool:
        """Apply a specific optimization strategy."""
        try:
            if strategy['type'] == 'provider_optimization':
                return await self._apply_provider_optimization(strategy)
            elif strategy['type'] == 'model_optimization':
                return await self._apply_model_optimization(strategy)
            elif strategy['type'] == 'resource_optimization':
                return await self._apply_resource_optimization(strategy)
            elif strategy['type'] == 'cache_optimization':
                return await self._apply_cache_optimization(strategy)
            elif strategy['type'] == 'cost_optimization':
                return await self._apply_cost_optimization(strategy)
            else:
                logger.warning(f"Unknown optimization type: {strategy['type']}")
                return False
                
        except Exception as e:
            logger.error(f"Error applying optimization: {e}")
            return False
            
    async def _apply_provider_optimization(self, strategy: Dict[str, Any]) -> bool:
        """Apply provider optimization."""
        # This would integrate with the actual provider routing system
        # For now, simulate the optimization
        logger.info(f"Applying provider optimization: {strategy}")
        return True
        
    async def _apply_model_optimization(self, strategy: Dict[str, Any]) -> bool:
        """Apply model optimization."""
        # This would integrate with the actual model selection system
        # For now, simulate the optimization
        logger.info(f"Applying model optimization: {strategy}")
        return True
        
    async def _apply_resource_optimization(self, strategy: Dict[str, Any]) -> bool:
        """Apply resource optimization."""
        # This would integrate with the actual resource management system
        # For now, simulate the optimization
        logger.info(f"Applying resource optimization: {strategy}")
        return True
        
    async def _apply_cache_optimization(self, strategy: Dict[str, Any]) -> bool:
        """Apply cache optimization."""
        # This would integrate with the actual caching system
        # For now, simulate the optimization
        logger.info(f"Applying cache optimization: {strategy}")
        return True
        
    async def _apply_cost_optimization(self, strategy: Dict[str, Any]) -> bool:
        """Apply cost optimization."""
        # This would integrate with the actual cost management system
        # For now, simulate the optimization
        logger.info(f"Applying cost optimization: {strategy}")
        return True
        
    async def _measure_current_performance(self) -> Dict[str, float]:
        """Measure current system performance."""
        # In a real implementation, this would collect actual metrics
        # For now, simulate performance measurement
        return {
            'response_time': np.random.normal(2000, 500),  # ms
            'success_rate': np.random.normal(0.95, 0.05),
            'cost_per_request': np.random.normal(0.005, 0.002),
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'error_rate': np.random.normal(0.02, 0.01)
        }
        
    def _calculate_improvement(self, baseline: Dict[str, float], 
                             new_performance: Dict[str, float], 
                             target_area: str) -> float:
        """Calculate improvement percentage for the target area."""
        if target_area not in baseline or target_area not in new_performance:
            return 0.0
            
        baseline_value = baseline[target_area]
        new_value = new_performance[target_area]
        
        # For metrics where lower is better (response_time, cost, error_rate)
        if target_area in ['response_time', 'cost', 'error_rate', 'cpu_usage', 'memory_usage']:
            if baseline_value == 0:
                return 0.0
            improvement = (baseline_value - new_value) / baseline_value * 100
        else:
            # For metrics where higher is better (success_rate)
            if baseline_value == 0:
                return 0.0
            improvement = (new_value - baseline_value) / baseline_value * 100
            
        return improvement
        
    def _should_rollback(self, improvement: float, strategy: Dict[str, Any]) -> bool:
        """Determine if optimization should be rolled back."""
        # Rollback if performance degraded significantly
        if improvement < -10:  # 10% degradation
            return True
            
        # Rollback based on sensitivity setting
        if self.rollback_sensitivity == "high" and improvement < -5:
            return True
        elif self.rollback_sensitivity == "medium" and improvement < -15:
            return True
        elif self.rollback_sensitivity == "low" and improvement < -25:
            return True
            
        return False
        
    async def _rollback_optimization(self, result: OptimizationResult):
        """Rollback a failed optimization."""
        logger.warning(f"Rolling back optimization {result.optimization_id}")
        
        # In a real implementation, this would reverse the applied changes
        # For now, just log the rollback
        logger.info(f"Optimization {result.optimization_id} rolled back successfully")
        
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status."""
        return {
            'running': self.running,
            'optimization_in_progress': self.optimization_in_progress,
            'last_optimization_time': self.last_optimization_time,
            'optimization_level': self.optimization_level,
            'optimization_frequency': self.optimization_frequency,
            'metrics_count': len(self.metrics_history),
            'optimization_count': len(self.optimization_history),
            'performance_targets': {
                'response_time': self.target_response_time,
                'throughput': self.target_throughput,
                'error_rate': self.target_error_rate,
                'cost_reduction': self.target_cost_reduction
            }
        }
        
    def get_optimization_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent optimization history."""
        recent_optimizations = list(self.optimization_history)[-limit:]
        return [asdict(opt) for opt in recent_optimizations]
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.metrics_history:
            return {}
            
        recent_metrics = list(self.metrics_history)[-50:]
        
        return {
            'avg_response_time': np.mean([m.response_time for m in recent_metrics]),
            'avg_success_rate': np.mean([m.success_rate for m in recent_metrics]),
            'avg_cost_per_request': np.mean([m.cost_per_request for m in recent_metrics]),
            'avg_cpu_usage': np.mean([m.cpu_usage for m in recent_metrics]),
            'avg_memory_usage': np.mean([m.memory_usage for m in recent_metrics]),
            'avg_error_rate': np.mean([m.error_rate for m in recent_metrics]),
            'total_optimizations': len(self.optimization_history),
            'successful_optimizations': sum(1 for opt in self.optimization_history if opt.success),
            'avg_improvement': np.mean([opt.improvement_percentage for opt in self.optimization_history if opt.success])
        }

# Global auto-optimizer instance
auto_optimizer = AutoOptimizer()