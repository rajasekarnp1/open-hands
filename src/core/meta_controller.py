"""
Meta-Model Controller for Intelligent Model Selection

Based on research from:
- FrugalGPT (arXiv:2305.05176)
- RouteLLM (arXiv:2406.18665)
- LLM-Blender (arXiv:2306.02561)
- Tree of Thoughts (arXiv:2305.10601)
"""

import json
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import re
import asyncio
from collections import defaultdict
import sqlite3
import pickle

from ..models import ChatCompletionRequest, ModelInfo, ModelCapability


@dataclass
class TaskComplexity:
    """Represents the complexity analysis of a user task."""
    reasoning_depth: float  # 0-1, how much reasoning is required
    domain_specificity: float  # 0-1, how domain-specific the task is
    context_length: int  # Required context length
    computational_intensity: float  # 0-1, computational requirements
    creativity_required: float  # 0-1, how much creativity is needed
    factual_accuracy_importance: float  # 0-1, importance of factual accuracy
    
    @property
    def overall_complexity(self) -> float:
        """Calculate overall complexity score."""
        weights = {
            'reasoning': 0.3,
            'domain': 0.2,
            'context': 0.15,
            'computation': 0.15,
            'creativity': 0.1,
            'accuracy': 0.1
        }
        
        normalized_context = min(self.context_length / 32000, 1.0)
        
        return (
            weights['reasoning'] * self.reasoning_depth +
            weights['domain'] * self.domain_specificity +
            weights['context'] * normalized_context +
            weights['computation'] * self.computational_intensity +
            weights['creativity'] * self.creativity_required +
            weights['accuracy'] * self.factual_accuracy_importance
        )


@dataclass
class ModelCapabilityProfile:
    """Detailed capability profile for a model."""
    model_name: str
    provider: str
    size_category: str  # "small", "medium", "large"
    
    # Capability scores (0-1)
    reasoning_ability: float
    code_generation: float
    mathematical_reasoning: float
    creative_writing: float
    factual_knowledge: float
    instruction_following: float
    context_handling: float
    
    # Performance metrics
    avg_response_time: float
    reliability_score: float
    cost_per_token: float
    max_context_length: int
    
    # Specializations
    domain_expertise: List[str]
    preferred_task_types: List[str]
    
    def compatibility_score(self, task_complexity: TaskComplexity) -> float:
        """Calculate how well this model matches the task complexity."""
        
        # Base compatibility based on capability alignment
        capability_match = (
            self.reasoning_ability * task_complexity.reasoning_depth +
            self.factual_knowledge * task_complexity.factual_accuracy_importance +
            self.creative_writing * task_complexity.creativity_required +
            self.context_handling * min(task_complexity.context_length / self.max_context_length, 1.0)
        ) / 4.0
        
        # Adjust for computational requirements
        if task_complexity.computational_intensity > 0.7 and self.size_category == "small":
            capability_match *= 0.7
        elif task_complexity.computational_intensity < 0.3 and self.size_category == "large":
            capability_match *= 0.9  # Slight penalty for overkill
        
        return min(capability_match, 1.0)


class ExternalMemorySystem:
    """External memory system for storing model performance data and task patterns."""
    
    def __init__(self, db_path: str = "model_memory.db"):
        self.db_path = db_path
        self.init_database()
        self.performance_cache = {}
        self.task_patterns = defaultdict(list)
    
    def init_database(self):
        """Initialize the SQLite database for persistent storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Model performance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                provider TEXT,
                task_type TEXT,
                complexity_score REAL,
                success_rate REAL,
                avg_response_time REAL,
                user_satisfaction REAL,
                timestamp DATETIME,
                task_hash TEXT
            )
        """)
        
        # Task patterns table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS task_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_hash TEXT,
                task_type TEXT,
                complexity_features TEXT,
                optimal_model TEXT,
                confidence_score REAL,
                timestamp DATETIME
            )
        """)
        
        # User preferences table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT,
                preference_type TEXT,
                preference_value TEXT,
                weight REAL,
                timestamp DATETIME
            )
        """)
        
        conn.commit()
        conn.close()
    
    def store_performance_data(self, model_name: str, provider: str, task_type: str, 
                             complexity_score: float, success_rate: float, 
                             response_time: float, user_satisfaction: float, task_hash: str):
        """Store performance data for a model on a specific task."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO model_performance 
            (model_name, provider, task_type, complexity_score, success_rate, 
             avg_response_time, user_satisfaction, timestamp, task_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (model_name, provider, task_type, complexity_score, success_rate,
              response_time, user_satisfaction, datetime.utcnow(), task_hash))
        
        conn.commit()
        conn.close()
    
    def get_model_performance_history(self, model_name: str, task_type: str = None, 
                                    days: int = 30) -> List[Dict]:
        """Retrieve performance history for a model."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        since_date = datetime.utcnow() - timedelta(days=days)
        
        if task_type:
            cursor.execute("""
                SELECT * FROM model_performance 
                WHERE model_name = ? AND task_type = ? AND timestamp > ?
                ORDER BY timestamp DESC
            """, (model_name, task_type, since_date))
        else:
            cursor.execute("""
                SELECT * FROM model_performance 
                WHERE model_name = ? AND timestamp > ?
                ORDER BY timestamp DESC
            """, (model_name, since_date))
        
        results = cursor.fetchall()
        conn.close()
        
        return [dict(zip([col[0] for col in cursor.description], row)) for row in results]
    
    def store_task_pattern(self, task_hash: str, task_type: str, complexity_features: Dict,
                          optimal_model: str, confidence_score: float):
        """Store a learned task pattern."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO task_patterns 
            (task_hash, task_type, complexity_features, optimal_model, confidence_score, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (task_hash, task_type, json.dumps(complexity_features), optimal_model, 
              confidence_score, datetime.utcnow()))
        
        conn.commit()
        conn.close()
    
    def find_similar_tasks(self, task_hash: str, task_type: str, limit: int = 5) -> List[Dict]:
        """Find similar tasks from memory."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM task_patterns 
            WHERE task_type = ? 
            ORDER BY confidence_score DESC 
            LIMIT ?
        """, (task_type, limit))
        
        results = cursor.fetchall()
        conn.close()
        
        return [dict(zip([col[0] for col in cursor.description], row)) for row in results]


class TaskComplexityAnalyzer:
    """Analyzes task complexity using lightweight NLP and pattern matching."""
    
    def __init__(self):
        self.reasoning_keywords = [
            'think', 'analyze', 'reason', 'solve', 'explain', 'why', 'how', 'because',
            'step by step', 'logical', 'deduce', 'infer', 'conclude', 'therefore'
        ]
        
        self.domain_patterns = {
            'code': ['code', 'program', 'function', 'algorithm', 'debug', 'python', 'javascript'],
            'math': ['calculate', 'equation', 'formula', 'solve', 'mathematics', 'algebra'],
            'creative': ['story', 'poem', 'creative', 'imagine', 'write', 'compose'],
            'factual': ['fact', 'information', 'data', 'research', 'knowledge', 'definition'],
            'reasoning': ['logic', 'reasoning', 'problem', 'puzzle', 'think', 'analyze']
        }
        
        self.complexity_indicators = {
            'high': ['complex', 'difficult', 'advanced', 'sophisticated', 'intricate'],
            'medium': ['moderate', 'intermediate', 'standard', 'typical'],
            'low': ['simple', 'basic', 'easy', 'straightforward', 'quick']
        }
    
    def analyze_task_complexity(self, request: ChatCompletionRequest) -> TaskComplexity:
        """Analyze the complexity of a chat completion request."""
        
        # Combine all message content
        full_text = " ".join([msg.content for msg in request.messages if msg.content])
        text_lower = full_text.lower()
        
        # Analyze reasoning depth
        reasoning_score = self._calculate_reasoning_score(text_lower)
        
        # Analyze domain specificity
        domain_score = self._calculate_domain_specificity(text_lower)
        
        # Estimate context requirements
        context_length = len(full_text.split())
        
        # Analyze computational intensity
        computational_score = self._calculate_computational_intensity(text_lower)
        
        # Analyze creativity requirements
        creativity_score = self._calculate_creativity_score(text_lower)
        
        # Analyze factual accuracy importance
        accuracy_score = self._calculate_accuracy_importance(text_lower)
        
        return TaskComplexity(
            reasoning_depth=reasoning_score,
            domain_specificity=domain_score,
            context_length=context_length,
            computational_intensity=computational_score,
            creativity_required=creativity_score,
            factual_accuracy_importance=accuracy_score
        )
    
    def _calculate_reasoning_score(self, text: str) -> float:
        """Calculate reasoning complexity score."""
        reasoning_count = sum(1 for keyword in self.reasoning_keywords if keyword in text)
        
        # Look for multi-step reasoning patterns
        step_patterns = ['step 1', 'first', 'then', 'next', 'finally', 'step by step']
        step_count = sum(1 for pattern in step_patterns if pattern in text)
        
        # Question complexity
        question_marks = text.count('?')
        
        score = min((reasoning_count * 0.1 + step_count * 0.2 + question_marks * 0.1), 1.0)
        return score
    
    def _calculate_domain_specificity(self, text: str) -> float:
        """Calculate domain specificity score."""
        domain_scores = {}
        
        for domain, keywords in self.domain_patterns.items():
            score = sum(1 for keyword in keywords if keyword in text)
            domain_scores[domain] = score
        
        max_score = max(domain_scores.values()) if domain_scores.values() else 0
        return min(max_score * 0.15, 1.0)
    
    def _calculate_computational_intensity(self, text: str) -> float:
        """Calculate computational intensity score."""
        high_intensity_keywords = [
            'complex', 'large', 'detailed', 'comprehensive', 'extensive',
            'analyze', 'process', 'calculate', 'generate', 'optimize'
        ]
        
        intensity_count = sum(1 for keyword in high_intensity_keywords if keyword in text)
        
        # Text length as indicator
        length_score = min(len(text.split()) / 1000, 1.0)
        
        return min(intensity_count * 0.1 + length_score * 0.3, 1.0)
    
    def _calculate_creativity_score(self, text: str) -> float:
        """Calculate creativity requirements score."""
        creative_keywords = [
            'creative', 'imagine', 'story', 'poem', 'write', 'compose',
            'invent', 'design', 'artistic', 'original', 'unique'
        ]
        
        creative_count = sum(1 for keyword in creative_keywords if keyword in text)
        return min(creative_count * 0.2, 1.0)
    
    def _calculate_accuracy_importance(self, text: str) -> float:
        """Calculate factual accuracy importance score."""
        accuracy_keywords = [
            'fact', 'accurate', 'correct', 'precise', 'exact', 'true',
            'research', 'data', 'information', 'knowledge', 'definition'
        ]
        
        accuracy_count = sum(1 for keyword in accuracy_keywords if keyword in text)
        return min(accuracy_count * 0.15, 1.0)


class FrugalCascadeRouter:
    """
    Implements FrugalGPT-style cascade routing.
    Starts with smaller models and escalates based on confidence and complexity.
    """
    
    def __init__(self, model_profiles: Dict[str, ModelCapabilityProfile]):
        self.model_profiles = model_profiles
        self.confidence_threshold = 0.8
        self.complexity_thresholds = {
            'small': 0.3,
            'medium': 0.6,
            'large': 1.0
        }
    
    def get_cascade_chain(self, task_complexity: TaskComplexity) -> List[str]:
        """Get the cascade chain of models to try."""
        complexity_score = task_complexity.overall_complexity
        
        # Sort models by size and capability
        small_models = [name for name, profile in self.model_profiles.items() 
                       if profile.size_category == "small"]
        medium_models = [name for name, profile in self.model_profiles.items() 
                        if profile.size_category == "medium"]
        large_models = [name for name, profile in self.model_profiles.items() 
                       if profile.size_category == "large"]
        
        cascade_chain = []
        
        # Always start with a small model for simple tasks
        if complexity_score <= self.complexity_thresholds['small'] and small_models:
            cascade_chain.extend(small_models[:2])
        
        # Add medium models for moderate complexity
        if complexity_score <= self.complexity_thresholds['medium'] and medium_models:
            cascade_chain.extend(medium_models[:2])
        
        # Add large models for high complexity
        if complexity_score > self.complexity_thresholds['medium'] and large_models:
            cascade_chain.extend(large_models[:2])
        
        # Fallback: add all available models
        if not cascade_chain:
            cascade_chain = list(self.model_profiles.keys())
        
        return cascade_chain
    
    def should_escalate(self, response_confidence: float, task_complexity: TaskComplexity) -> bool:
        """Determine if we should escalate to the next model in the cascade."""
        
        # Lower confidence threshold for more complex tasks
        adjusted_threshold = self.confidence_threshold - (task_complexity.overall_complexity * 0.2)
        
        return response_confidence < adjusted_threshold


class MetaModelController:
    """
    Main meta-model controller that orchestrates intelligent model selection.
    Uses a small language model with external memory for decision making.
    """
    
    def __init__(self, model_profiles: Dict[str, ModelCapabilityProfile]):
        self.model_profiles = model_profiles
        self.memory_system = ExternalMemorySystem()
        self.complexity_analyzer = TaskComplexityAnalyzer()
        self.cascade_router = FrugalCascadeRouter(model_profiles)
        
        # Performance tracking
        self.performance_history = defaultdict(list)
        self.user_preferences = defaultdict(dict)
        
        # Learning parameters
        self.learning_rate = 0.1
        self.exploration_rate = 0.1
    
    async def select_optimal_model(self, request: ChatCompletionRequest, 
                                 user_id: Optional[str] = None) -> Tuple[str, float]:
        """
        Select the optimal model for a given request.
        Returns (model_name, confidence_score).
        """
        
        # Analyze task complexity
        task_complexity = self.complexity_analyzer.analyze_task_complexity(request)
        
        # Generate task hash for memory lookup
        task_hash = self._generate_task_hash(request)
        
        # Check memory for similar tasks
        similar_tasks = self.memory_system.find_similar_tasks(
            task_hash, self._classify_task_type(request)
        )
        
        # Get candidate models
        candidate_models = self._get_candidate_models(task_complexity, similar_tasks)
        
        # Score each candidate
        model_scores = {}
        for model_name in candidate_models:
            if model_name in self.model_profiles:
                profile = self.model_profiles[model_name]
                
                # Base compatibility score
                compatibility = profile.compatibility_score(task_complexity)
                
                # Historical performance adjustment
                historical_score = self._get_historical_performance(model_name, task_complexity)
                
                # User preference adjustment
                preference_score = self._get_user_preference_score(model_name, user_id)
                
                # Cost-benefit analysis
                cost_benefit = self._calculate_cost_benefit(profile, task_complexity)
                
                # Combined score
                final_score = (
                    compatibility * 0.4 +
                    historical_score * 0.3 +
                    preference_score * 0.2 +
                    cost_benefit * 0.1
                )
                
                model_scores[model_name] = final_score
        
        # Select best model
        if model_scores:
            best_model = max(model_scores.items(), key=lambda x: x[1])
            return best_model[0], best_model[1]
        
        # Fallback to cascade routing
        cascade_chain = self.cascade_router.get_cascade_chain(task_complexity)
        return cascade_chain[0] if cascade_chain else "auto", 0.5
    
    async def get_cascade_chain(self, request: ChatCompletionRequest) -> List[str]:
        """Get the full cascade chain for a request."""
        task_complexity = self.complexity_analyzer.analyze_task_complexity(request)
        return self.cascade_router.get_cascade_chain(task_complexity)
    
    async def update_performance_feedback(self, model_name: str, request: ChatCompletionRequest,
                                        success: bool, response_time: float, 
                                        user_satisfaction: Optional[float] = None):
        """Update performance feedback for continuous learning."""
        
        task_complexity = self.complexity_analyzer.analyze_task_complexity(request)
        task_type = self._classify_task_type(request)
        task_hash = self._generate_task_hash(request)
        
        # Store performance data
        self.memory_system.store_performance_data(
            model_name=model_name,
            provider=self.model_profiles[model_name].provider if model_name in self.model_profiles else "unknown",
            task_type=task_type,
            complexity_score=task_complexity.overall_complexity,
            success_rate=1.0 if success else 0.0,
            response_time=response_time,
            user_satisfaction=user_satisfaction or (0.8 if success else 0.2),
            task_hash=task_hash
        )
        
        # Update model profile if needed
        if model_name in self.model_profiles:
            self._update_model_profile(model_name, task_complexity, success, response_time)
    
    def _get_candidate_models(self, task_complexity: TaskComplexity, 
                            similar_tasks: List[Dict]) -> List[str]:
        """Get candidate models based on task complexity and similar tasks."""
        
        candidates = set()
        
        # Add models from similar tasks
        for task in similar_tasks:
            candidates.add(task['optimal_model'])
        
        # Add models based on complexity
        for model_name, profile in self.model_profiles.items():
            compatibility = profile.compatibility_score(task_complexity)
            if compatibility > 0.5:  # Threshold for consideration
                candidates.add(model_name)
        
        return list(candidates)
    
    def _get_historical_performance(self, model_name: str, task_complexity: TaskComplexity) -> float:
        """Get historical performance score for a model on similar tasks."""
        
        task_type = "general"  # Simplified for now
        history = self.memory_system.get_model_performance_history(model_name, task_type)
        
        if not history:
            return 0.5  # Neutral score for unknown models
        
        # Calculate weighted average of recent performance
        total_weight = 0
        weighted_score = 0
        
        for record in history:
            # Weight more recent records higher
            age_days = (datetime.utcnow() - datetime.fromisoformat(record['timestamp'])).days
            weight = max(0.1, 1.0 - (age_days / 30))  # Decay over 30 days
            
            # Combine success rate and user satisfaction
            performance = (record['success_rate'] + record['user_satisfaction']) / 2
            
            weighted_score += performance * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.5
    
    def _get_user_preference_score(self, model_name: str, user_id: Optional[str]) -> float:
        """Get user preference score for a model."""
        if not user_id or user_id not in self.user_preferences:
            return 0.5  # Neutral score
        
        preferences = self.user_preferences[user_id]
        return preferences.get(model_name, 0.5)
    
    def _calculate_cost_benefit(self, profile: ModelCapabilityProfile, 
                              task_complexity: TaskComplexity) -> float:
        """Calculate cost-benefit score for a model."""
        
        # Higher score for better cost-performance ratio
        performance_estimate = profile.compatibility_score(task_complexity)
        cost_factor = 1.0 - min(profile.cost_per_token * 1000, 1.0)  # Normalize cost
        
        return (performance_estimate + cost_factor) / 2
    
    def _classify_task_type(self, request: ChatCompletionRequest) -> str:
        """Classify the type of task based on the request."""
        
        full_text = " ".join([msg.content for msg in request.messages if msg.content]).lower()
        
        # Simple keyword-based classification
        if any(keyword in full_text for keyword in ['code', 'program', 'function']):
            return 'code'
        elif any(keyword in full_text for keyword in ['math', 'calculate', 'equation']):
            return 'math'
        elif any(keyword in full_text for keyword in ['story', 'creative', 'write']):
            return 'creative'
        elif any(keyword in full_text for keyword in ['analyze', 'reason', 'solve']):
            return 'reasoning'
        else:
            return 'general'
    
    def _generate_task_hash(self, request: ChatCompletionRequest) -> str:
        """Generate a hash for the task to identify similar tasks."""
        
        # Simplified hash based on message content and type
        content = " ".join([msg.content for msg in request.messages if msg.content])
        task_type = self._classify_task_type(request)
        
        # Create a simple hash
        import hashlib
        hash_input = f"{task_type}:{content[:200]}"  # First 200 chars
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _update_model_profile(self, model_name: str, task_complexity: TaskComplexity,
                            success: bool, response_time: float):
        """Update model profile based on performance feedback."""
        
        if model_name not in self.model_profiles:
            return
        
        profile = self.model_profiles[model_name]
        
        # Update reliability score
        current_reliability = profile.reliability_score
        new_reliability = current_reliability + self.learning_rate * (
            (1.0 if success else 0.0) - current_reliability
        )
        profile.reliability_score = max(0.0, min(1.0, new_reliability))
        
        # Update average response time
        current_time = profile.avg_response_time
        profile.avg_response_time = current_time + self.learning_rate * (response_time - current_time)
    
    def get_model_insights(self) -> Dict[str, Any]:
        """Get insights about model performance and usage patterns."""
        
        insights = {
            'model_performance': {},
            'task_patterns': {},
            'recommendations': []
        }
        
        # Analyze model performance
        for model_name, profile in self.model_profiles.items():
            insights['model_performance'][model_name] = {
                'reliability_score': profile.reliability_score,
                'avg_response_time': profile.avg_response_time,
                'size_category': profile.size_category,
                'cost_per_token': profile.cost_per_token
            }
        
        # Generate recommendations
        insights['recommendations'] = self._generate_recommendations()
        
        return insights
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for improving model selection."""
        
        recommendations = []
        
        # Analyze model usage patterns
        high_performers = [
            name for name, profile in self.model_profiles.items()
            if profile.reliability_score > 0.8
        ]
        
        if high_performers:
            recommendations.append(
                f"Consider prioritizing high-performing models: {', '.join(high_performers[:3])}"
            )
        
        # Check for underutilized models
        fast_models = [
            name for name, profile in self.model_profiles.items()
            if profile.avg_response_time < 2.0 and profile.size_category == "small"
        ]
        
        if fast_models:
            recommendations.append(
                f"For simple tasks, consider using fast models: {', '.join(fast_models[:2])}"
            )
        
        return recommendations