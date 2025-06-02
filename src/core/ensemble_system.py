"""
Ensemble System for Multi-Model Response Fusion

Based on research from:
- LLM-Blender (arXiv:2306.02561)
- Mixture of Experts (arXiv:2305.14705)
- Constitutional AI (arXiv:2212.08073)
"""

import asyncio
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json
import re
from collections import defaultdict

from ..models import ChatCompletionRequest, ChatCompletionResponse, ChatMessage


@dataclass
class ResponseCandidate:
    """Represents a candidate response from a model."""
    model_name: str
    provider: str
    response: ChatCompletionResponse
    confidence_score: float
    response_time: float
    cost_estimate: float
    
    # Quality metrics
    coherence_score: float = 0.0
    relevance_score: float = 0.0
    factual_accuracy_score: float = 0.0
    creativity_score: float = 0.0
    safety_score: float = 0.0


class ResponseQualityEvaluator:
    """Evaluates the quality of model responses using multiple metrics."""
    
    def __init__(self):
        self.coherence_patterns = [
            r'\b(however|therefore|moreover|furthermore|consequently)\b',
            r'\b(first|second|third|finally)\b',
            r'\b(in conclusion|to summarize|overall)\b'
        ]
        
        self.factual_indicators = [
            r'\b(according to|research shows|studies indicate)\b',
            r'\b(fact|data|evidence|statistics)\b',
            r'\b(\d{4}|\d+%|\d+\.\d+)\b'  # Numbers, years, percentages
        ]
        
        self.creativity_indicators = [
            r'\b(imagine|creative|unique|innovative|original)\b',
            r'\b(story|poem|metaphor|analogy)\b',
            r'[!]{2,}|[?]{2,}'  # Multiple punctuation marks
        ]
        
        self.safety_concerns = [
            r'\b(harmful|dangerous|illegal|unethical)\b',
            r'\b(violence|hate|discrimination)\b',
            r'\b(sorry|cannot|unable|inappropriate)\b'
        ]
    
    def evaluate_response(self, response: ChatCompletionResponse, 
                         original_request: ChatCompletionRequest) -> Dict[str, float]:
        """Evaluate a response across multiple quality dimensions."""
        
        if not response.choices or not response.choices[0].get('message', {}).get('content'):
            return {
                'coherence_score': 0.0,
                'relevance_score': 0.0,
                'factual_accuracy_score': 0.0,
                'creativity_score': 0.0,
                'safety_score': 0.0,
                'overall_quality': 0.0
            }
        
        content = response.choices[0]['message']['content']
        
        # Evaluate coherence
        coherence_score = self._evaluate_coherence(content)
        
        # Evaluate relevance to the original request
        relevance_score = self._evaluate_relevance(content, original_request)
        
        # Evaluate factual accuracy indicators
        factual_score = self._evaluate_factual_indicators(content)
        
        # Evaluate creativity
        creativity_score = self._evaluate_creativity(content)
        
        # Evaluate safety
        safety_score = self._evaluate_safety(content)
        
        # Calculate overall quality
        overall_quality = (
            coherence_score * 0.25 +
            relevance_score * 0.30 +
            factual_score * 0.20 +
            creativity_score * 0.15 +
            safety_score * 0.10
        )
        
        return {
            'coherence_score': coherence_score,
            'relevance_score': relevance_score,
            'factual_accuracy_score': factual_score,
            'creativity_score': creativity_score,
            'safety_score': safety_score,
            'overall_quality': overall_quality
        }
    
    def _evaluate_coherence(self, content: str) -> float:
        """Evaluate the coherence of the response."""
        
        # Check for logical connectors
        connector_count = sum(
            len(re.findall(pattern, content, re.IGNORECASE))
            for pattern in self.coherence_patterns
        )
        
        # Check sentence structure
        sentences = content.split('.')
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        
        # Penalize very short or very long sentences
        length_score = 1.0 - abs(avg_sentence_length - 15) / 30
        length_score = max(0.0, min(1.0, length_score))
        
        # Check for repetition
        words = content.lower().split()
        unique_words = len(set(words))
        repetition_score = unique_words / len(words) if words else 0
        
        # Combine scores
        coherence_score = (
            min(connector_count * 0.1, 0.4) +  # Logical connectors
            length_score * 0.3 +               # Sentence length
            repetition_score * 0.3             # Word diversity
        )
        
        return min(coherence_score, 1.0)
    
    def _evaluate_relevance(self, content: str, request: ChatCompletionRequest) -> float:
        """Evaluate how relevant the response is to the original request."""
        
        # Extract keywords from the request
        request_text = " ".join([msg.content for msg in request.messages if msg.content])
        request_keywords = set(re.findall(r'\b\w+\b', request_text.lower()))
        
        # Extract keywords from the response
        response_keywords = set(re.findall(r'\b\w+\b', content.lower()))
        
        # Calculate keyword overlap
        if not request_keywords:
            return 0.5  # Neutral score if no keywords
        
        overlap = len(request_keywords.intersection(response_keywords))
        relevance_score = overlap / len(request_keywords)
        
        # Boost score if response directly addresses the question
        if any(word in content.lower() for word in ['answer', 'solution', 'result']):
            relevance_score += 0.1
        
        return min(relevance_score, 1.0)
    
    def _evaluate_factual_indicators(self, content: str) -> float:
        """Evaluate indicators of factual accuracy."""
        
        # Count factual indicators
        factual_count = sum(
            len(re.findall(pattern, content, re.IGNORECASE))
            for pattern in self.factual_indicators
        )
        
        # Check for hedging language (indicates uncertainty)
        hedging_patterns = [r'\b(might|could|possibly|perhaps|maybe)\b']
        hedging_count = sum(
            len(re.findall(pattern, content, re.IGNORECASE))
            for pattern in hedging_patterns
        )
        
        # Check for confident assertions
        confident_patterns = [r'\b(definitely|certainly|clearly|obviously)\b']
        confident_count = sum(
            len(re.findall(pattern, content, re.IGNORECASE))
            for pattern in confident_patterns
        )
        
        # Calculate score
        factual_score = (
            min(factual_count * 0.1, 0.5) +     # Factual indicators
            min(confident_count * 0.1, 0.3) -   # Confident language
            min(hedging_count * 0.05, 0.2)      # Penalty for hedging
        )
        
        return max(0.0, min(factual_score + 0.4, 1.0))  # Base score of 0.4
    
    def _evaluate_creativity(self, content: str) -> float:
        """Evaluate the creativity of the response."""
        
        # Count creativity indicators
        creativity_count = sum(
            len(re.findall(pattern, content, re.IGNORECASE))
            for pattern in self.creativity_indicators
        )
        
        # Check for varied sentence structures
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        sentence_starts = [s.split()[0].lower() if s.split() else '' for s in sentences]
        start_diversity = len(set(sentence_starts)) / len(sentence_starts) if sentence_starts else 0
        
        # Check for descriptive language
        adjectives = len(re.findall(r'\b\w+ly\b|\b\w+ing\b|\b\w+ed\b', content))
        descriptive_score = min(adjectives / 50, 0.3)
        
        creativity_score = (
            min(creativity_count * 0.1, 0.4) +  # Creativity indicators
            start_diversity * 0.3 +             # Sentence diversity
            descriptive_score                   # Descriptive language
        )
        
        return min(creativity_score, 1.0)
    
    def _evaluate_safety(self, content: str) -> float:
        """Evaluate the safety of the response."""
        
        # Count safety concerns
        concern_count = sum(
            len(re.findall(pattern, content, re.IGNORECASE))
            for pattern in self.safety_concerns
        )
        
        # Check for refusal patterns (good for safety)
        refusal_patterns = [r'\b(cannot|unable|inappropriate|sorry)\b']
        refusal_count = sum(
            len(re.findall(pattern, content, re.IGNORECASE))
            for pattern in refusal_patterns
        )
        
        # Base safety score
        safety_score = 0.8
        
        # Penalty for concerning content
        safety_score -= min(concern_count * 0.2, 0.6)
        
        # Bonus for appropriate refusals (context-dependent)
        if refusal_count > 0 and concern_count > 0:
            safety_score += 0.2
        
        return max(0.0, min(safety_score, 1.0))


class PairwiseRanker:
    """Implements pairwise ranking for response comparison."""
    
    def __init__(self):
        self.comparison_criteria = [
            'overall_quality',
            'relevance_score',
            'coherence_score',
            'factual_accuracy_score'
        ]
    
    def rank_responses(self, candidates: List[ResponseCandidate]) -> List[ResponseCandidate]:
        """Rank responses using pairwise comparison."""
        
        if len(candidates) <= 1:
            return candidates
        
        # Create pairwise comparison matrix
        n = len(candidates)
        comparison_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    comparison_matrix[i][j] = self._compare_responses(candidates[i], candidates[j])
        
        # Calculate ranking scores
        ranking_scores = np.sum(comparison_matrix, axis=1)
        
        # Sort candidates by ranking scores
        ranked_indices = np.argsort(ranking_scores)[::-1]
        return [candidates[i] for i in ranked_indices]
    
    def _compare_responses(self, candidate_a: ResponseCandidate, 
                          candidate_b: ResponseCandidate) -> float:
        """Compare two response candidates. Returns 1 if A is better, 0 if B is better."""
        
        score_a = 0
        score_b = 0
        
        # Compare quality metrics
        if candidate_a.coherence_score > candidate_b.coherence_score:
            score_a += 1
        else:
            score_b += 1
        
        if candidate_a.relevance_score > candidate_b.relevance_score:
            score_a += 1
        else:
            score_b += 1
        
        if candidate_a.factual_accuracy_score > candidate_b.factual_accuracy_score:
            score_a += 1
        else:
            score_b += 1
        
        # Consider confidence and response time
        if candidate_a.confidence_score > candidate_b.confidence_score:
            score_a += 0.5
        else:
            score_b += 0.5
        
        if candidate_a.response_time < candidate_b.response_time:
            score_a += 0.5
        else:
            score_b += 0.5
        
        return 1.0 if score_a > score_b else 0.0


class ResponseFuser:
    """Fuses multiple responses into a single, improved response."""
    
    def __init__(self):
        self.fusion_strategies = {
            'best_response': self._select_best_response,
            'consensus_fusion': self._consensus_fusion,
            'weighted_fusion': self._weighted_fusion,
            'extractive_fusion': self._extractive_fusion
        }
    
    def fuse_responses(self, candidates: List[ResponseCandidate], 
                      strategy: str = 'weighted_fusion') -> ChatCompletionResponse:
        """Fuse multiple response candidates into a single response."""
        
        if not candidates:
            raise ValueError("No candidates provided for fusion")
        
        if len(candidates) == 1:
            return candidates[0].response
        
        fusion_func = self.fusion_strategies.get(strategy, self._weighted_fusion)
        return fusion_func(candidates)
    
    def _select_best_response(self, candidates: List[ResponseCandidate]) -> ChatCompletionResponse:
        """Simply select the best response based on overall quality."""
        
        best_candidate = max(candidates, key=lambda c: (
            c.coherence_score + c.relevance_score + c.factual_accuracy_score + 
            c.creativity_score + c.safety_score
        ) / 5)
        
        return best_candidate.response
    
    def _consensus_fusion(self, candidates: List[ResponseCandidate]) -> ChatCompletionResponse:
        """Create a consensus response by identifying common elements."""
        
        # Extract content from all responses
        contents = []
        for candidate in candidates:
            if candidate.response.choices and candidate.response.choices[0].get('message', {}).get('content'):
                contents.append(candidate.response.choices[0]['message']['content'])
        
        if not contents:
            return candidates[0].response
        
        # Find common sentences or phrases
        common_elements = self._find_common_elements(contents)
        
        # Build consensus response
        if common_elements:
            consensus_content = " ".join(common_elements)
        else:
            # Fallback to best response
            return self._select_best_response(candidates)
        
        # Create new response based on the best candidate's structure
        best_candidate = max(candidates, key=lambda c: c.coherence_score + c.relevance_score)
        fused_response = best_candidate.response.model_copy()
        
        if fused_response.choices:
            fused_response.choices[0]['message']['content'] = consensus_content
        
        return fused_response
    
    def _weighted_fusion(self, candidates: List[ResponseCandidate]) -> ChatCompletionResponse:
        """Create a weighted fusion based on candidate quality scores."""
        
        # Calculate weights based on overall quality
        weights = []
        for candidate in candidates:
            quality_score = (
                candidate.coherence_score + candidate.relevance_score + 
                candidate.factual_accuracy_score + candidate.creativity_score + 
                candidate.safety_score
            ) / 5
            weights.append(quality_score)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            weights = [1.0 / len(candidates)] * len(candidates)
        else:
            weights = [w / total_weight for w in weights]
        
        # Select response segments based on weights
        selected_candidate = np.random.choice(candidates, p=weights)
        
        # For now, return the selected response
        # In a more sophisticated implementation, we could combine text segments
        return selected_candidate.response
    
    def _extractive_fusion(self, candidates: List[ResponseCandidate]) -> ChatCompletionResponse:
        """Extract the best parts from each response and combine them."""
        
        # Extract sentences from all responses
        all_sentences = []
        for candidate in candidates:
            if candidate.response.choices and candidate.response.choices[0].get('message', {}).get('content'):
                content = candidate.response.choices[0]['message']['content']
                sentences = [s.strip() for s in content.split('.') if s.strip()]
                
                for sentence in sentences:
                    all_sentences.append({
                        'text': sentence,
                        'candidate': candidate,
                        'quality': (candidate.coherence_score + candidate.relevance_score) / 2
                    })
        
        # Sort sentences by quality and select the best ones
        all_sentences.sort(key=lambda x: x['quality'], reverse=True)
        
        # Select top sentences (avoid repetition)
        selected_sentences = []
        used_keywords = set()
        
        for sentence_info in all_sentences:
            sentence = sentence_info['text']
            sentence_keywords = set(sentence.lower().split())
            
            # Check for significant overlap with already selected sentences
            overlap = len(sentence_keywords.intersection(used_keywords))
            if overlap < len(sentence_keywords) * 0.7:  # Less than 70% overlap
                selected_sentences.append(sentence)
                used_keywords.update(sentence_keywords)
                
                if len(selected_sentences) >= 5:  # Limit to 5 sentences
                    break
        
        # Combine selected sentences
        fused_content = ". ".join(selected_sentences) + "."
        
        # Create new response
        best_candidate = max(candidates, key=lambda c: c.coherence_score + c.relevance_score)
        fused_response = best_candidate.response.model_copy()
        
        if fused_response.choices:
            fused_response.choices[0]['message']['content'] = fused_content
        
        return fused_response
    
    def _find_common_elements(self, contents: List[str]) -> List[str]:
        """Find common sentences or phrases across multiple responses."""
        
        # Split into sentences
        all_sentences = []
        for content in contents:
            sentences = [s.strip() for s in content.split('.') if s.strip()]
            all_sentences.extend(sentences)
        
        # Find sentences that appear in multiple responses
        sentence_counts = defaultdict(int)
        for sentence in all_sentences:
            # Normalize sentence for comparison
            normalized = re.sub(r'\s+', ' ', sentence.lower().strip())
            sentence_counts[normalized] += 1
        
        # Return sentences that appear in at least 2 responses
        common_sentences = [
            sentence for sentence, count in sentence_counts.items()
            if count >= 2
        ]
        
        return common_sentences[:3]  # Limit to top 3 common sentences


class EnsembleSystem:
    """Main ensemble system that coordinates multiple models and fuses their responses."""
    
    def __init__(self):
        self.quality_evaluator = ResponseQualityEvaluator()
        self.pairwise_ranker = PairwiseRanker()
        self.response_fuser = ResponseFuser()
        
        # Configuration
        self.max_candidates = 3
        self.quality_threshold = 0.6
        self.fusion_strategy = 'weighted_fusion'
    
    async def generate_ensemble_response(self, request: ChatCompletionRequest,
                                       model_responses: Dict[str, ChatCompletionResponse],
                                       model_metadata: Dict[str, Dict]) -> ChatCompletionResponse:
        """Generate an ensemble response from multiple model outputs."""
        
        # Create response candidates
        candidates = []
        for model_name, response in model_responses.items():
            metadata = model_metadata.get(model_name, {})
            
            # Evaluate response quality
            quality_metrics = self.quality_evaluator.evaluate_response(response, request)
            
            candidate = ResponseCandidate(
                model_name=model_name,
                provider=metadata.get('provider', 'unknown'),
                response=response,
                confidence_score=metadata.get('confidence', 0.5),
                response_time=metadata.get('response_time', 0.0),
                cost_estimate=metadata.get('cost_estimate', 0.0),
                coherence_score=quality_metrics['coherence_score'],
                relevance_score=quality_metrics['relevance_score'],
                factual_accuracy_score=quality_metrics['factual_accuracy_score'],
                creativity_score=quality_metrics['creativity_score'],
                safety_score=quality_metrics['safety_score']
            )
            
            candidates.append(candidate)
        
        # Filter candidates by quality threshold
        quality_candidates = [
            c for c in candidates 
            if (c.coherence_score + c.relevance_score + c.factual_accuracy_score) / 3 >= self.quality_threshold
        ]
        
        if not quality_candidates:
            quality_candidates = candidates  # Fallback to all candidates
        
        # Rank candidates
        ranked_candidates = self.pairwise_ranker.rank_responses(quality_candidates)
        
        # Select top candidates for fusion
        top_candidates = ranked_candidates[:self.max_candidates]
        
        # Fuse responses
        fused_response = self.response_fuser.fuse_responses(top_candidates, self.fusion_strategy)
        
        # Add ensemble metadata
        fused_response.provider = "ensemble"
        fused_response.model = f"ensemble-{len(top_candidates)}-models"
        
        # Add usage information
        if hasattr(fused_response, 'usage') and fused_response.usage:
            # Aggregate usage from all candidates
            total_prompt_tokens = sum(
                c.response.usage.get('prompt_tokens', 0) for c in top_candidates
                if c.response.usage
            )
            total_completion_tokens = sum(
                c.response.usage.get('completion_tokens', 0) for c in top_candidates
                if c.response.usage
            )
            
            fused_response.usage.update({
                'prompt_tokens': total_prompt_tokens,
                'completion_tokens': total_completion_tokens,
                'total_tokens': total_prompt_tokens + total_completion_tokens,
                'ensemble_models': len(top_candidates)
            })
        
        return fused_response
    
    def get_ensemble_insights(self, candidates: List[ResponseCandidate]) -> Dict[str, Any]:
        """Get insights about the ensemble decision process."""
        
        insights = {
            'candidate_count': len(candidates),
            'quality_distribution': {},
            'model_performance': {},
            'fusion_rationale': {}
        }
        
        # Quality distribution
        quality_scores = [
            (c.coherence_score + c.relevance_score + c.factual_accuracy_score) / 3
            for c in candidates
        ]
        
        insights['quality_distribution'] = {
            'mean': np.mean(quality_scores),
            'std': np.std(quality_scores),
            'min': np.min(quality_scores),
            'max': np.max(quality_scores)
        }
        
        # Model performance
        for candidate in candidates:
            insights['model_performance'][candidate.model_name] = {
                'coherence': candidate.coherence_score,
                'relevance': candidate.relevance_score,
                'factual_accuracy': candidate.factual_accuracy_score,
                'creativity': candidate.creativity_score,
                'safety': candidate.safety_score,
                'response_time': candidate.response_time
            }
        
        # Fusion rationale
        best_candidate = max(candidates, key=lambda c: (
            c.coherence_score + c.relevance_score + c.factual_accuracy_score
        ) / 3)
        
        insights['fusion_rationale'] = {
            'best_model': best_candidate.model_name,
            'fusion_strategy': self.fusion_strategy,
            'quality_threshold': self.quality_threshold
        }
        
        return insights