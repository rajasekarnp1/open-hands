#!/usr/bin/env python3
"""
Enhanced Demo showcasing research-based improvements to the LLM API Aggregator.

Features demonstrated:
1. Meta-controller with intelligent model selection
2. Task complexity analysis
3. Ensemble system for improved accuracy
4. Performance learning and adaptation
5. ArXiv research integration
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models import ChatMessage, ChatCompletionRequest
from src.core.aggregator import LLMAggregator
from src.core.account_manager import AccountManager
from src.core.router import ProviderRouter
from src.core.rate_limiter import RateLimiter
from src.providers.openrouter import create_openrouter_provider
from src.providers.groq import create_groq_provider
from src.providers.cerebras import create_cerebras_provider

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.json import JSON
from rich.tree import Tree


console = Console()


async def demo_research_enhancements():
    """Demonstrate the research-based enhancements."""

    console.print(Panel.fit(
        "🧠 Enhanced LLM API Aggregator Demo\n"
        "Showcasing ArXiv research integration:\n"
        "• FrugalGPT-style cascade routing\n"
        "• Meta-controller with external memory\n"
        "• LLM-Blender ensemble system\n"
        "• Task complexity analysis\n"
        "• Continuous learning and adaptation",
        title="Research-Enhanced Demo"
    ))

    # Initialize components with enhanced features
    console.print("\n[bold blue]1. Initializing Enhanced System[/bold blue]")

    account_manager = AccountManager()
    rate_limiter = RateLimiter()

    # Create providers
    providers = []

    console.print("   📡 Creating OpenRouter provider...")
    openrouter = create_openrouter_provider([])
    providers.append(openrouter)

    console.print("   ⚡ Creating Groq provider...")
    groq = create_groq_provider([])
    providers.append(groq)

    console.print("   🧠 Creating Cerebras provider...")
    cerebras = create_cerebras_provider([])
    providers.append(cerebras)

    # Create provider configs
    provider_configs = {provider.name: provider.config for provider in providers}

    # Initialize router
    router = ProviderRouter(provider_configs)

    # Initialize enhanced aggregator with meta-controller and ensemble
    aggregator = LLMAggregator(
        providers=providers,
        account_manager=account_manager,
        router=router,
        rate_limiter=rate_limiter,
        enable_meta_controller=True,  # Enable meta-controller
        enable_ensemble=False  # Start with meta-controller only
    )

    console.print("   ✅ Enhanced system initialized with meta-controller!")

    # Demonstrate task complexity analysis
    console.print("\n[bold blue]2. Task Complexity Analysis[/bold blue]")

    test_tasks = [
        {
            "description": "Simple question",
            "request": ChatCompletionRequest(
                model="auto",
                messages=[ChatMessage(role="user", content="What is the capital of France?")]
            )
        },
        {
            "description": "Code generation task",
            "request": ChatCompletionRequest(
                model="auto",
                messages=[ChatMessage(role="user", content="Write a Python function to implement quicksort algorithm with detailed comments")]
            )
        },
        {
            "description": "Complex reasoning task",
            "request": ChatCompletionRequest(
                model="auto",
                messages=[ChatMessage(role="user", content="Think step by step and solve this logic puzzle: If all roses are flowers, and some flowers are red, and all red things are beautiful, what can we conclude about roses?")]
            )
        },
        {
            "description": "Creative writing task",
            "request": ChatCompletionRequest(
                model="auto",
                messages=[ChatMessage(role="user", content="Write a creative short story about a robot who discovers emotions, incorporating themes of identity and consciousness")]
            )
        }
    ]

    for i, task in enumerate(test_tasks, 1):
        console.print(f"\n   [bold]Task {i}: {task['description']}[/bold]")

        # Analyze task complexity
        complexity = await aggregator.analyze_task_complexity(task["request"])

        if complexity:
            complexity_table = Table(title=f"Complexity Analysis - Task {i}")
            complexity_table.add_column("Dimension", style="cyan")
            complexity_table.add_column("Score", style="green")
            complexity_table.add_column("Description", style="yellow")

            complexity_table.add_row(
                "Reasoning Depth",
                f"{complexity['reasoning_depth']:.2f}",
                "How much logical reasoning is required"
            )
            complexity_table.add_row(
                "Domain Specificity",
                f"{complexity['domain_specificity']:.2f}",
                "How specialized the domain knowledge is"
            )
            complexity_table.add_row(
                "Computational Intensity",
                f"{complexity['computational_intensity']:.2f}",
                "How computationally demanding the task is"
            )
            complexity_table.add_row(
                "Creativity Required",
                f"{complexity['creativity_required']:.2f}",
                "How much creative thinking is needed"
            )
            complexity_table.add_row(
                "Factual Accuracy",
                f"{complexity['factual_accuracy_importance']:.2f}",
                "How important factual accuracy is"
            )
            complexity_table.add_row(
                "Overall Complexity",
                f"{complexity['overall_complexity']:.2f}",
                "Combined complexity score"
            )

            console.print(complexity_table)

    # Demonstrate intelligent model selection
    console.print("\n[bold blue]3. Intelligent Model Selection[/bold blue]")

    for i, task in enumerate(test_tasks, 1):
        console.print(f"\n   [bold]Task {i}: {task['description']}[/bold]")

        # Get model recommendations
        recommendations = await aggregator.get_model_recommendations(task["request"])

        console.print(f"   [green]Traditional routing:[/green] {' → '.join(recommendations['traditional_routing'])}")

        if recommendations['meta_controller_insights']:
            insights = recommendations['meta_controller_insights']
            console.print(f"   [blue]Meta-controller optimal model:[/blue] {insights['optimal_model']}")
            console.print(f"   [blue]Confidence:[/blue] {insights['confidence']:.2f}")
            console.print(f"   [blue]Cascade chain:[/blue] {' → '.join(insights['cascade_chain'])}")

    # Demonstrate meta-controller insights
    console.print("\n[bold blue]4. Meta-Controller Performance Insights[/bold blue]")

    insights = await aggregator.get_meta_controller_insights()

    if insights:
        # Model performance table
        perf_table = Table(title="Model Performance Analysis")
        perf_table.add_column("Model", style="cyan")
        perf_table.add_column("Reliability", style="green")
        perf_table.add_column("Avg Response Time", style="yellow")
        perf_table.add_column("Size Category", style="blue")
        perf_table.add_column("Cost/Token", style="magenta")

        for model_name, performance in insights['model_performance'].items():
            perf_table.add_row(
                model_name[:30] + "..." if len(model_name) > 30 else model_name,
                f"{performance['reliability_score']:.2f}",
                f"{performance['avg_response_time']:.1f}s",
                performance['size_category'],
                f"{performance['cost_per_token']:.4f}"
            )

        console.print(perf_table)

        # Recommendations
        if insights['recommendations']:
            console.print("\n   [bold green]System Recommendations:[/bold green]")
            for rec in insights['recommendations']:
                console.print(f"   • {rec}")

    # Demonstrate ensemble system
    console.print("\n[bold blue]5. Ensemble System Demo[/bold blue]")

    # Enable ensemble for demonstration
    aggregator.enable_ensemble = True
    if not aggregator.ensemble_system:
        from src.core.ensemble_system import EnsembleSystem
        aggregator.ensemble_system = EnsembleSystem()

    console.print("   🔄 Ensemble system enabled")

    # Test ensemble on a complex task
    complex_task = ChatCompletionRequest(
        model="auto",
        messages=[ChatMessage(role="user", content="Analyze the pros and cons of different machine learning algorithms for text classification")]
    )

    ensemble_insights = await aggregator.get_ensemble_insights(complex_task)

    if ensemble_insights:
        console.print("\n   [bold]Ensemble Decision Analysis:[/bold]")

        # Quality distribution
        quality_dist = ensemble_insights['quality_distribution']
        console.print(f"   Quality distribution: μ={quality_dist['mean']:.2f}, σ={quality_dist['std']:.2f}")

        # Model performance in ensemble
        console.print("\n   [bold]Model Performance in Ensemble:[/bold]")
        for model_name, performance in ensemble_insights['model_performance'].items():
            console.print(f"   {model_name}:")
            console.print(f"     Coherence: {performance['coherence']:.2f}")
            console.print(f"     Relevance: {performance['relevance']:.2f}")
            console.print(f"     Factual Accuracy: {performance['factual_accuracy']:.2f}")
            console.print(f"     Response Time: {performance['response_time']:.1f}s")

    # Demonstrate learning and adaptation
    console.print("\n[bold blue]6. Learning and Adaptation[/bold blue]")

    console.print("   📚 The system continuously learns from:")
    console.print("   • Response quality and user satisfaction")
    console.print("   • Model performance across different task types")
    console.print("   • User preferences and feedback")
    console.print("   • Provider reliability and response times")

    console.print("\n   🧠 External memory system stores:")
    console.print("   • Task patterns and optimal model mappings")
    console.print("   • Historical performance data")
    console.print("   • User preference profiles")
    console.print("   • Model capability assessments")

    # Show research integration
    console.print("\n[bold blue]7. ArXiv Research Integration[/bold blue]")

    research_tree = Tree("🔬 Integrated Research Papers")

    frugal_node = research_tree.add("📄 FrugalGPT (arXiv:2305.05176)")
    frugal_node.add("✓ Cascade routing from small to large models")
    frugal_node.add("✓ Query complexity scoring")
    frugal_node.add("✓ Cost-performance optimization")

    route_node = research_tree.add("📄 RouteLLM (arXiv:2406.18665)")
    route_node.add("✓ Learning routing policies from preference data")
    route_node.add("✓ Cost-quality trade-offs")
    route_node.add("✓ Adaptive routing based on feedback")

    blender_node = research_tree.add("📄 LLM-Blender (arXiv:2306.02561)")
    blender_node.add("✓ Pairwise ranking for model comparison")
    blender_node.add("✓ Generative fusion of multiple outputs")
    blender_node.add("✓ Quality-based ensemble selection")

    moe_node = research_tree.add("📄 Mixture of Experts (arXiv:2305.14705)")
    moe_node.add("✓ Task-specific expert activation")
    moe_node.add("✓ Gating mechanisms for routing")
    moe_node.add("✓ Dynamic model selection")

    tree_node = research_tree.add("📄 Tree of Thoughts (arXiv:2305.10601)")
    tree_node.add("✓ Deliberate problem-solving approach")
    tree_node.add("✓ Reasoning path exploration")
    tree_node.add("✓ Model selection based on reasoning needs")

    console.print(research_tree)

    # Performance comparison
    console.print("\n[bold blue]8. Performance Comparison[/bold blue]")

    comparison_table = Table(title="Traditional vs Enhanced System")
    comparison_table.add_column("Feature", style="cyan")
    comparison_table.add_column("Traditional", style="red")
    comparison_table.add_column("Enhanced", style="green")

    comparison_table.add_row("Model Selection", "Round-robin/Random", "Intelligent task-based")
    comparison_table.add_row("Task Analysis", "Basic keyword matching", "Multi-dimensional complexity")
    comparison_table.add_row("Learning", "Static configuration", "Continuous adaptation")
    comparison_table.add_row("Quality Assurance", "Single model output", "Ensemble validation")
    comparison_table.add_row("Cost Optimization", "Manual configuration", "Automatic FrugalGPT")
    comparison_table.add_row("Failure Handling", "Simple fallback", "Intelligent cascade")
    comparison_table.add_row("Performance Tracking", "Basic metrics", "Comprehensive analytics")

    console.print(comparison_table)

    # Cleanup
    await aggregator.close()

    console.print(Panel.fit(
        "🎉 Enhanced Demo Complete!\n\n"
        "Key Research-Based Improvements:\n"
        "• 🧠 Meta-controller with external memory for intelligent model selection\n"
        "• 📊 Multi-dimensional task complexity analysis\n"
        "• 🔄 FrugalGPT-style cascade routing for cost optimization\n"
        "• 🤝 LLM-Blender ensemble system for improved accuracy\n"
        "• 📚 Continuous learning from performance feedback\n"
        "• 🎯 Task-specific model specialization\n\n"
        "The system now intelligently adapts to different tasks,\n"
        "learns from experience, and optimizes for both cost and quality!\n\n"
        "Next: Try the enhanced system with real API keys! 🚀",
        title="Research Integration Success"
    ))


def main():
    """Main demo function."""

    console.print("[bold]🧠 Enhanced LLM API Aggregator - Research Integration Demo[/bold]")
    console.print()

    # Run enhanced demo
    asyncio.run(demo_research_enhancements())


if __name__ == "__main__":
    main()