"""
Topic naming using DSPy - Declarative LLM programming.
Uses signatures and predictors with few-shot examples from ground truth.
"""

import json
import os
from pathlib import Path
from typing import List, Optional

import dspy

# Path to ground truth file
DATASETS_DIR = Path(__file__).parent.parent / "datasets"
GROUND_TRUTH_FILE = DATASETS_DIR / "naming_ground_truth.json"


# =============================================================================
# DSPy Signature for Topic Naming
# =============================================================================

class TopicNamingSignature(dspy.Signature):
    """Generate a short, descriptive name for a discussion topic based on keywords and example comments."""
    
    keywords: str = dspy.InputField(desc="Comma-separated list of representative keywords for the topic")
    example_comments: str = dspy.InputField(desc="Sample comments from this topic cluster")
    
    topic_name: str = dspy.OutputField(desc="A short, descriptive name for this topic (2-5 words)")


# =============================================================================
# Topic Namer Module with Few-Shot Support
# =============================================================================

class TopicNamer(dspy.Module):
    """DSPy module for naming topics based on their content, with few-shot examples."""
    
    def __init__(self, demos: List[dspy.Example] = None):
        super().__init__()
        self.predictor = dspy.Predict(TopicNamingSignature)
        
        # If demos provided, set them on the predictor
        if demos:
            self.predictor.demos = demos
    
    def forward(self, keywords: List[str], comments: List[str], max_comments: int = 5) -> str:
        """
        Generate a name for a topic.
        
        Args:
            keywords: List of representative keywords for the topic
            comments: List of example comments from the topic
            max_comments: Maximum number of comments to include in context
            
        Returns:
            Generated topic name
        """
        # Format inputs
        keywords_str = ", ".join(keywords[:20])
        
        comments_str = "\n".join([
            f"- {comment[:300]}{'...' if len(comment) > 300 else ''}"
            for comment in comments[:max_comments]
        ])
        
        # Run prediction
        result = self.predictor(
            keywords=keywords_str,
            example_comments=comments_str
        )
        
        return result.topic_name.strip()


# =============================================================================
# Ground Truth Loading
# =============================================================================

def load_ground_truth() -> dict:
    """Load the ground truth naming data from bootstrap."""
    if GROUND_TRUTH_FILE.exists():
        with open(GROUND_TRUTH_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def load_topics_data() -> dict:
    """Load the latest topics result file."""
    from . import get_latest_file
    
    topics_file = get_latest_file(DATASETS_DIR, "topics_result_*.json")
    if not topics_file:
        return {}
    
    with open(topics_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_few_shot_examples(max_examples: int = 10) -> List[dspy.Example]:
    """
    Create DSPy Examples from ground truth for few-shot learning.
    
    Args:
        max_examples: Maximum number of examples to include
        
    Returns:
        List of dspy.Example objects
    """
    ground_truth = load_ground_truth()
    topics_data = load_topics_data()
    
    if not ground_truth or not topics_data:
        return []
    
    # Build a lookup of topic data by ID
    topics_by_id = {str(t['id']): t for t in topics_data.get('topics', [])}
    
    examples = []
    for topic_id, name in ground_truth.items():
        topic = topics_by_id.get(str(topic_id))
        if not topic:
            continue
        
        # Get keywords and comments
        keywords = topic.get('top_words_centroid_mmr', topic.get('top_words', []))
        comments = topic.get(
            'example_comments_centroid_mmr',
            topic.get('example_comments_original', topic.get('example_comments', []))
        )
        
        if not keywords:
            continue
        
        # Format for the signature
        keywords_str = ", ".join(keywords[:15])
        comments_str = "\n".join([
            f"- {c[:200]}{'...' if len(c) > 200 else ''}"
            for c in comments[:3]
        ])
        
        # Create DSPy Example
        example = dspy.Example(
            keywords=keywords_str,
            example_comments=comments_str,
            topic_name=name
        ).with_inputs("keywords", "example_comments")
        
        examples.append(example)
        
        if len(examples) >= max_examples:
            break
    
    return examples


# =============================================================================
# Optimized Namer with Bootstrap Few-Shot
# =============================================================================

def create_optimized_namer(max_examples: int = 10) -> TopicNamer:
    """
    Create a TopicNamer with few-shot examples from ground truth.
    
    If ground truth exists, examples are loaded and used as demonstrations.
    Otherwise, returns a basic namer without examples.
    
    Args:
        max_examples: Maximum number of few-shot examples to use
        
    Returns:
        Configured TopicNamer instance
    """
    examples = create_few_shot_examples(max_examples)
    
    if examples:
        print(f"  Loaded {len(examples)} few-shot examples from ground truth")
        return TopicNamer(demos=examples)
    else:
        print("  No ground truth found, using zero-shot naming")
        return TopicNamer()


# =============================================================================
# Configuration and Setup
# =============================================================================

def configure_dspy(model_name: str = "openai/gpt-5-mini", api_key: str = None):
    """
    Configure DSPy with a language model.
    
    Supported models:
    - openai/gpt-4o-mini (default), openai/gpt-4o
    - anthropic/claude-3-haiku-20240307
    - ollama_chat/llama3.2 (local Ollama)
    """
    # Load from .env if python-dotenv is available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    if model_name.startswith("openai/"):
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set. Add it to .env or environment.")
        lm = dspy.LM(model_name, api_key=api_key)
    elif model_name.startswith("anthropic/"):
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set. Add it to .env or environment.")
        lm = dspy.LM(model_name, api_key=api_key)
    elif model_name.startswith("ollama"):
        lm = dspy.LM(model_name)
    else:
        lm = dspy.LM(model_name, api_key=api_key)
    
    dspy.configure(lm=lm)
    return lm


def name_topic(keywords: List[str], comments: List[str], 
               model: str = "openai/gpt-4o-mini",
               use_few_shot: bool = True) -> str:
    """
    Convenience function to name a single topic.
    
    Args:
        keywords: List of topic keywords
        comments: List of example comments
        model: LLM model to use
        use_few_shot: Whether to use ground truth examples
        
    Returns:
        Generated topic name
    """
    configure_dspy(model)
    
    if use_few_shot:
        namer = create_optimized_namer()
    else:
        namer = TopicNamer()
    
    return namer(keywords=keywords, comments=comments)


# =============================================================================
# CLI for Testing
# =============================================================================

if __name__ == "__main__":
    # Test with sample data
    test_keywords = ["AI", "artificial intelligence", "machine learning", "future", "technology"]
    test_comments = [
        "I think AI is going to change everything in the next 10 years",
        "The potential of machine learning is incredible",
        "We need to be careful about how we develop AI systems",
    ]
    
    print("Testing DSPy Topic Namer...")
    print(f"Keywords: {test_keywords}")
    print(f"Comments: {len(test_comments)}")
    
    # Check for ground truth
    gt = load_ground_truth()
    print(f"Ground truth entries: {len(gt)}")
    
    # Try with ollama first (local), then OpenAI
    try:
        configure_dspy("ollama_chat/llama3.2")
        namer = create_optimized_namer()
        name = namer(keywords=test_keywords, comments=test_comments)
        print(f"\nGenerated name (Ollama): {name}")
    except Exception as e:
        print(f"Ollama failed: {e}")
        try:
            configure_dspy("openai/gpt-4o-mini")
            namer = create_optimized_namer()
            name = namer(keywords=test_keywords, comments=test_comments)
            print(f"\nGenerated name (OpenAI): {name}")
        except Exception as e2:
            print(f"OpenAI also failed: {e2}")
