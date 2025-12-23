"""
Topic naming using DSPy - Declarative LLM programming.
Uses signatures and predictors for generating topic names.
"""
import os
from typing import Optional

import dspy


# =============================================================================
# DSPy Signatures
# =============================================================================

class TopicNamingSignature(dspy.Signature):
    """Generate a short, descriptive name for a discussion topic based on keywords and example comments."""
    
    keywords: str = dspy.InputField(desc="Comma-separated list of representative keywords for the topic")
    example_comments: str = dspy.InputField(desc="Sample comments from this topic cluster")
    
    topic_name: str = dspy.OutputField(desc="A short, descriptive name for this topic (2-5 words)")


class SubtopicNamingSignature(dspy.Signature):
    """Generate a sub-category name that refines the parent topic. Must be distinct from parent."""
    
    parent_topic_name: str = dspy.InputField(desc="Name of the parent topic (e.g., 'Cuisine')")
    keywords: str = dspy.InputField(desc="Keywords specific to this sub-cluster")
    example_comments: str = dspy.InputField(desc="Sample comments from this sub-cluster")
    
    subtopic_name: str = dspy.OutputField(
        desc="A short, specific sub-category name (2-4 words). Must refine the parent, not repeat it."
    )


# =============================================================================
# Topic Namer Module
# =============================================================================

class TopicNamer(dspy.Module):
    """DSPy module for naming topics based on their content."""
    
    def __init__(self, demos: list = None):
        super().__init__()
        self.predictor = dspy.Predict(TopicNamingSignature)
        
        if demos:
            self.predictor.demos = demos
    
    def forward(self, keywords: list[str], comments: list[str], max_comments: int = 5) -> str:
        """
        Generate a name for a topic.
        
        Args:
            keywords: List of representative keywords for the topic
            comments: List of example comments from the topic
            max_comments: Maximum number of comments to include in context
            
        Returns:
            Generated topic name
        """
        keywords_str = ", ".join(keywords[:20])
        
        comments_str = "\n".join([
            f"- {comment[:300]}{'...' if len(comment) > 300 else ''}"
            for comment in comments[:max_comments]
        ])
        
        result = self.predictor(
            keywords=keywords_str,
            example_comments=comments_str
        )
        
        return result.topic_name.strip()


class SubtopicNamer(dspy.Module):
    """DSPy module for naming sub-topics with parent context."""
    
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(SubtopicNamingSignature)
    
    def forward(
        self,
        parent_name: str,
        keywords: list[str],
        comments: list[str],
        max_comments: int = 5,
    ) -> str:
        """
        Generate a name for a sub-topic.
        
        Args:
            parent_name: Name of the parent topic
            keywords: List of representative keywords for the sub-topic
            comments: List of example comments
            max_comments: Maximum number of comments to include
            
        Returns:
            Generated sub-topic name
        """
        keywords_str = ", ".join(keywords[:15])
        
        comments_str = "\n".join([
            f"- {comment[:250]}{'...' if len(comment) > 250 else ''}"
            for comment in comments[:max_comments]
        ])
        
        result = self.predictor(
            parent_topic_name=parent_name,
            keywords=keywords_str,
            example_comments=comments_str
        )
        
        return result.subtopic_name.strip()


# =============================================================================
# Configuration
# =============================================================================

def configure_dspy(model_name: str = "openai/gpt-4o-mini", api_key: str = None):
    """
    Configure DSPy with a language model.
    
    Supported models:
    - openai/gpt-4o-mini (default), openai/gpt-4o
    - anthropic/claude-3-haiku-20240307
    - ollama_chat/llama3.2 (local Ollama)
    """
    # Load from .env if available
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


# =============================================================================
# Singletons for reuse
# =============================================================================

_topic_namer: Optional[TopicNamer] = None
_subtopic_namer: Optional[SubtopicNamer] = None


def get_topic_namer() -> TopicNamer:
    """Get or create the topic namer singleton."""
    global _topic_namer
    if _topic_namer is None:
        _topic_namer = TopicNamer()
    return _topic_namer


def get_subtopic_namer() -> SubtopicNamer:
    """Get or create the subtopic namer singleton."""
    global _subtopic_namer
    if _subtopic_namer is None:
        _subtopic_namer = SubtopicNamer()
    return _subtopic_namer


def name_topic(keywords: list[str], comments: list[str]) -> str:
    """Convenience function to name a topic."""
    namer = get_topic_namer()
    return namer(keywords=keywords, comments=comments)


def name_subtopic(parent_name: str, keywords: list[str], comments: list[str]) -> str:
    """Convenience function to name a subtopic."""
    namer = get_subtopic_namer()
    return namer(parent_name=parent_name, keywords=keywords, comments=comments)

