#!/usr/bin/env python3
"""
Profile topic extraction to find bottlenecks.
"""

import cProfile
import pstats
import io
import random
from pathlib import Path

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

import polars as pl

DATASETS_DIR = Path(__file__).parent / "datasets"
SAMPLE_SIZE = 1000


def load_sample():
    """Load a small sample of comments from Parquet."""
    parquet_file = DATASETS_DIR / "comments_full.parquet"
    
    df = pl.read_parquet(parquet_file)
    comments = [c for c in df["text"].to_list() if c and len(c) > 20]
    
    random.seed(42)
    return random.sample(comments, min(SAMPLE_SIZE, len(comments)))


def run_bertopic(comments):
    """Run BERTopic on comments."""
    from bertopic import BERTopic
    
    topic_model = BERTopic(
        language="multilingual",
        min_topic_size=2,
        verbose=False
    )
    
    topics, probs = topic_model.fit_transform(comments)
    return topics


def main():
    print("=" * 60)
    print(f"Profiling BERTopic on {SAMPLE_SIZE} comments")
    print("=" * 60)
    
    # Load sample outside profiler
    print("\nLoading sample...")
    comments = load_sample()
    print(f"Loaded {len(comments)} comments")
    
    # Profile
    print("\nRunning profiler...")
    profiler = cProfile.Profile()
    profiler.enable()
    
    topics = run_bertopic(comments)
    
    profiler.disable()
    
    # Print results
    print("\n" + "=" * 60)
    print("PROFILING RESULTS (top 30 by cumulative time)")
    print("=" * 60 + "\n")
    
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    stats.print_stats(30)
    print(s.getvalue())
    
    # Also show by total time
    print("\n" + "=" * 60)
    print("TOP 20 by total time (in function only)")
    print("=" * 60 + "\n")
    
    s2 = io.StringIO()
    stats2 = pstats.Stats(profiler, stream=s2).sort_stats('tottime')
    stats2.print_stats(20)
    print(s2.getvalue())


if __name__ == "__main__":
    main()

