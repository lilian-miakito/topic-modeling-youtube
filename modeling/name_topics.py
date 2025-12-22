#!/usr/bin/env python3
"""
Name topics using DSPy - Declarative LLM programming.
Takes the most recent topics_result file and generates human-readable names
for each topic using a DSPy signature and predictor.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

from lib import DATASETS_DIR, get_latest_file
from lib.topic_namer import configure_dspy, create_optimized_namer, load_ground_truth

# ==== CONFIGURATION ====
# Supported: "openai/gpt-5-mini" (default), "openai/gpt-4o", "ollama_chat/llama3.2"
DEFAULT_MODEL = "openai/gpt-5-mini"

NUM_TOP_WORDS = 20              # Number of keywords to include per topic
NUM_COMMENTS = 8                # Number of example comments per topic
NUM_PARALLEL_WORKERS = 10       # Number of parallel API requests


def get_latest_topics_file():
    """Find the most recent topics_result file."""
    latest = get_latest_file(DATASETS_DIR, "topics_result_*.json")
    if latest:
        return latest
    # Fallback to old filename
    old_file = DATASETS_DIR / "topics_result.json"
    return old_file if old_file.exists() else None


def load_topics(filepath):
    """Load topics from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    print("=" * 60)
    print("Topic Naming with DSPy")
    print("=" * 60)
    
    # Load .env file if available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    # Get model from env or use default
    model = os.environ.get("DSPY_MODEL", DEFAULT_MODEL)
    
    # Find latest topics file
    topics_file = get_latest_topics_file()
    if not topics_file:
        print("No topics_result file found. Run extract_topics.py first!")
        return
    
    print(f"\nLoading: {topics_file.name}")
    data = load_topics(topics_file)
    
    topics = data.get('topics', [])
    print(f"Found {len(topics)} topics to name")
    
    if not topics:
        print("No topics found in file!")
        return
    
    # Configure DSPy
    print(f"\nConfiguring DSPy with model: {model}")
    try:
        configure_dspy(model)
    except Exception as e:
        print(f"Failed to configure DSPy: {e}")
        return
    
    # Check for ground truth (few-shot examples)
    ground_truth = load_ground_truth()
    print(f"Ground truth entries available: {len(ground_truth)}")
    
    # Create namer with few-shot examples if available (max 10)
    namer = create_optimized_namer(max_examples=10)
    
    # Generate names for each topic IN PARALLEL
    print(f"\nGenerating topic names ({NUM_PARALLEL_WORKERS} workers)...")
    print("-" * 60)
    
    def name_single_topic(topic):
        """Name a single topic - runs in thread."""
        topic_id = topic.get('id', '?')
        
        # Get keywords (prefer semantic centroid_mmr)
        top_words = topic.get('top_words_centroid_mmr', topic.get('top_words', []))
        
        # Get example comments (prefer centroid_mmr representatives)
        example_comments = topic.get(
            'example_comments_centroid_mmr', 
            topic.get('example_comments_original', topic.get('example_comments', []))
        )
        
        if not top_words:
            name = "Unknown Topic"
        else:
            try:
                name = namer(
                    keywords=top_words[:NUM_TOP_WORDS],
                    comments=example_comments[:NUM_COMMENTS]
                )
            except Exception as e:
                print(f"  Error naming topic {topic_id}: {e}")
                name = f"Topic {topic_id}"
        
        return {
            **topic,
            'generated_name': name
        }
    
    # Run in parallel with ThreadPoolExecutor
    named_topics = [None] * len(topics)
    completed = 0
    
    with ThreadPoolExecutor(max_workers=NUM_PARALLEL_WORKERS) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(name_single_topic, topic): idx 
            for idx, topic in enumerate(topics)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                named_topics[idx] = result
                completed += 1
                
                # Progress output
                topic_id = result.get('id', '?')
                name = result.get('generated_name', '?')
                keywords = result.get('top_words_centroid_mmr', result.get('top_words', []))[:5]
                print(f"[{completed}/{len(topics)}] Topic {topic_id}: {name}")
                print(f"         Keywords: {', '.join(keywords)}")
            except Exception as e:
                print(f"  Error processing topic at index {idx}: {e}")
                named_topics[idx] = {**topics[idx], 'generated_name': f"Topic {topics[idx].get('id', idx)}"}
    
    # Merge in-place: update original file with generated names
    data['topics'] = named_topics
    data['naming'] = {
        'generated_at': datetime.now().isoformat(),
        'model_used': model,
        'method': 'dspy'
    }
    
    with open(topics_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print("=" * 60)
    print(f"Names merged into: {topics_file.name}")
    
    # Print summary
    print("\n--- Summary ---")
    for topic in named_topics:
        print(f"  [{topic['id']}] {topic['generated_name']} ({topic.get('count', '?')} comments)")


if __name__ == "__main__":
    main()
