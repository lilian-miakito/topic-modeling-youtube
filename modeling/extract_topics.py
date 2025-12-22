#!/usr/bin/env python3
"""
Extract topics from YouTube comments using BERTopic.
Takes a random sample of 1000 comments for quick experimentation.
"""

import random
import json
from pathlib import Path
from datetime import datetime

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

from bertopic import BERTopic


DATASETS_DIR = Path(__file__).parent / "datasets"
SAMPLE_SIZE = 10000


def load_comments():
    """Load comments from the dataset."""
    comments_file = DATASETS_DIR / "comments.txt"
    
    if not comments_file.exists():
        print(f"Dataset not found: {comments_file}")
        print("Run create_dataset.py first!")
        return []
    
    with open(comments_file, 'r', encoding='utf-8') as f:
        comments = [line.strip() for line in f if line.strip()]
    
    return comments


def main():
    print("=" * 60)
    print("Topic Extraction with BERTopic")
    print("=" * 60)
    
    # Load comments
    print("\n1. Loading comments...")
    all_comments = load_comments()
    
    if not all_comments:
        return
    
    print(f"   Total comments available: {len(all_comments):,}")
    
    # Random sample
    print(f"\n2. Sampling {SAMPLE_SIZE} random comments...")
    random.seed(42)  # For reproducibility
    sample = random.sample(all_comments, min(SAMPLE_SIZE, len(all_comments)))
    print(f"   Sample size: {len(sample)}")
    
    # Filter very short comments (less useful for topic modeling)
    sample = [c for c in sample if len(c) > 20]
    print(f"   After filtering short comments: {len(sample)}")
    
    # Create BERTopic model
    print("\n3. Creating BERTopic model...")
    print("   (This may take a minute on first run - downloading model)")
    
    topic_model = BERTopic(
        language="multilingual",  # Works for English + other languages
        min_topic_size=5,         # Minimum documents per topic
        verbose=True
    )
    
    # Fit the model
    print("\n4. Fitting model to comments...")
    topics, probs = topic_model.fit_transform(sample)
    
    # Get topic info
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    topic_info = topic_model.get_topic_info()
    print(f"\nFound {len(topic_info) - 1} topics (excluding outliers)")
    
    print("\n--- Topic Overview ---")
    print(topic_info.to_string())
    
    # Show top words for each topic
    print("\n--- Top Words per Topic ---")
    for topic_id in topic_info['Topic'].tolist():
        if topic_id == -1:
            continue  # Skip outlier topic
        topic_words = topic_model.get_topic(topic_id)
        if topic_words:
            words = [word for word, _ in topic_words[:8]]
            print(f"\nTopic {topic_id}: {', '.join(words)}")
    
    # Show example comments for top topics
    print("\n--- Example Comments per Topic ---")
    for topic_id in topic_info['Topic'].tolist()[:6]:  # Top 5 + outliers
        if topic_id == -1:
            continue
        
        # Find comments in this topic
        topic_comments = [sample[i] for i, t in enumerate(topics) if t == topic_id]
        
        print(f"\n[Topic {topic_id}]")
        for comment in topic_comments[:3]:  # Show 3 examples
            preview = comment[:100] + "..." if len(comment) > 100 else comment
            print(f"  • {preview}")
    
    # Save results
    output_file = DATASETS_DIR / "topics_result.json"
    result = {
        'generated_at': datetime.now().isoformat(),
        'sample_size': len(sample),
        'num_topics': len(topic_info) - 1,
        'topics': []
    }
    
    for topic_id in topic_info['Topic'].tolist():
        if topic_id == -1:
            continue
        topic_words = topic_model.get_topic(topic_id)
        topic_comments = [sample[i] for i, t in enumerate(topics) if t == topic_id]
        result['topics'].append({
            'id': topic_id,
            'count': len(topic_comments),
            'top_words': [word for word, _ in topic_words[:10]] if topic_words else [],
            'example_comments': topic_comments[:5]
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

