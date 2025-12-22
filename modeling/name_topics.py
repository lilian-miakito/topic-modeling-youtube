#!/usr/bin/env python3
"""
Name topics using Google's Flan-T5 model.
Takes the most recent topics_result file and generates human-readable names
for each topic based on top words and example comments.
"""

import json
from pathlib import Path
from datetime import datetime

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

DATASETS_DIR = Path(__file__).parent / "datasets"
MODEL_NAME = "google/flan-t5-large"  # 780M params, bien meilleur

def get_latest_topics_file():
    """Find the most recent topics_result file."""
    files = list(DATASETS_DIR.glob("topics_result_*.json"))
    if not files:
        # Try the old filename
        old_file = DATASETS_DIR / "topics_result.json"
        if old_file.exists():
            return old_file
        return None
    
    # Sort by modification time, most recent first
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return files[0]


def load_topics(filepath):
    """Load topics from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_topic_name(model, tokenizer, top_words, example_comments, device):
    """Generate a topic name using Flan-T5."""
    # Build the prompt
    words_str = ", ".join(top_words[:10])
    
    # Take first 5 comments, truncate if too long
    comments_str = ""
    for i, comment in enumerate(example_comments[:5], 1):
        truncated = comment[:150] + "..." if len(comment) > 150 else comment
        comments_str += f"{i}. {truncated}\n"
    
    prompt = f"""Given these keywords and example comments from a discussion topic, generate a short, descriptive name for this topic (2-5 words max).

Keywords: {words_str}

Example comments:
{comments_str}

Topic name:"""

    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            num_beams=3,
            temperature=0.7,
            do_sample=False
        )
    
    # Decode
    name = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return name.strip()


def main():
    print("=" * 60)
    print("Topic Naming with Flan-T5")
    print("=" * 60)
    
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
    
    # Load model
    print(f"\nLoading model: {MODEL_NAME}")
    print("(This may take a moment on first run)")
    
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()
    
    # Generate names for each topic
    print("\nGenerating topic names...")
    print("-" * 60)
    
    named_topics = []
    for topic in topics:
        topic_id = topic.get('id', '?')
        top_words = topic.get('top_words', [])
        example_comments = topic.get('example_comments', [])
        
        if not top_words:
            name = "Unknown Topic"
        else:
            name = generate_topic_name(model, tokenizer, top_words, example_comments, device)
        
        named_topics.append({
            **topic,
            'generated_name': name
        })
        
        print(f"Topic {topic_id}: {name}")
        print(f"  Keywords: {', '.join(top_words[:5])}")
        print()
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = DATASETS_DIR / f"topics_named_{timestamp}.json"
    
    result = {
        'generated_at': datetime.now().isoformat(),
        'source_file': topics_file.name,
        'model_used': MODEL_NAME,
        'num_topics': len(named_topics),
        'topics': named_topics
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print("=" * 60)
    print(f"Results saved to: {output_file}")
    
    # Also print a summary
    print("\n--- Summary ---")
    for topic in named_topics:
        print(f"  [{topic['id']}] {topic['generated_name']} ({topic.get('count', '?')} comments)")


if __name__ == "__main__":
    main()

