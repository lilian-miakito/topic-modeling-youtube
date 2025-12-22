#!/usr/bin/env python3
"""
Name topics using Google's Flan-T5 model.
Takes the most recent topics_result file and generates human-readable names
for each topic based on top words and example comments.
"""

import json
from datetime import datetime

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

from lib import DATASETS_DIR, get_latest_file

# ==== CONSTANTES D'AJUSTEMENT ====
MODEL_NAME = "google/flan-t5-large"  # 780M params, bien meilleur

NUM_TOP_WORDS = 20              # Nombre de mots-clés à inclure par topic
NUM_COMMENTS = 10                # Nombre de commentaires exemples par topic
MAX_COMMENT_LENGTH = 500        # Longueur max d'un commentaire avant troncation
PROMPT_MAX_LENGTH = 1024         # Longueur max du prompt (tokenizer)
MAX_NEW_TOKENS = 128             # max_new_tokens pour la génération du nom
NUM_BEAMS = 4                   # num_beams pour la génération
TEMPERATURE = 0.7               # temperature pour la génération
DO_SAMPLE = False               # do_sample pour la génération
TOPIC_NAME_WORDS_HINT = "2-5 words max"


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


def generate_topic_name(model, tokenizer, top_words, example_comments, device):
    """Generate a topic name using Flan-T5."""
    # Build the prompt
    words_str = ", ".join(top_words[:NUM_TOP_WORDS])
    
    # Take first N comments, truncate if too long
    comments_str = ""
    for i, comment in enumerate(example_comments[:NUM_COMMENTS], 1):
        truncated = comment[:MAX_COMMENT_LENGTH] + "..." if len(comment) > MAX_COMMENT_LENGTH else comment
        comments_str += f"{i}. {truncated}\n"
    
    prompt = (
        f"Given these keywords and example comments from a discussion topic, generate a short, descriptive name for this topic ({TOPIC_NAME_WORDS_HINT}).\n\n"
        f"<example>\n"
        f"Keywords: AI, human-computer interaction, future, technology, innovation\n"
        f"Example comments:\n"
        f"You always talk about AI but i think it's really important.\n"
        f"I'm excited about the potential of AI to help us solve complex problems and make better decisions.\n"
        f"AI is already changing the way we work and live. It's going to be even more impactful in the future.\n"
        f"Topic name: The Future of AI and Human-Computer Interaction\n"
        f"</example>\n\n"
        f"Keywords: {words_str}\n\n"
        f"Example comments:\n{comments_str}\n"
        f"Topic name:"
    )

    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt", max_length=PROMPT_MAX_LENGTH, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            temperature=TEMPERATURE,
            do_sample=DO_SAMPLE
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
        # Use centroid_mmr words (semantic), fallback to old field name
        top_words = topic.get('top_words_centroid_mmr', topic.get('top_words', []))
        # Use centroid_mmr comments (representative), fallback to original
        example_comments = topic.get('example_comments_centroid_mmr', topic.get('example_comments_original', topic.get('example_comments', [])))
        
        if not top_words:
            name = "Unknown Topic"
        else:
            name = generate_topic_name(model, tokenizer, top_words, example_comments, device)
        
        named_topics.append({
            **topic,
            'generated_name': name
        })
        
        print(f"Topic {topic_id}: {name}")
        print(f"  Keywords: {', '.join(top_words[:min(5, NUM_TOP_WORDS)])}")
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

