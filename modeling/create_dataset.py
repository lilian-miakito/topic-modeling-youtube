#!/usr/bin/env python3
"""
Create a dataset from all comments in the data folder.
Outputs a simple text file with one comment per line and a JSON with metadata.
"""

import os
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent / "datasets"


def clean_text(text):
    """Clean text by removing problematic Unicode characters."""
    if not text:
        return ""
    # Remove Unicode line/paragraph separators and other problematic chars
    # U+2028 = Line Separator, U+2029 = Paragraph Separator
    # Also remove other control characters
    text = text.replace('\u2028', ' ')  # Line Separator
    text = text.replace('\u2029', ' ')  # Paragraph Separator
    text = text.replace('\u0085', ' ')  # Next Line
    text = text.replace('\u000b', ' ')  # Vertical Tab
    text = text.replace('\u000c', ' ')  # Form Feed
    text = text.replace('\r\n', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('\n', ' ')
    # Collapse multiple spaces
    while '  ' in text:
        text = text.replace('  ', ' ')
    return text.strip()


def load_all_comments():
    """Load all comments from all JSON files in the data directory."""
    all_comments = []
    
    if not DATA_DIR.exists():
        print(f"Data directory not found: {DATA_DIR}")
        return all_comments
    
    json_files = list(DATA_DIR.glob("*.json"))
    print(f"Found {len(json_files)} JSON files in {DATA_DIR}")
    
    for json_file in json_files:
        print(f"  Loading: {json_file.name}")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            channel_name = data.get('channel_name', 'Unknown')
            videos = data.get('videos', [])
            
            for video in videos:
                video_id = video.get('video_id', '')
                video_title = clean_text(video.get('title', ''))
                comments = video.get('comments', [])
                
                for comment in comments:
                    text = clean_text(comment.get('text', ''))
                    if text:  # Only add non-empty comments
                        all_comments.append({
                            'text': text,
                            'channel': channel_name,
                            'video_id': video_id,
                            'video_title': video_title,
                            'author': clean_text(comment.get('author', '')),
                            'likes': comment.get('likes', 0),
                            'is_reply': comment.get('is_reply', False)
                        })
        except Exception as e:
            print(f"  Error loading {json_file.name}: {e}")
    
    return all_comments


def create_dataset(comments):
    """Create dataset files from comments."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Create a simple text file (one comment per line)
    text_file = OUTPUT_DIR / "comments.txt"
    with open(text_file, 'w', encoding='utf-8', newline='\n') as f:
        for comment in comments:
            # Text is already cleaned, just write it
            f.write(comment['text'] + '\n')
    
    print(f"Created: {text_file} ({len(comments)} lines)")
    
    # Create a JSON file with full metadata
    json_file = OUTPUT_DIR / "comments_full.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(comments, f, ensure_ascii=False, indent=2)
    
    print(f"Created: {json_file}")
    
    # Create a stats summary
    stats = {
        'total_comments': len(comments),
        'unique_channels': len(set(c['channel'] for c in comments)),
        'unique_videos': len(set(c['video_id'] for c in comments)),
        'replies': sum(1 for c in comments if c['is_reply']),
        'top_level_comments': sum(1 for c in comments if not c['is_reply']),
        'avg_length': sum(len(c['text']) for c in comments) / len(comments) if comments else 0,
        'channels': {}
    }
    
    # Per-channel stats
    for comment in comments:
        channel = comment['channel']
        if channel not in stats['channels']:
            stats['channels'][channel] = {'comments': 0, 'videos': set()}
        stats['channels'][channel]['comments'] += 1
        stats['channels'][channel]['videos'].add(comment['video_id'])
    
    # Convert sets to counts
    for channel in stats['channels']:
        stats['channels'][channel]['videos'] = len(stats['channels'][channel]['videos'])
    
    stats_file = OUTPUT_DIR / "stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"Created: {stats_file}")
    
    return stats


def main():
    print("=" * 60)
    print("Creating dataset from YouTube comments")
    print("=" * 60)
    
    # Load all comments
    comments = load_all_comments()
    
    if not comments:
        print("\nNo comments found. Make sure you have JSON files in the data/ folder.")
        return
    
    print(f"\nTotal comments loaded: {len(comments)}")
    
    # Create dataset
    stats = create_dataset(comments)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    print(f"Total comments: {stats['total_comments']:,}")
    print(f"Unique channels: {stats['unique_channels']}")
    print(f"Unique videos: {stats['unique_videos']}")
    print(f"Top-level comments: {stats['top_level_comments']:,}")
    print(f"Replies: {stats['replies']:,}")
    print(f"Average comment length: {stats['avg_length']:.1f} chars")
    
    print("\nPer channel:")
    for channel, data in stats['channels'].items():
        print(f"  - {channel}: {data['comments']:,} comments from {data['videos']} videos")
    
    print(f"\nDataset files created in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

