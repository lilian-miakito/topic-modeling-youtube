"""
Utility functions for topic modeling.
"""

from pathlib import Path


def clean_text(text: str) -> str:
    """
    Clean text by removing problematic Unicode characters.
    Handles line separators, paragraph separators, and other control chars.
    """
    if not text:
        return ""
    
    # Remove Unicode line/paragraph separators and other problematic chars
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


def get_latest_file(directory: Path, pattern: str) -> Path | None:
    """
    Find the most recent file matching a glob pattern in a directory.
    
    Args:
        directory: Directory to search in
        pattern: Glob pattern (e.g., "topics_result_*.json")
    
    Returns:
        Path to the most recent file, or None if no files found
    """
    files = list(directory.glob(pattern))
    if not files:
        return None
    
    # Sort by modification time, most recent first
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return files[0]

