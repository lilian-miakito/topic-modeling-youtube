"""
Utility functions for topic modeling.
"""


def clean_text(text: str) -> str:
    """
    Clean text by removing problematic Unicode characters.
    Handles line separators, paragraph separators, and other control chars.
    """
    if not text:
        return ""
    
    # Remove Unicode line/paragraph separators and other problematic chars
    text = text.replace("\u2028", " ")  # Line Separator
    text = text.replace("\u2029", " ")  # Paragraph Separator
    text = text.replace("\u0085", " ")  # Next Line
    text = text.replace("\u000b", " ")  # Vertical Tab
    text = text.replace("\u000c", " ")  # Form Feed
    text = text.replace("\r\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    
    # Collapse multiple spaces
    while "  " in text:
        text = text.replace("  ", " ")
    
    return text.strip()

