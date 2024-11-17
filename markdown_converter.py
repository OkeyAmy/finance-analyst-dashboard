import re

class MarkdownConverter:
    """Converts markdown text to plain text by removing markdown formatting."""
    
    @staticmethod
    def remove_markdown(text: str) -> str:
        """
        Remove markdown formatting from text.
        
        Args:
            text: Text containing markdown formatting
            
        Returns:
            Plain text with markdown formatting removed
        """
        if not text:
            return ""
            
        # Remove headers (# Header)
        text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
        
        # Remove bold/italic (**text** or *text*)
        text = re.sub(r'\*{1,2}(.*?)\*{1,2}', r'\1', text)
        
        # Remove inline code (`code`)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # Remove code blocks
        text = re.sub(r'```.*?\n(.*?)```', r'\1', text, flags=re.DOTALL)
        
        # Remove bullet points
        text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
        
        # Remove numbered lists
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        
        # Remove blockquotes
        text = re.sub(r'^\s*>\s+', '', text, flags=re.MULTILINE)
        
        # Remove horizontal rules
        text = re.sub(r'\n\s*[-*_]{3,}\s*\n', '\n\n', text)
        
        # Remove links [text](url)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        # Remove extra whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = text.strip()
        
        return text