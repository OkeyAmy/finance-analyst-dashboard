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



import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from plotly.subplots import make_subplots
from credit_score import AICreditScorer
from markdown_converter import MarkdownConverter
import re
from typing import Dict, Optional

class CreditScoringDashboard:
    def __init__(self):
        self.setup_page_config()
        self.setup_styles()
        self.scorer = AICreditScorer(output_dir="credit_analysis_outputs")
        self.markdown_converter = MarkdownConverter()
        
    def setup_page_config(self):
        st.set_page_config(
            page_title="Credit Score Analysis",
            page_icon="💳",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
    def setup_styles(self):
        st.markdown("""
            <style>
                .stat-card {
                    padding: 20px;
                    border-radius: 10px;
                    background: white;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin: 10px 0;
                }
                .metric-big {
                    font-size: 3rem;
                    font-weight: bold;
                    text-align: center;
                }
                .metric-label {
                    font-size: 1.2rem;
                    color: #666;
                    text-align: center;
                }
                .summary-metric {
                    font-size: 1.5rem;
                    font-weight: bold;
                    color: #1E3D59;
                }
                .risk-label {
                    font-weight: bold;
                    margin-right: 10px;
                }
                .section-header {
                    font-size: 1.5rem;
                    color: #1E3D59;
                    margin: 20px 0;
                    font-weight: bold;
                }
                .recommendation-card {
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 10px 0;
                }
                .risk-high { color: #dc3545; }
                .risk-medium { color: #ffc107; }
                .risk-low { color: #28a745; }
            </style>
        """, unsafe_allow_html=True)

    
