import streamlit as st
import pandas as pd
import re
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from ai.credit_scorer import AICreditScorer
from markdown.markdown_converter import MarkdownConverter
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
import seaborn as sns

class CreditScoringUI:
    def __init__(self):
        # Set page configuration
        st.set_page_config(page_title="AI Credit Scoring", page_icon="💳", layout="wide")
        
        # Initialize session state
        if 'page' not in st.session_state:
            st.session_state.page = 'upload'
        
        # Store all previous analysis results
        if 'previous_analyses' not in st.session_state:
            st.session_state.previous_analyses = []
        
        # Current analysis result
        if 'current_analysis' not in st.session_state:
            st.session_state.current_analysis = None

    def save_file(self, uploaded_file, directory):
        """Save uploaded file and return the path"""
        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = save_dir / f"{timestamp}_{uploaded_file.name}"
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        return str(file_path)

    def create_sidebar(self):
        """Create a universal sidebar for downloads"""
        with st.sidebar:
            st.header("📥 Previous Analyses")
            
            if st.session_state.previous_analyses:
                for idx, analysis in enumerate(st.session_state.previous_analyses, 1):
                    with st.expander(f"Analysis {idx}: {analysis['business_name']}"):
                        # Business Purpose Report
                        business_purpose_text = MarkdownConverter.remove_markdown(
                            analysis['processed_purpose']
                        )
                        analysis_text = MarkdownConverter.remove_markdown(analysis['analysis'])

                        st.download_button(
                            label="Business Purpose Report",
                            data=business_purpose_text,
                            file_name=f"business_purpose_report_{analysis['business_name']}.txt",
                            mime="text/plain",
                            key=f"download_purpose_{idx}"
                        )
                        
                        # Detailed Analysis Report
                        st.download_button(
                            label="Detailed Analysis Report",
                            data=analysis_text,
                            file_name=f"financial_analysis_{analysis['business_name']}.txt",
                            mime="text/plain",
                            key=f"download_analysis_{idx}"
                        )

                        # Additional reports from saved_paths
                        if 'saved_paths' in analysis:
                            for output_type, path in analysis['saved_paths'].items():
                                # Skip business purpose and analysis reports as they're handled above
                                if output_type not in ['business_purpose', 'analysis']:
                                    if os.path.exists(path):
                                        with open(path, 'rb') as f:
                                            st.download_button(
                                                label=f"Download {output_type.title()} Report",
                                                data=f,
                                                file_name=os.path.basename(path),
                                                mime='application/octet-stream',
                                                key=f"download_{output_type}_{idx}"
                                            )
            else:
                st.info("No previous analyses available.")

    def upload_page(self):
        # Create sidebar for downloads
        self.create_sidebar()
        
        st.title("🏦 AI Credit Scoring System for Small Businesses")
        st.markdown("""
            This system analyzes your business's financial data and provides a comprehensive credit assessment
            using advanced AI algorithms.
        """)

        with st.form("upload_form", clear_on_submit=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Business Information")
                business_name = st.text_input("Business Name")
                financial_doc = st.file_uploader(
                    "Upload Financial Document", 
                    type=['pdf', 'csv']
                )

            with col2:
                st.subheader("Business Model Recording")
                st.markdown("Please Provide Your Business Model Description and Financial Data for Analysis (Please be detailed in your business model).")
                audio_file = st.audio_input("Record a voice message")

            submit_button = st.form_submit_button("Analyze Business")

            if submit_button and business_name and financial_doc and audio_file:
                scorer = AICreditScorer()
                
                # Save files
                financial_path = self.save_file(financial_doc, "temp/financial_docs")
                audio_path = self.save_file(audio_file, "temp/audio_files")

                with st.spinner("Analyzing business data...\nWait while we analyze your data."):
                    try:
                        # Perform analysis
                        results = scorer.analyze_financial_data(
                            file_path=financial_path,
                            business_name=business_name,
                            audio_purpose_path=audio_path
                        )

                        # Store current analysis
                        st.session_state.current_analysis = results
                        
                        # Add to previous analyses
                        st.session_state.previous_analyses.append(results)

                        # Navigate to results page
                        st.session_state.page = 'results'
                        st.rerun()
                    
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
                
                # Clean up temporary files
                os.remove(financial_path)
                os.remove(audio_path)

    def results_page(self):
        # Create sidebar for downloads
        self.create_sidebar()
        
        # Verify analysis results exist
        if not st.session_state.current_analysis:
            st.error("No analysis results available.")
            if st.button("Back to Upload"):
                st.session_state.page = 'upload'
            return

        results = st.session_state.current_analysis

        # Main results display
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.title("🏦 Credit Analysis Results")

        with col2:
            if st.button("Back to Upload"):
                st.session_state.page = 'upload'

        # Credit Score Section
        self.render_credit_score_section(
            results['credit_score'], 
            results['rating']
        )

        # Risk Analysis
        with st.header("⚠️ Risk Analysis"):
            self.render_risk_analysis_details(results['risk_factors'])

        # Financial Trends
        self.render_financial_trends(results['dataframe'])

        # Detailed Analysis
        self.render_detailed_analysis(results['analysis'])

    def render_credit_score_section(self, score, rating):
        """Render credit score details with explanations"""
        col1, col2 = st.columns(2)
        
        with col1:
            # Gauge chart for credit score
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"Credit Score: {rating}"},
                gauge = {
                    'axis': {'range': [300, 850]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [300, 600], 'color': "red"},
                        {'range': [600, 700], 'color': "yellow"},
                        {'range': [700, 800], 'color': "lightgreen"},
                        {'range': [800, 850], 'color': "green"}
                    ]
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Score Interpretation")
            
            # Detailed score breakdown with explanations
            score_categories = {
                'Poor (300-599)': "High risk. Indicates significant financial challenges. Urgent improvement needed.",
                'Fair (600-699)': "Moderate risk. Some financial improvement opportunities exist.",
                'Good (700-799)': "Low risk. Strong financial health with room for optimization.",
                'Excellent (800-850)': "Very low risk. Outstanding financial management and stability."
            }
            
            # Determine the score category
            score_category = next(
                (cat for cat, desc in score_categories.items() if self.check_score_range(cat, score)), 
                "Unknown"
            )
            
            st.markdown(f"""
                - **Current Score:** {score}
                - **Rating:** {rating}
                - **Category:** {score_category}
                
                **What This Means:**
                {score_categories.get(score_category, "Financial assessment in progress.")}
            """)

    def check_score_range(self, category, score):
        """Helper method to check score ranges"""
        ranges = {
            'Poor (300-599)': (300, 599),
            'Fair (600-699)': (600, 699),
            'Good (700-799)': (700, 799),
            'Excellent (800-850)': (800, 850)
        }
        min_score, max_score = ranges.get(category, (0, 0))
        return min_score <= score <= max_score

    def render_risk_analysis_details(self, risk_factors):
        """Render the risk analysis section"""
        # Color mapping for risk levels
        colors = {
            'low': 'green',
            'medium': 'orange',
            'high': 'red'
        }
        
        # Risk explanations dictionary matching credit_scorer.py factors
        risk_explanations = {
            'cash_flow_stability': "Assesses the ability to maintain positive cash flow and meet financial obligations",
            'payment_history': "Evaluates the consistency and reliability of payment transactions",
            'debt_management': "Analyzes the ability to manage and repay existing debts effectively",
            'business_growth': "Measures the overall business growth trajectory and revenue trends",
            'operating_history': "Evaluates the length and consistency of business operations"
        }
        
        # Columns for risk factors
        cols = st.columns(len(risk_factors))
        
        for col, (factor, level) in zip(cols, risk_factors.items()):
            with col:
                # Metric card with improved styling
                st.markdown(f"""
                    <div style="
                        background-color: #f0f2f6;
                        border-radius: 10px;
                        padding: 15px;
                        margin: 5px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1)
                    ">
                        <h3 style="color: #1f1f1f; margin-bottom: 10px;">
                            {factor.replace('_', ' ').title()}
                        </h3>
                        <p style="
                            color: {colors[level]}; 
                            font-weight: bold; 
                            font-size: 1.2em;
                            margin-bottom: 10px;
                        ">
                            {level.upper()}
                        </p>
                        <p style="
                            font-size: 0.9em;
                            color: #424242;
                            line-height: 1.4;
                        ">
                            {risk_explanations.get(factor, "Assessment based on multiple financial indicators.")}
                        </p>
                    </div>
                """, unsafe_allow_html=True)

    def create_transaction_plot(self, df: pd.DataFrame) -> go.Figure:
        """Create a subplot figure with credit/debit trends and balance history"""
        # Sort data by date
        df = df.sort_values(by='Date')

        # Handle missing values by forward filling
        df[['Credit', 'Debit', 'Balance']] = df[['Credit', 'Debit', 'Balance']].fillna(method='ffill')

        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=('Credit/Debit Trends', 'Balance History'),
                           vertical_spacing=0.15)

        fig.add_trace(
            go.Scatter(
                x=df['Date'], y=df['Credit'],
                name='Credit', line=dict(color='#28a745'),
                connectgaps=True
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df['Date'], y=df['Debit'],
                name='Debit', line=dict(color='#dc3545'),
                connectgaps=True
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['Date'], y=df['Balance'],
                name='Balance', line=dict(color='#17a2b8'),
                connectgaps=True
            ),
            row=2, col=1
        )

        fig.update_layout(
            height=600,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=1.1,
                xanchor="left",
                x=0
            ),
            margin=dict(t=60, b=40, l=40, r=40)
        )

        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Amount", row=1, col=1)
        fig.update_yaxes(title_text="Balance", row=2, col=1)

        return fig

    def render_financial_trends(self, df):
        """Render financial trends visualization"""
        st.header("📈 Financial Trends")
        
        # Create and display the combined plot
        fig = self.create_transaction_plot(df)
        st.plotly_chart(fig, use_container_width=True)

    def render_detailed_analysis(self, analysis):
            """Render the detailed analysis section"""
            st.header("📑 Detailed Analysis")
            st.markdown(analysis)
    
    def main(self):
        if st.session_state.page == 'upload':
            self.upload_page()
        else:
            self.results_page()

def main():
    app = CreditScoringUI()
    app.main()

if __name__ == "__main__":
    main()


