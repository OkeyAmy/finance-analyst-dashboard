import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from plotly.subplots import make_subplots
from credit_score import AICreditScorer
from markdown_converter import MarkdownConverter
import re
from typing import Dict, Optional
import streamlit.components.v1 as components


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
                .chat-container {
                    position: fixed;
                    bottom: 20px;
                    left: 20px;
                    width: 300px;
                    background: white;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    padding: 15px;
                    z-index: 1000;
                }
            </style>
        """, unsafe_allow_html=True)

    def create_transaction_plot(self, df: pd.DataFrame) -> go.Figure:
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=('Credit/Debit Trends', 'Balance History'),
                           vertical_spacing=0.15)

        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['Credit'],
                      name='Credit', line=dict(color='#28a745')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['Debit'],
                      name='Debit', line=dict(color='#dc3545')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['Balance'],
                      name='Balance', line=dict(color='#17a2b8')),
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

    def extract_sections(self, text: str) -> dict:
        """Extract sections and their content from the analysis text."""
        # First, find all headings and their positions
        heading_pattern = r'^#+\s+(.+)$|^\*\*([^*]+)\*\*:?$'
        sections = {}
        current_heading = None
        current_content = []
        
        for line in text.split('\n'):
            heading_match = re.match(heading_pattern, line.strip())
            if heading_match:
                # Save previous section if it exists
                if current_heading and current_content:
                    sections[current_heading] = '\n'.join(current_content).strip()
                
                # Get new heading (either from # or **)
                current_heading = heading_match.group(1) or heading_match.group(2)
                current_content = []
            else:
                if current_heading:
                    current_content.append(line)
        
        # Don't forget to add the last section
        if current_heading and current_content:
            sections[current_heading] = '\n'.join(current_content).strip()
        
        return sections

    def display_score_header(self, score: int, rating: str):
        st.markdown("# Credit Score Analysis Dashboard")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="metric-label">Credit Score</div>
                    <div class="metric-big">{score}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col2:
            rating_colors = {
                'Excellent': '#28a745',
                'Good': '#17a2b8',
                'Fair': '#ffc107',
                'Poor': '#dc3545'
            }
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="metric-label">Rating</div>
                    <div class="metric-big" style="color: {rating_colors.get(rating, '#666')}">{rating}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    def display_transaction_analysis(self, df: pd.DataFrame):
        st.markdown("### 📈 Transaction Analysis")
        
        fig = self.create_transaction_plot(df)
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="metric-label">Average Credit</div>
                    <div class="summary-metric">{df['Credit'].mean():.2f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with col2:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="metric-label">Average Debit</div>
                    <div class="summary-metric">{df['Debit'].mean():.2f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with col3:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="metric-label">Average Balance</div>
                    <div class="summary-metric">{df['Balance'].mean():.2f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    def display_financial_summary(self, analysis_text: str):
        st.markdown("### 📊 Financial Summary")
        sections = self.extract_sections(analysis_text)
        financial_summary = sections.get("Financial Summary", "")
        
        summary_lines = [line for line in financial_summary.split('\n') 
                        if any(metric in line for metric in 
                              ['Total Credits:', 'Total Debits:', 'Average Balance:', 'Transaction Count:'])]
        
        col1, col2 = st.columns(2)
        
        for i, line in enumerate(summary_lines):
            if ':' in line:
                metric, value = line.split(':')
                value = value.strip()
                with col1 if i < 2 else col2:
                    st.markdown(
                        f"""
                        <div class="stat-card">
                            <div class="metric-label">{metric}</div>
                            <div class="summary-metric">{value}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    def display_risk_analysis(self, risk_factors: Dict):
        st.markdown("### 🎯 Risk Factor Analysis")
        
        categories = list(risk_factors.keys())
        risk_values = {'low': 1, 'medium': 2, 'high': 3}
        values = [risk_values[v] for v in risk_factors.values()]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=[c.replace('_', ' ').title() for c in categories],
            fill='toself',
            line_color='#1E3D59'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 3],
                    ticktext=['', 'Low', 'Medium', 'High'],
                    tickvals=[0, 1, 2, 3],
                )
            ),
            showlegend=False
        )
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            for factor, level in risk_factors.items():
                risk_color = {
                    'high': 'risk-high',
                    'medium': 'risk-medium',
                    'low': 'risk-low'
                }[level]
                st.markdown(
                    f"""
                    <div class="stat-card">
                        <span class="risk-label">{factor.replace('_', ' ').title()}:</span>
                        <span class="{risk_color}">{level.upper()}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    def display_detailed_analysis(self, analysis_text: str):
        st.markdown("### 📝 Detailed Analysis")
        
        tabs = st.tabs([
            "Key Findings",
            "Recommendations",
            "Red Flags",
            "Action Plan"
        ])
        
        # Helper function to extract section content
        def extract_section(start_marker: str, end_marker: str) -> str:
            start_idx = analysis_text.lower().find(start_marker.lower())
            end_idx = analysis_text.lower().find(end_marker.lower())
            
            if start_idx == -1:
                # Try alternate markers without asterisks
                clean_marker = start_marker.replace('*', '')
                start_idx = analysis_text.lower().find(clean_marker.lower())
                
            if end_idx == -1:
                # Try alternate markers without asterisks
                clean_marker = end_marker.replace('*', '')
                end_idx = analysis_text.lower().find(clean_marker.lower())
                
            if start_idx == -1:
                return "Section not found"
                
            if end_idx == -1:
                # If end marker not found, take until the end
                return analysis_text[start_idx:].strip()
                
            return analysis_text[start_idx:end_idx].strip()
        
        # Remove markdown formatting from the analysis text
        clean_text = self.markdown_converter.remove_markdown(analysis_text)
        
        with tabs[0]:
            st.markdown("#### Strengths and Weaknesses")
            
            # Extract strengths and weaknesses sections
            strengths = extract_section(
                "Strengths:", 
                "Weaknesses:"
            ).replace("Strengths:", "").strip()
            weaknesses = extract_section(
                "Weaknesses:", 
                "2. Specific Recommendations"
            ).replace("Weaknesses:", "").strip()
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Weaknesses")
                st.markdown("**")
                st.markdown(weaknesses)
            
            with col2:
                st.markdown("##### Strengths")
                st.markdown("**")
                st.markdown(strengths)
        
        with tabs[1]:
            recommendations = extract_section(
                "2. Specific Recommendations", 
                "3. Potential Red Flags"
            )
            st.markdown(recommendations)
        
        with tabs[2]:
            red_flags = extract_section(
                "3. Potential Red Flags", 
                "4. Short-Term"
            )
            st.markdown(red_flags)
        
        with tabs[3]:
            action_plan = extract_section(
                "4. Short-Term", 
                "Disclaimer:"
            )
            st.markdown(action_plan)

    def display_ai_chat(self):
        # Watson Assistant script
        watson_script = """
        <script>
          window.watsonAssistantChatOptions = {
            integrationID: "2605fdba-a0b1-4918-9de6-9cb51ad371c6", // The ID of this integration.
            region: "jp-tok", // The region your integration is hosted in.
            serviceInstanceID: "a1ff6cfd-4109-4994-a7c5-4b075ba70342", // The ID of your service instance.
            onLoad: async (instance) => { await instance.render(); }
          };
          setTimeout(function(){
            const t=document.createElement('script');
            t.src="https://web-chat.global.assistant.watson.appdomain.cloud/versions/" + (window.watsonAssistantChatOptions.clientVersion || 'latest') + "/WatsonAssistantChatEntry.js";
            document.head.appendChild(t);
          });
        </script>
        """
        # Embed the script in the Streamlit app
        components.html(watson_script, height=600)


    def run(self):
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Select Page", ["New Analysis", "Previous Analyses"])
        
        if page == "New Analysis":
            uploaded_file = st.file_uploader(
                "Upload Financial Statement",
                type=['pdf', 'csv', 'xlsx', 'xls']
            )
            
            if uploaded_file:
                temp_dir = Path("temp_uploads")
                temp_dir.mkdir(exist_ok=True)
                file_path = temp_dir / uploaded_file.name
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                with st.spinner("Analyzing financial data..."):
                    try:
                        results = self.scorer.analyze_financial_data(str(file_path))
                        
                        self.display_score_header(
                            results['credit_score'],
                            results['rating']
                        )
                        
                        self.display_transaction_analysis(results['dataframe'])
                        self.display_financial_summary(results['analysis'])
                        self.display_risk_analysis(results['risk_factors'])
                        self.display_detailed_analysis(results['analysis'])
                        
                        st.success("Analysis completed successfully!")
                        
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
                    finally:
                        file_path.unlink(missing_ok=True)
        
        else:
            st.markdown("### Previous Analyses")
            prev_results = self.scorer.load_previous_analyses()
            if prev_results:
                self.display_score_header(
                    prev_results['credit_score'],
                    prev_results['rating']
                )
                
                self.display_transaction_analysis(prev_results['dataframe'])
                self.display_financial_summary(prev_results['analysis'])
                self.display_risk_analysis(prev_results['risk_factors'])
                self.display_detailed_analysis(prev_results['analysis'])
                # self.display_ai_chat()

        self.display_ai_chat()

if __name__ == "__main__":
    dashboard = CreditScoringDashboard()
    dashboard.run() 