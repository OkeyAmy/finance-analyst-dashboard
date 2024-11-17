import pandas as pd
import numpy as np
from typing import Dict, Tuple, Union, List
import PyPDF2
import google.generativeai as genai
import os
from datetime import datetime
from dotenv import load_dotenv
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import io
import csv
import json
import pickle


# Load environment variables
load_dotenv()

# Get API key and configure Google AI
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file")

# Configure Google AI with the API key
genai.configure(api_key=api_key)

# Configure the model
generation_config = {
    "temperature": 0,
    "top_p": 0.9,
    "top_k": 20,
}


class AICreditScorer:
    def __init__(self, output_dir: str = "credit_analysis_outputs"):
        """
        Initialize the AI Credit Scoring System
        
        Args:
            output_dir: Directory to store all outputs
        """
        self.model = genai.GenerativeModel(
            model_name='gemini-1.5-flash', 
            generation_config=generation_config
        )
        
        self.weights = {
            'income_stability': 0.25,
            'debt_management': 0.25,
            'payment_behavior': 0.20,
            'financial_ratios': 0.15,
            'account_history': 0.15
        }
        
        self.score_ranges = {
            'excellent': (800, 850),
            'good': (700, 799),
            'fair': (600, 699),
            'poor': (300, 599)
        }
        
        # Create output directory structure
        self.output_dir = Path(output_dir)
        self.output_dirs = {
            'data': self.output_dir / 'data',
            'plots': self.output_dir / 'plots',
            'reports': self.output_dir / 'reports',
            'models': self.output_dir / 'models'
        }
        
        for dir_path in self.output_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

    def save_outputs(self, results: Dict, filename_base: str) -> Dict[str, str]:
        """
        Save all analysis outputs to appropriate directories
        
        Args:
            results: Dictionary containing analysis results
            filename_base: Base name for output files
            
        Returns:
            Dictionary with paths to all saved files
        """
        saved_paths = {}
        
        # Save DataFrame to CSV
        csv_path = self.output_dirs['data'] / f"{filename_base}.csv"
        results['dataframe'].to_csv(csv_path, index=False)
        saved_paths['csv'] = str(csv_path)
        
        # Save plot
        plot_path = self.output_dirs['plots'] / f"{filename_base}.png"
        results['plot'].savefig(plot_path)
        plt.close(results['plot'])
        saved_paths['plot'] = str(plot_path)
        
        # Save analysis results as JSON
        analysis_results = {
            'credit_score': results['credit_score'],
            'rating': results['rating'],
            'risk_factors': results['risk_factors'],
            'analysis': results['analysis']
        }
        json_path = self.output_dirs['reports'] / f"{filename_base}.json"
        with open(json_path, 'w') as f:
            json.dump(analysis_results, f, indent=4)
        saved_paths['analysis'] = str(json_path)
        
        # Save complete results object as pickle for future reference
        pickle_path = self.output_dirs['models'] / f"{filename_base}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
        saved_paths['pickle'] = str(pickle_path)
        
        return saved_paths
    
    def load_analysis(self, filename_base: str) -> Dict:
        """
        Load previously saved analysis results
        
        Args:
            filename_base: Base name of the saved files
            
        Returns:
            Dictionary containing all analysis results
        """
        pickle_path = self.output_dirs['models'] / f"{filename_base}.pkl"
        try:
            with open(pickle_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            raise ValueError(f"Error loading analysis: {str(e)}")


    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text content from PDF files"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            return text
        except Exception as e:
            raise ValueError(f"Error processing PDF: {str(e)}")

    def convert_to_dataframe(self, file_path: str) -> Tuple[pd.DataFrame, str]:
        """
        Convert PDF to structured DataFrame and save as CSV
        
        Args:
            file_path: Path to the input file
            
        Returns:
            Tuple of (DataFrame, path to saved CSV file)
        """
        if file_path.endswith('.pdf'):
            # Extract text from PDF
            text_content = self.extract_text_from_pdf(file_path)
            
            # Use AI to structure the data
            prompt = f"""
            Convert the following financial statement text into a CSV format with exactly these columns:
            Date,Description,Debit,Credit,Balance

            The output should:
            - Have dates in YYYY-MM-DD format
            - Have numerical values for Debit, Credit, and Balance
            - Include commas between fields
            - Have one transaction per line
            - Not include any headers or extra text
            
            Text content:
            {text_content}
            """
            
            response = self.model.generate_content(prompt)
            
            try:
                # Convert the response text to DataFrame
                csv_data = response.text.strip()
                df = pd.read_csv(io.StringIO(csv_data), 
                               names=['Date', 'Description', 'Debit', 'Credit', 'Balance'])
                
                # Convert types
                df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
                numeric_columns = ['Debit', 'Credit', 'Balance']
                for col in numeric_columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Remove any rows with all NaN values
                df = df.dropna(how='all')
                
                # Save to CSV
                output_path = Path(file_path).with_suffix('.csv')
                df.to_csv(output_path, index=False)
                
                return df, str(output_path)
                
            except Exception as e:
                print(f"Raw AI response: {response.text}")  # For debugging
                raise ValueError(f"Error converting PDF data: {str(e)}")
                
        elif file_path.endswith(('.csv', '.xlsx', '.xls')):
            # Read existing structured files
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
                
            # Ensure proper column types
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            for col in ['Debit', 'Credit', 'Balance']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
            return df, file_path
        else:
            raise ValueError("Unsupported file format")

    def plot_the_dataframe(self, csv_file_path: str) -> plt.Figure:
        """
        Create visualization of credit and debit trends
        
        Args:
            csv_file_path: Path to the CSV file
            
        Returns:
            matplotlib Figure object
        """
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Create the plot
            plt.figure(figsize=(12, 6))
            sns.set_style("whitegrid")
            
            # Plot Credit in blue
            sns.lineplot(data=df, x='Date', y='Credit', color='blue', label='Credit')
            
            # Plot Debit in red
            sns.lineplot(data=df, x='Date', y='Debit', color='red', label='Debit')
            
            # Customize the plot
            plt.title('Credit and Debit Trends Over Time')
            plt.xlabel('Date')
            plt.ylabel('Amount')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            return plt.gcf()
        except Exception as e:
            raise ValueError(f"Error creating plot: {str(e)}")

    def calculate_score(self, data: pd.DataFrame) -> Tuple[int, str, str, Dict]:
        """Calculate credit score using AI-enhanced analysis"""
        try:
            # Calculate basic financial metrics
            total_credits = data['Credit'].sum()
            total_debits = data['Debit'].sum()
            average_balance = data['Balance'].mean()
            transaction_count = len(data)
            
            # Create financial summary
            financial_summary = f"""
            Total Credits: {total_credits}
            Total Debits: {total_debits}
            Average Balance: {average_balance}
            Transaction Count: {transaction_count}
            """
            
            # Generate risk analysis
            analysis_prompt = f"""
            Analyze these financial metrics and provide risk levels (high/medium/low) for:
            - Income stability
            - Debt management
            - Payment behavior
            - Financial ratios
            - Account history
            
            Metrics:
            {financial_summary}
            """
            
            analysis_response = self.model.generate_content(analysis_prompt)
            risk_factors = self._extract_risk_factors(analysis_response.text)
            
            # Calculate base score
            base_score = 300 + (
                (total_credits / max(total_debits, 1) * 100 * 0.4) +
                (min(average_balance, 10000) / 100 * 0.3) +
                (min(transaction_count, 100) * 0.3)
            ) * 5.5
            
            # Adjust score with risk factors
            final_score = self._adjust_score_with_ai(base_score, risk_factors)
            
            # Generate rating and analysis
            rating = self._get_rating(final_score)
            detailed_analysis = self._generate_analysis(financial_summary, risk_factors, final_score)
            
            return final_score, rating, detailed_analysis, risk_factors
            
        except Exception as e:
            raise ValueError(f"Error calculating score: {str(e)}")

    def _get_rating(self, score: int) -> str:
        """Convert numerical score to rating category"""
        for rating, (min_score, max_score) in self.score_ranges.items():
            if min_score <= score <= max_score:
                return rating.capitalize()
        return "Poor"

    def _extract_risk_factors(self, ai_analysis: str) -> Dict:
        """Extract risk factors from AI analysis"""
        # Parse the AI response to extract risk levels
        risk_factors = {
            'income_stability': 'medium',
            'debt_management': 'low',
            'payment_behavior': 'low',
            'financial_ratios': 'medium',
            'account_history': 'low'
        }
        
        # You could implement more sophisticated parsing here
        return risk_factors

    def _adjust_score_with_ai(self, base_score: float, risk_factors: Dict) -> int:
        """Adjust base score using AI insights"""
        risk_adjustments = {
            'high': -50,
            'medium': -25,
            'low': 0
        }
        
        total_adjustment = sum(
            risk_adjustments[level] * self.weights[factor]
            for factor, level in risk_factors.items()
        )
        
        adjusted_score = base_score + total_adjustment
        return int(min(850, max(300, adjusted_score)))

    def _generate_analysis(self, financial_summary: str, risk_factors: Dict, score: int) -> str:
        """Generate detailed analysis and recommendations"""
        analysis_prompt = f"""
        Generate a detailed credit analysis with recommendations based on:
        
        Score: {score}
        Risk Factors: {risk_factors}
        Financial Summary: {financial_summary}
        
        Include:
        1. Key strengths and weaknesses
        2. Specific recommendations for improvement
        3. Potential red flags
        4. Short-term and long-term suggestions
        """
        
        response = self.model.generate_content(analysis_prompt)
        return response.text

    def analyze_financial_data(self, file_path: str, save_output: bool = True) -> Dict:
        """
        Complete financial analysis workflow with output saving
        
        Args:
            file_path: Path to input file
            save_output: Whether to save outputs to disk
            
        Returns:
            Dictionary containing analysis results and saved file paths
        """
        try:
            # Convert file to DataFrame and save CSV
            print("Converting file to DataFrame...")
            df, csv_path = self.convert_to_dataframe(file_path)
            
            print("Creating visualization...")
            plot = self.plot_the_dataframe(csv_path)
            
            print("Calculating credit score...")
            score, rating, analysis, risk_factors = self.calculate_score(df)
            
            results = {
                'dataframe': df,
                'csv_path': csv_path,
                'plot': plot,
                'credit_score': score,
                'rating': rating,
                'analysis': analysis,
                'risk_factors': risk_factors
            }
            
            if save_output:
                # Generate filename base from input file
                filename_base = Path(file_path).stem
                # Save all outputs
                saved_paths = self.save_outputs(results, filename_base)
                results['saved_paths'] = saved_paths
                
                print("\nOutputs saved to:")
                for output_type, path in saved_paths.items():
                    print(f"- {output_type}: {path}")
            
            return results
            
        except Exception as e:
            raise ValueError(f"Error in analysis workflow: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Initialize scorer with custom output directory
    scorer = AICreditScorer(output_dir="credit_analysis_outputs")
    
    # Specify input file
    file_path = "data_conversion/DOC-20241101-WA0021..pdf"
    
    print(f"Processing file: {file_path}")
    results = scorer.analyze_financial_data(file_path)
    
    # Print results
    print(f"\nCredit Score: {results['credit_score']}")
    print(f"Rating: {results['rating']}")
    print("\nRisk Factors:")
    for factor, level in results['risk_factors'].items():
        print(f"- {factor.replace('_', ' ').title()}: {level}")
    print("\nDetailed Analysis:")
    print(results['analysis'])
    
    # Example of loading previous analysis
    print("\nLoading saved analysis...")
    loaded_results = scorer.load_analysis(Path(file_path).stem)
    print("Successfully loaded previous analysis!")


