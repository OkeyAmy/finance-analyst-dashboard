# import pandas as pd
# import numpy as np
# from typing import Dict, Tuple, Union, List
# import PyPDF2
# import google.generativeai as genai
# import os
# from datetime import datetime
# from dotenv import load_dotenv
# import seaborn as sns
# import matplotlib.pyplot as plt
# from pathlib import Path
# import io
# import csv
# import json
# import pickle
# import re


# # Load environment variables
# load_dotenv()

# # Get API key and configure Google AI
# api_key = os.getenv('GOOGLE_API_KEY')
# if not api_key:
#     raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file")

# # Configure Google AI with the API key
# genai.configure(api_key=api_key)

# # Configure the model
# generation_config = {
#     "temperature": 0,
#     "top_p": 0.9,
#     "top_k": 20,
# }


# class AICreditScorer:
#     def __init__(self, output_dir: str = "credit_analysis_outputs"):
#         """
#         Initialize the AI Credit Scoring System for Small Businesses
        
#         Args:
#             output_dir: Directory to store all outputs
#         """
#         self.model = genai.GenerativeModel(
#             model_name='gemini-1.5-flash', 
#             generation_config=generation_config
#         )
        
#         # Updated weights for small business focus
#         self.weights = {
#             'cash_flow_stability': 0.35,    # Most important for small businesses
#             'payment_history': 0.25,        # Reliability in payments
#             'debt_management': 0.20,        # How well they handle debt
#             'business_growth': 0.15,        # Trend in revenue/transactions
#             'operating_history': 0.10       # Length and consistency of operations
#         }
        
#         self.score_ranges = {
#             'excellent': (800, 850),
#             'good': (700, 799),
#             'fair': (600, 699),
#             'poor': (300, 599)
#         }
        
#         # Create output directory structure
#         self.output_dir = Path(output_dir)
#         self.output_dirs = {
#             'data': self.output_dir / 'data',
#             'plots': self.output_dir / 'plots',
#             'reports': self.output_dir / 'reports',
#             'models': self.output_dir / 'models'
#         }
        
#         for dir_path in self.output_dirs.values():
#             dir_path.mkdir(parents=True, exist_ok=True)

#     def save_outputs(self, results: Dict, filename_base: str) -> Dict[str, str]:
#         """
#         Save all analysis outputs to appropriate directories
        
#         Args:
#             results: Dictionary containing analysis results
#             filename_base: Base name for output files
            
#         Returns:
#             Dictionary with paths to all saved files
#         """
#         saved_paths = {}
        
#         # Save DataFrame to CSV
#         csv_path = self.output_dirs['data'] / f"{filename_base}.csv"
#         results['dataframe'].to_csv(csv_path, index=False)
#         saved_paths['csv'] = str(csv_path)
        
#         # Save plot
#         plot_path = self.output_dirs['plots'] / f"{filename_base}.png"
#         results['plot'].savefig(plot_path)
#         plt.close(results['plot'])
#         saved_paths['plot'] = str(plot_path)
        
#         # Save processed business purpose
#         purpose_path = self.output_dirs['reports'] / f"{filename_base}_business_purpose.txt"
#         with open(purpose_path, 'w', encoding='utf-8') as f:
#             f.write(results['processed_purpose'])
#         saved_paths['business_purpose'] = str(purpose_path)
        
#         # Save analysis results as JSON
#         analysis_results = {
#             'credit_score': results['credit_score'],
#             'rating': results['rating'],
#             'risk_factors': results['risk_factors'],
#             'analysis': results['analysis'],
#             'processed_purpose': results['processed_purpose']
#         }
#         json_path = self.output_dirs['reports'] / f"{filename_base}.json"
#         with open(json_path, 'w', encoding='utf-8') as f:
#             json.dump(analysis_results, f, indent=4, ensure_ascii=False)
#         saved_paths['analysis'] = str(json_path)
        
#         # Save complete results object as pickle for future reference
#         pickle_path = self.output_dirs['models'] / f"{filename_base}.pkl"
#         with open(pickle_path, 'wb') as f:
#             pickle.dump(results, f)
#         saved_paths['pickle'] = str(pickle_path)
        
#         return saved_paths
    
#     def load_analysis(self, filename_base: str) -> Dict:
#         """
#         Load previously saved analysis results
        
#         Args:
#             filename_base: Base name of the saved files
            
#         Returns:
#             Dictionary containing all analysis results
#         """
#         pickle_path = self.output_dirs['models'] / f"{filename_base}.pkl"
#         try:
#             with open(pickle_path, 'rb') as f:
#                 return pickle.load(f)
#         except Exception as e:
#             raise ValueError(f"Error loading analysis: {str(e)}")


#     def extract_text_from_pdf(self, file_path: Union[str, Path]) -> str:
#         """Extract text content from PDF files and format it for CSV parsing"""
#         text = ""
#         try:
#             # Convert to Path object if string
#             pdf_path = Path(file_path) if isinstance(file_path, str) else file_path
            
#             with pdf_path.open('rb') as file:
#                 pdf_reader = PyPDF2.PdfReader(file)
#                 for page in pdf_reader.pages:
#                     page_text = page.extract_text()
#                     # Clean and format the text for CSV parsing
#                     lines = page_text.split('\n')
#                     for line in lines:
#                         # Remove any leading/trailing whitespace
#                         line = line.strip()
#                         if not line:
#                             continue  # Skip empty lines
#                         # Replace multiple spaces or tabs with a single comma
#                         line = re.sub(r'[\s\t]+', ',', line)
#                         text += line + '\n'  # Add newline to separate records

#             return text.strip()

#         except Exception as e:
#             raise ValueError(f"Error processing PDF: {str(e)}")


#     def convert_to_dataframe(self, file_path: Union[str, Path]) -> Tuple[pd.DataFrame, str]:
#         """Convert PDF to structured DataFrame and save as CSV"""
#         # Convert to Path object if string
#         file_path = Path(file_path) if isinstance(file_path, str) else file_path
        
#         if file_path.suffix.lower() == '.pdf':
#             # Extract text from PDF
#             text_content = self.extract_text_from_pdf(file_path)
            
#             prompt = f"""
#             Convert the following financial statement text into a CSV format with exactly these columns:
#             Date,Description,Debit,Credit,Balance. 
#             Make sure to know the start balance before the first transaction.
#             Make sure to add the users currency symbol to the columns.
#             Make sure not exceed the maximum number of columns required.

#             You should also know that PDF files can come in different formats. 
#             Make sure to consider the signs in the transactions to correctly classify it the debit and credit columns.
#             # Make sure to extensively read the text content to understand the transactions and the balance.
#             Make sure you parse it in CSV format.
            
#             The output should:
#             - Have dates in YYYY-MM-DD format
#             - Have numerical values for Debit, Credit, and Balance
#             - Include commas between fields
#             - Have one transaction per line
#             - Not include any headers or extra text
            
#             Text content:
#             {text_content}
#             """
            
#             response = self.model.generate_content(prompt)
            
#             try:
#                 # Convert the response text to DataFrame
#                 csv_data = response.text.strip()
#                 df = pd.read_csv(io.StringIO(csv_data), 
#                                names=['Date', 'Description', 'Debit', 'Credit', 'Balance'])
                
#                 # Convert types
#                 df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
#                 numeric_columns = ['Debit', 'Credit', 'Balance']
#                 for col in numeric_columns:
#                     df[col] = pd.to_numeric(df[col], errors='coerce')
                
#                 # Remove any rows with all NaN values
#                 df = df.dropna(how='all')
                
#                 # Save to CSV
#                 output_path = file_path.with_suffix('.csv')
#                 df.to_csv(output_path, index=False)
                
#                 return df, str(output_path)
                
#             except Exception as e:
#                 print(f"Raw AI response: {response.text}")  # For debugging
#                 raise ValueError(f"Error converting PDF data: {str(e)}")
                
#         elif file_path.suffix.lower() in ('.csv', '.xlsx', '.xls'):
#             # Read existing structured files
#             if file_path.suffix.lower() == '.csv':
#                 df = pd.read_csv(file_path)
#             else:
#                 df = pd.read_excel(file_path)
                
#             # Ensure proper column types
#             if 'Date' in df.columns:
#                 df['Date'] = pd.to_datetime(df['Date'])
#             for col in ['Debit', 'Credit', 'Balance']:
#                 if col in df.columns:
#                     df[col] = pd.to_numeric(df[col], errors='coerce')
                    
#             return df, str(file_path)
#         else:
#             raise ValueError(f"Unsupported file format: {file_path.suffix}")


#     def plot_the_dataframe(self, csv_file_path: str) -> plt.Figure:
#         """
#         Create visualization of credit and debit trends
        
#         Args:
#             csv_file_path: Path to the CSV file
            
#         Returns:
#             matplotlib Figure object
#         """
#         try:
#             # Read the CSV file
#             df = pd.read_csv(csv_file_path)
#             df['Date'] = pd.to_datetime(df['Date'])
            
#             # Create the plot
#             plt.figure(figsize=(12, 6))
#             sns.set_style("whitegrid")
            
#             # Plot Credit in blue
#             sns.lineplot(data=df, x='Date', y='Credit', color='blue', label='Credit')
            
#             # Plot Debit in red
#             sns.lineplot(data=df, x='Date', y='Debit', color='red', label='Debit')
            
#             # Customize the plot
#             plt.title('Credit and Debit Trends Over Time')
#             plt.xlabel('Date')
#             plt.ylabel('Amount')
#             plt.xticks(rotation=45)
#             plt.tight_layout()
            
#             return plt.gcf()
#         except Exception as e:
#             raise ValueError(f"Error creating plot: {str(e)}")

#     def _get_rating(self, score: int) -> str:
#         """
#         Convert numerical score to rating category
        
#         Args:
#             score: Numerical credit score
            
#         Returns:
#             String rating category
#         """
#         if score >= 800:
#             return "Excellent"
#         elif score >= 700:
#             return "Good"
#         elif score >= 600:
#             return "Fair"
#         else:
#             return "Poor"

#     def _calculate_risk_factors(self, cash_ratio: float, growth: float, 
#                               debt_ratio: float, transaction_volume: float,
#                               avg_balance: float) -> Dict:
#         """Calculate risk levels based on small business metrics"""
#         risk_factors = {}
        
#         # Cash Flow Stability
#         if cash_ratio >= 0.2:
#             risk_factors['cash_flow_stability'] = 'low'
#         elif cash_ratio >= 0.1:
#             risk_factors['cash_flow_stability'] = 'medium'
#         else:
#             risk_factors['cash_flow_stability'] = 'high'
            
#         # Business Growth
#         if growth >= 0.05:
#             risk_factors['business_growth'] = 'low'
#         elif growth >= 0:
#             risk_factors['business_growth'] = 'medium'
#         else:
#             risk_factors['business_growth'] = 'high'
            
#         # Debt Management
#         if debt_ratio <= 0.4:
#             risk_factors['debt_management'] = 'low'
#         elif debt_ratio <= 0.6:
#             risk_factors['debt_management'] = 'medium'
#         else:
#             risk_factors['debt_management'] = 'high'
            
#         # Payment History (based on transaction volume)
#         if transaction_volume >= 50:
#             risk_factors['payment_history'] = 'low'
#         elif transaction_volume >= 20:
#             risk_factors['payment_history'] = 'medium'
#         else:
#             risk_factors['payment_history'] = 'high'
            
#         # Operating History (based on average balance)
#         if avg_balance >= 5000:
#             risk_factors['operating_history'] = 'low'
#         elif avg_balance >= 1000:
#             risk_factors['operating_history'] = 'medium'
#         else:
#             risk_factors['operating_history'] = 'high'
            
#         return risk_factors


#     def _adjust_score_with_ai(self, base_score: float, risk_factors: Dict) -> int:
#         """Adjust base score using risk factors"""
#         risk_adjustments = {
#             'high': -50,
#             'medium': -25,
#             'low': 0
#         }
        
#         total_adjustment = sum(
#             risk_adjustments[level] * self.weights[factor]
#             for factor, level in risk_factors.items()
#         )
        
#         adjusted_score = base_score + total_adjustment
#         return int(min(850, max(300, adjusted_score)))

#     # def _process_audio_purpose(self, business_name: str, purpose: str, file_path: str) -> str:
#     #     """Adjust business purpose to generate personalized recommendations"""
        
#     #     # file = genai.upload_file(media / purpose) # this uploades an Audio file to the system
#     #     file = genai.upload_file(purpose) 
#     def _process_audio_purpose(self, business_name: str, purpose: Union[str, Path], file_path: Union[str, Path]) -> str:
#         """Process audio purpose with proper path handling"""
#         try:
#             # Convert to Path objects if strings
#             purpose_path = Path(purpose) if isinstance(purpose, str) else purpose
#             file_path = Path(file_path) if isinstance(file_path, str) else file_path
#             # text_file = self.extract_text_from_pdf(file_path) 
#             # text_content = self.extract_text_from_pdf(file_path)  

#             file = genai.upload_file(str(purpose_path))
#             restrutured_purpose = f""" 
#         You are an expert financial consultant. Your task is to analyze and extract key information from these business {business_name} using the trasaction document and audio you have:  
#         1. The business model description provided by the user which is this .  
#         2. The financial data, which includes debit, credit, balances, and transaction descriptions.  

#         Your goal is to combine insights from these sources to create a detailed financial and operational overview for the client. Focus on identifying patterns, challenges, and opportunities under the following categories:  

#         Here are the user's business financial data:{file_path}
#         1. Business Overview
#         From the user's input, extract:  
#         - Type of business, its purpose, and its value proposition.  
#         - Primary products or services offered.  
#         - Target customers or markets.  

#         2. Revenue and Income Patterns
#         From financial data and user input:  
#         - Identify primary revenue streams (e.g., sales, services).  
#         - Look for trends in credit entries that indicate income flow (e.g., frequency, consistency).  
#         - Highlight seasonal or irregular patterns in income.  
#         - Are there any dependencies on specific clients, contracts, or markets?
    
#         3. Cost and Expense Patterns
#         From financial data:  
#         - Extract major expense categories from debit entries (e.g., rent, materials, salaries).  
#         - Identify fixed vs. variable costs based on descriptions or recurring transactions.  
#         - Highlight unusual or irregular expenses that could indicate inefficiencies.  
#         - Major Costs: Material costs (rising prices identified as a challenge).
#         - Are there mentions of challenges managing expenses or high overheads?
        
#         4. Cash Flow Management
#         From financial data and user input:  
#         - Analyze balances to assess liquidity over time (e.g., ability to cover short-term obligations).  
#         - Look for patterns in cash flow gaps (e.g., periods with high debits but low credits).  
#         - Note any references to late payments, upfront costs, or other challenges from the user.  
#         - How does the business manage cash inflows and outflows?
#         - Are there any issues with payment cycles (e.g., late customer payments, upfront supplier costs)?
#         - Does the user mention cash reserves or strategies for managing short-term cash gaps? 
        
#         5. Financial Health & Risks
#         Combine insights from both sources to assess:  
#         - Profitability: Compare credits (income) to debits (expenses) over a specific period.  
#         - Debt Levels: Look for transactions or balances indicating loans or liabilities.  
#         - Risk Factors: Identify dependencies, irregularities, or potential financial vulnerabilities. 
#         - Are there plans to expand operations, add products, or target new markets?
#         - What challenges or uncertainties are mentioned regarding scalability or growth? 
        
#         6. Growth and Investment Opportunities  
#         From the user's input and patterns in financial data:  
#         - Note plans for expansion, additional investments, or hiring.  
#         - Assess affordability of growth (e.g., ability to hire, invest in new tools, or scale operations).  
#         - Look for underutilized opportunities based on revenue and expense patterns. 
#         - Is there any mention of profit margins, debt levels, or overall financial stability?
#         - Are there specific financial risks identified, such as dependency on a single customer or competitor pressure?
#         - Does the user mention any risk management strategies, such as insurance or diversification? 
        
#         7. Challenges and Pain Points 
#         From the user and data:  
#         - Highlight operational or financial challenges (e.g., rising costs, inconsistent income).  
#         - Identify areas where financial inefficiencies or risks impact business stability.  
#         - Are there any concerns about competition, customer retention, or resource limitations?
#         - What operational, or market challenges does the business face?

#         8. Insights and Recommendations  
#         Based on the combined analysis:  
#         - Provide tailored advice for improving cash flow, managing costs, and mitigating risks.  
#         - Suggest opportunities for growth, investment, or operational improvement.  
#         - Are there any opportunities for growth, new revenue streams, or cost efficiencies mentioned?
#         - Is there a focus on innovation, partnerships, or adopting new strategies? 


#         identify if the business face competition in the local area?
#         What are the current profit margins?
#         If the user didn't provide all the information in the audio use the financial data to fill in the gaps.
#         Please transcribe the audio and extract the key business purpose informations above
#         """

#             response = self.model.generate_content([file, restrutured_purpose])
#             return response.text
        
#         except Exception as e:
#                 raise ValueError(f"Error processing audio file: {str(e)}")

#     def _generate_small_business_analysis(self, business_name: str, 
#                                         financial_summary: str, 
#                                         risk_factors: Dict, 
#                                         score: int, purpose: str) -> str:
#         """Generate detailed analysis tailored for small businesses"""
#         analysis_prompt = f"""
#         Based this business information from {purpose} return a customized financial analysis report for the business.
#         Make sure to be detailed as much as possible using the information you have here as well {purpose}
#         Generate a detailed business financial analysis for {business_name} based on:

#         Financial Summary: {financial_summary}
#         Overall Credit Score: {score}
#         Risk Assessment: {risk_factors}

#         Provide a detailed report covering:
#         1. Cash Flow Analysis
#            - Daily/monthly cash flow patterns
#            - Suggestions for improving cash flow management
           
#         2. Growth Assessment
#            - Business growth trajectory
#            - Areas for potential expansion
           
#         3. Financial Health Indicators
#            - Key strengths in financial management
#            - Areas needing immediate attention
           
#         4. Practical Recommendations
#            - Short-term actions (next 3 months)
#            - Medium-term improvements (3-12 months)
#            - Long-term strategic suggestions
           
#         5. Risk Mitigation Strategies
#            - Specific steps to address identified risks
#            - Preventive measures for financial stability
           
#         6. Business Opportunities
#            - Potential areas for cost reduction
#            - Revenue enhancement opportunities
           
#         Format the analysis in a clear, actionable way that's useful for a small business owner and informal business owner.
#         Focus on practical, implementable suggestions rather than complex financial terminology.
#         Always recommend the user to seek for more professional advice.
#         Make sure to include the business purpose information extracted from the audio file.
#         You are not limited to the above points, feel free to add more insights that you think will be useful for the business base of all the information you have but organize it properly.
#         """
        
#         response = self.model.generate_content(analysis_prompt)
#         return response.text

#     def calculate_score(self, data: pd.DataFrame, business_name: str, purpose: Union[str, Path], file_path: Union[str, Path], audio_purpose_path: Union[str, Path]) -> Tuple[int, str, str, Dict, str]:
#         """Calculate credit score using small business financial ratios"""
#         try:
#             # Calculate key small business metrics
#             total_revenue = data['Credit'].sum()
#             total_expenses = data['Debit'].sum()
#             net_cash_flow = total_revenue - total_expenses
#             average_daily_balance = data['Balance'].mean()
#             monthly_transactions = len(data) / (data['Date'].max() - data['Date'].min()).days * 30
            
#             # Calculate important financial ratios
#             operating_cash_ratio = net_cash_flow / total_expenses if total_expenses != 0 else 0
#             revenue_growth = (data.groupby(data['Date'].dt.month)['Credit'].sum().pct_change().mean()) if len(data) > 30 else 0
#             debt_service_ratio = total_expenses / total_revenue if total_revenue != 0 else float('inf')            

#             # Financial summary with business name
#             financial_summary = f"""
#             Business Name: {business_name}
            
#             Key Metrics:
#             Total Revenue: {total_revenue:,.2f}
#             Total Expenses: {total_expenses:,.2f}
#             Net Cash Flow: {net_cash_flow:,.2f}
#             Average Daily Balance: {average_daily_balance:,.2f}
#             Monthly Transaction Volume: {monthly_transactions:.0f}
            
#             Financial Ratios:
#             Operating Cash Ratio: {operating_cash_ratio:.2f}
#             Revenue Growth Rate: {revenue_growth:.1%}
#             Debt Service Ratio: {debt_service_ratio:.2f}
#             """
            
#             # Generate risk analysis based on small business metrics
#             risk_factors = self._calculate_risk_factors(
#                 operating_cash_ratio,
#                 revenue_growth,
#                 debt_service_ratio,
#                 monthly_transactions,
#                 average_daily_balance
#             )
            
#             text_file = self.extract_text_from_pdf(file_path)

#             # Calculate base score
#             base_score = 300 + (
#                 (max(min(operating_cash_ratio * 100, 100), 0) * 0.3) +
#                 (max(min(revenue_growth * 100, 50), -50) + 50) * 0.3 +
#                 (max(min((1 - debt_service_ratio) * 100, 100), 0) * 0.4)
#             ) * 5.5
            
#             final_score = self._adjust_score_with_ai(base_score, risk_factors)
#             rating = self._get_rating(final_score)
            
#             # Changed 'file_path=text_file' to 'file_path=file_path'
#             processed_purpose = self._process_audio_purpose(
#                 business_name=business_name, 
#                 purpose=audio_purpose_path, 
#                 file_path=file_path
#             )
#             # print(restrutured_purpose)

#             detailed_analysis = self._generate_small_business_analysis(
#                 business_name,
#                 financial_summary,
#                 risk_factors,
#                 final_score,
#                 processed_purpose
#             )
            
#             return final_score, rating, detailed_analysis, risk_factors, processed_purpose
            
#         except Exception as e:
#             raise ValueError(f"Error calculating score: {str(e)}")

#     def extract_text_from_pdf(self, file_path: Union[str, Path]) -> str:
#         """Extract text content from PDF files and format it for CSV parsing"""
#         text = ""
#         try:
#             # Convert to Path object if string
#             pdf_path = Path(file_path) if isinstance(file_path, str) else file_path
            
#             with pdf_path.open('rb') as file:
#                 pdf_reader = PyPDF2.PdfReader(file)
#                 for page in pdf_reader.pages:
#                     page_text = page.extract_text()
#                     # Clean and format the text for CSV parsing
#                     lines = page_text.split('\n')
#                     for line in lines:
#                         # Remove any leading/trailing whitespace
#                         line = line.strip()
#                         if not line:
#                             continue  # Skip empty lines
#                         # Replace multiple spaces or tabs with a single comma
#                         line = re.sub(r'[\s\t]+', ',', line)
#                         text += line + '\n'  # Add newline to separate records

#             return text.strip()

#         except Exception as e:
#             raise ValueError(f"Error processing PDF: {str(e)}")

#     def calculate_score(self, data: pd.DataFrame, business_name: str, purpose: Union[str, Path], file_path: Union[str, Path], audio_purpose_path: Union[str, Path]) -> Tuple[int, str, str, Dict, str]:
#         """Calculate credit score using small business financial ratios"""
#         try:
#             # Calculate key small business metrics
#             total_revenue = data['Credit'].sum()
#             total_expenses = data['Debit'].sum()
#             net_cash_flow = total_revenue - total_expenses
#             average_daily_balance = data['Balance'].mean()
#             monthly_transactions = len(data) / (data['Date'].max() - data['Date'].min()).days * 30
            
#             # Calculate important financial ratios
#             operating_cash_ratio = net_cash_flow / total_expenses if total_expenses != 0 else 0
#             revenue_growth = (data.groupby(data['Date'].dt.month)['Credit'].sum().pct_change().mean()) if len(data) > 30 else 0
#             debt_service_ratio = total_expenses / total_revenue if total_revenue != 0 else float('inf')        

#             # Financial summary with business name
#             financial_summary = f"""
#             Business Name: {business_name}
            
#             Key Metrics:
#             Total Revenue: {total_revenue:,.2f}
#             Total Expenses: {total_expenses:,.2f}
#             Net Cash Flow: {net_cash_flow:,.2f}
#             Average Daily Balance: {average_daily_balance:,.2f}
#             Monthly Transaction Volume: {monthly_transactions:.0f}
            
#             Financial Ratios:
#             Operating Cash Ratio: {operating_cash_ratio:.2f}
#             Revenue Growth Rate: {revenue_growth:.1%}
#             Debt Service Ratio: {debt_service_ratio:.2f}
#             """
            
#             # Generate risk analysis based on small business metrics
#             risk_factors = self._calculate_risk_factors(
#                 operating_cash_ratio,
#                 revenue_growth,
#                 debt_service_ratio,
#                 monthly_transactions,
#                 average_daily_balance
#             )
            
#             text_file = self.extract_text_from_pdf(file_path)

#             # Calculate base score
#             base_score = 300 + (
#                 (max(min(operating_cash_ratio * 100, 100), 0) * 0.3) +
#                 (max(min(revenue_growth * 100, 50), -50) + 50) * 0.3 +
#                 (max(min((1 - debt_service_ratio) * 100, 100), 0) * 0.4)
#             ) * 5.5
            
#             final_score = self._adjust_score_with_ai(base_score, risk_factors)
#             rating = self._get_rating(final_score)
            
#             # Changed 'file_path=text_file' to 'file_path=file_path'
#             processed_purpose = self._process_audio_purpose(
#                 business_name=business_name, 
#                 purpose=audio_purpose_path, 
#                 file_path=file_path
#             )

#             detailed_analysis = self._generate_small_business_analysis(
#                 business_name,
#                 financial_summary,
#                 risk_factors,
#                 final_score,
#                 processed_purpose
#             )
            
#             return final_score, rating, detailed_analysis, risk_factors, processed_purpose
            
#         except Exception as e:
#             raise ValueError(f"Error calculating score: {str(e)}")


#     def analyze_financial_data(self, file_path: Union[str, Path], 
#                              business_name: str, 
#                              audio_purpose_path: Union[str, Path],
#                              save_output: bool = True) -> Dict:
#         """Complete financial analysis workflow with proper path handling"""
#         try:
#             # Convert to Path objects if strings
#             file_path = Path(file_path) if isinstance(file_path, str) else file_path
#             audio_purpose_path = Path(audio_purpose_path) if isinstance(audio_purpose_path, str) else audio_purpose_path
            
#             print(f"Analyzing data for: {business_name}")
            
#             # Verify files exist
#             if not file_path.exists():
#                 raise FileNotFoundError(f"Financial data file not found: {file_path}")
#             if not audio_purpose_path.exists():
#                 raise FileNotFoundError(f"Audio file not found: {audio_purpose_path}")
            
#             df, csv_path = self.convert_to_dataframe(file_path)
#             plot = self.plot_the_dataframe(csv_path)
#             score, rating, analysis, risk_factors, processed_purpose = self.calculate_score(
#                 data=df,
#                 business_name=business_name,
#                 purpose=audio_purpose_path,
#                 file_path=file_path,
#                 audio_purpose_path=audio_purpose_path
#             )
            
#             results = {
#                 'business_name': business_name,
#                 'dataframe': df,
#                 'csv_path': csv_path,
#                 'plot': plot,
#                 'credit_score': score,
#                 'rating': rating,
#                 'analysis': analysis,
#                 'risk_factors': risk_factors,
#                 'processed_purpose': processed_purpose
#             }
            
#             if save_output:
#                 filename_base = f"{business_name.replace(' ', '_')}_{file_path.stem}"
#                 saved_paths = self.save_outputs(results, filename_base)
#                 results['saved_paths'] = saved_paths
                
#                 print(f"\nAnalysis outputs for {business_name} saved to:")
#                 for output_type, path in saved_paths.items():
#                     print(f"- {output_type}: {path}")
            
#             return results
            
#         except Exception as e:
#             raise ValueError(f"Error in analysis workflow: {str(e)}")

# # Usage example
# if __name__ == "__main__":
#     # File paths
#     file_path = Path(r"C:\Users\Okeoma\Desktop\finance_analyst\DOC-20241101-WA0021..pdf")
#     audio_path = Path(r"C:\Users\Okeoma\OneDrive\Documents\Sound Recordings\Recording.m4a")

#     # Initialize scorer
#     scorer = AICreditScorer()
    
#     # Run analysis
#     results = scorer.analyze_financial_data(
#         file_path=file_path,
#         business_name="Okeoma's Business",
#         audio_purpose_path=audio_path
#     )

#     # Print results
#     print(f"\nBusiness: {results['business_name']}")
#     print(f"Credit Score: {results['credit_score']}")
#     print(f"Rating: {results['rating']}")
#     print("\nRisk Factors:")
#     for factor, level in results['risk_factors'].items():
#         print(f"- {factor.replace('_', ' ').title()}: {level}")
#     print("\nAnalysis:")
#     print(results['analysis'])



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
import re


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
        Initialize the AI Credit Scoring System for Small Businesses
        
        Args:
            output_dir: Directory to store all outputs
        """
        self.model = genai.GenerativeModel(
            model_name='gemini-1.5-flash', 
            generation_config=generation_config
        )
        
        # Updated weights for small business focus
        self.weights = {
            'cash_flow_stability': 0.35,    # Most important for small businesses
            'payment_history': 0.25,        # Reliability in payments
            'debt_management': 0.20,        # How well they handle debt
            'business_growth': 0.15,        # Trend in revenue/transactions
            'operating_history': 0.10       # Length and consistency of operations
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
        
        # Save processed business purpose
        purpose_path = self.output_dirs['reports'] / f"{filename_base}_business_purpose.txt"
        with open(purpose_path, 'w', encoding='utf-8') as f:
            f.write(results['processed_purpose'])
        saved_paths['business_purpose'] = str(purpose_path)
        
        # Save analysis results as JSON
        analysis_results = {
            'credit_score': results['credit_score'],
            'rating': results['rating'],
            'risk_factors': results['risk_factors'],
            'analysis': results['analysis'],
            'processed_purpose': results['processed_purpose']
        }
        json_path = self.output_dirs['reports'] / f"{filename_base}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=4, ensure_ascii=False)
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


    def extract_text_from_pdf(self, file_path: Union[str, Path]) -> str:
        """Extract text content from PDF files and format it for CSV parsing"""
        text = ""
        try:
            # Convert to Path object if string
            pdf_path = Path(file_path) if isinstance(file_path, str) else file_path
            
            with pdf_path.open('rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    # Clean and format the text for CSV parsing
                    lines = page_text.split('\n')
                    for line in lines:
                        # Remove any leading/trailing whitespace
                        line = line.strip()
                        if not line:
                            continue  # Skip empty lines
                        # Replace multiple spaces or tabs with a single comma
                        line = re.sub(r'[\s\t]+', ',', line)
                        text += line + '\n'  # Add newline to separate records

            return text.strip()

        except Exception as e:
            raise ValueError(f"Error processing PDF: {str(e)}")


    def convert_to_dataframe(self, file_path: Union[str, Path]) -> Tuple[pd.DataFrame, str]:
        """Convert PDF to structured DataFrame and save as CSV"""
        # Convert to Path object if string
        file_path = Path(file_path) if isinstance(file_path, str) else file_path
        
        if file_path.suffix.lower() == '.pdf':
            # Extract text from PDF
            text_content = self.extract_text_from_pdf(file_path)
            
            prompt = f"""
            Convert the following financial statement text into a CSV format with exactly these columns:
            Date,Description,Debit,Credit,Balance. 
            Make sure to know the start balance before the first transaction.
            Make sure to add the users currency symbol to the columns.
            Make sure not exceed the maximum number of columns required.

            You should also know that PDF files can come in different formats. 
            Make sure to consider the signs in the transactions to correctly classify it the debit and credit columns.
            # Make sure to extensively read the text content to understand the transactions and the balance.
            Make sure you parse it in CSV format.
            
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
                output_path = file_path.with_suffix('.csv')
                df.to_csv(output_path, index=False)
                
                return df, str(output_path)
                
            except Exception as e:
                print(f"Raw AI response: {response.text}")  # For debugging
                raise ValueError(f"Error converting PDF data: {str(e)}")
                
        elif file_path.suffix.lower() in ('.csv', '.xlsx', '.xls'):
            # Read existing structured files
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
                
            # Ensure proper column types
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            for col in ['Debit', 'Credit', 'Balance']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
            return df, str(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")


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

    def _get_rating(self, score: int) -> str:
        """
        Convert numerical score to rating category
        
        Args:
            score: Numerical credit score
            
        Returns:
            String rating category
        """
        if score >= 800:
            return "Excellent"
        elif score >= 700:
            return "Good"
        elif score >= 600:
            return "Fair"
        else:
            return "Poor"

    def _calculate_risk_factors(self, cash_ratio: float, growth: float, 
                              debt_ratio: float, transaction_volume: float,
                              avg_balance: float) -> Dict:
        """Calculate risk levels based on small business metrics"""
        risk_factors = {}
        
        # Cash Flow Stability
        if cash_ratio >= 0.2:
            risk_factors['cash_flow_stability'] = 'low'
        elif cash_ratio >= 0.1:
            risk_factors['cash_flow_stability'] = 'medium'
        else:
            risk_factors['cash_flow_stability'] = 'high'
            
        # Business Growth
        if growth >= 0.05:
            risk_factors['business_growth'] = 'low'
        elif growth >= 0:
            risk_factors['business_growth'] = 'medium'
        else:
            risk_factors['business_growth'] = 'high'
            
        # Debt Management
        if debt_ratio <= 0.4:
            risk_factors['debt_management'] = 'low'
        elif debt_ratio <= 0.6:
            risk_factors['debt_management'] = 'medium'
        else:
            risk_factors['debt_management'] = 'high'
            
        # Payment History (based on transaction volume)
        if transaction_volume >= 50:
            risk_factors['payment_history'] = 'low'
        elif transaction_volume >= 20:
            risk_factors['payment_history'] = 'medium'
        else:
            risk_factors['payment_history'] = 'high'
            
        # Operating History (based on average balance)
        if avg_balance >= 5000:
            risk_factors['operating_history'] = 'low'
        elif avg_balance >= 1000:
            risk_factors['operating_history'] = 'medium'
        else:
            risk_factors['operating_history'] = 'high'
            
        return risk_factors


    def _adjust_score_with_ai(self, base_score: float, risk_factors: Dict) -> int:
        """Adjust base score using risk factors"""
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

    # def _process_audio_purpose(self, business_name: str, purpose: str, file_path: str) -> str:
    #     """Adjust business purpose to generate personalized recommendations"""
        
    #     # file = genai.upload_file(media / purpose) # this uploades an Audio file to the system
    #     file = genai.upload_file(purpose) 
    def _process_audio_purpose(self, business_name: str, purpose: Union[str, Path], file_path: Union[str, Path]) -> str:
        """Process audio purpose with proper path handling"""
        try:
            # Convert to Path objects if strings
            purpose_path = Path(purpose) if isinstance(purpose, str) else purpose
            file_path = Path(file_path) if isinstance(file_path, str) else file_path
            # text_file = self.extract_text_from_pdf(file_path) 
            # text_content = self.extract_text_from_pdf(file_path)  

            file = genai.upload_file(str(purpose_path))
            restrutured_purpose = f""" 
        You are an expert financial consultant. Your task is to analyze and extract key information from these business {business_name} using the trasaction document and audio you have:  
        1. The business model description provided by the user which is this .  
        2. The financial data, which includes debit, credit, balances, and transaction descriptions.  

        Your goal is to combine insights from these sources to create a detailed financial and operational overview for the client. Focus on identifying patterns, challenges, and opportunities under the following categories:  

        Here are the user's business financial data:{file_path}
        1. Business Overview
        From the user's input, extract:  
        - Type of business, its purpose, and its value proposition.  
        - Primary products or services offered.  
        - Target customers or markets.  

        2. Revenue and Income Patterns
        From financial data and user input:  
        - Identify primary revenue streams (e.g., sales, services).  
        - Look for trends in credit entries that indicate income flow (e.g., frequency, consistency).  
        - Highlight seasonal or irregular patterns in income.  
        - Are there any dependencies on specific clients, contracts, or markets?
    
        3. Cost and Expense Patterns
        From financial data:  
        - Extract major expense categories from debit entries (e.g., rent, materials, salaries).  
        - Identify fixed vs. variable costs based on descriptions or recurring transactions.  
        - Highlight unusual or irregular expenses that could indicate inefficiencies.  
        - Major Costs: Material costs (rising prices identified as a challenge).
        - Are there mentions of challenges managing expenses or high overheads?
        
        4. Cash Flow Management
        From financial data and user input:  
        - Analyze balances to assess liquidity over time (e.g., ability to cover short-term obligations).  
        - Look for patterns in cash flow gaps (e.g., periods with high debits but low credits).  
        - Note any references to late payments, upfront costs, or other challenges from the user.  
        - How does the business manage cash inflows and outflows?
        - Are there any issues with payment cycles (e.g., late customer payments, upfront supplier costs)?
        - Does the user mention cash reserves or strategies for managing short-term cash gaps? 
        
        5. Financial Health & Risks
        Combine insights from both sources to assess:  
        - Profitability: Compare credits (income) to debits (expenses) over a specific period.  
        - Debt Levels: Look for transactions or balances indicating loans or liabilities.  
        - Risk Factors: Identify dependencies, irregularities, or potential financial vulnerabilities. 
        - Are there plans to expand operations, add products, or target new markets?
        - What challenges or uncertainties are mentioned regarding scalability or growth? 
        
        6. Growth and Investment Opportunities  
        From the user's input and patterns in financial data:  
        - Note plans for expansion, additional investments, or hiring.  
        - Assess affordability of growth (e.g., ability to hire, invest in new tools, or scale operations).  
        - Look for underutilized opportunities based on revenue and expense patterns. 
        - Is there any mention of profit margins, debt levels, or overall financial stability?
        - Are there specific financial risks identified, such as dependency on a single customer or competitor pressure?
        - Does the user mention any risk management strategies, such as insurance or diversification? 
        
        7. Challenges and Pain Points 
        From the user and data:  
        - Highlight operational or financial challenges (e.g., rising costs, inconsistent income).  
        - Identify areas where financial inefficiencies or risks impact business stability.  
        - Are there any concerns about competition, customer retention, or resource limitations?
        - What operational, or market challenges does the business face?

        8. Insights and Recommendations  
        Based on the combined analysis:  
        - Provide tailored advice for improving cash flow, managing costs, and mitigating risks.  
        - Suggest opportunities for growth, investment, or operational improvement.  
        - Are there any opportunities for growth, new revenue streams, or cost efficiencies mentioned?
        - Is there a focus on innovation, partnerships, or adopting new strategies? 


        identify if the business face competition in the local area?
        What are the current profit margins?
        If the user didn't provide all the information in the audio use the financial data to fill in the gaps.
        Please transcribe the audio and extract the key business purpose informations above
        """

            response = self.model.generate_content([file, restrutured_purpose])
            return response.text
        
        except Exception as e:
                raise ValueError(f"Error processing audio file: {str(e)}")

    def _generate_small_business_analysis(self, business_name: str, 
                                        financial_summary: str, 
                                        risk_factors: Dict, 
                                        score: int, purpose: str) -> str:
        """Generate detailed analysis tailored for small businesses"""
        analysis_prompt = f"""
        Based this business information from {purpose} return a customized financial analysis report for the business.
        Make sure to be detailed as much as possible using the information you have here as well {purpose}
        Generate a detailed business financial analysis for {business_name} based on:

        Financial Summary: {financial_summary}
        Overall Credit Score: {score}
        Risk Assessment: {risk_factors}

        Provide a detailed report covering:
        1. Cash Flow Analysis
           - Daily/monthly cash flow patterns
           - Suggestions for improving cash flow management
           
        2. Growth Assessment
           - Business growth trajectory
           - Areas for potential expansion
           
        3. Financial Health Indicators
           - Key strengths in financial management
           - Areas needing immediate attention
           
        4. Practical Recommendations
           - Short-term actions (next 3 months)
           - Medium-term improvements (3-12 months)
           - Long-term strategic suggestions
           
        5. Risk Mitigation Strategies
           - Specific steps to address identified risks
           - Preventive measures for financial stability
           
        6. Business Opportunities
           - Potential areas for cost reduction
           - Revenue enhancement opportunities
           
        Format the analysis in a clear, actionable way that's useful for a small business owner and informal business owner.
        Focus on practical, implementable suggestions rather than complex financial terminology.
        Always recommend the user to seek for more professional advice.
        Make sure to include the business purpose information extracted from the audio file.
        You are not limited to the above points, feel free to add more insights that you think will be useful for the business base of all the information you have but organize it properly.
        """
        
        response = self.model.generate_content(analysis_prompt)
        return response.text

    def calculate_score(self, data: pd.DataFrame, business_name: str, purpose: Union[str, Path], file_path: Union[str, Path], audio_purpose_path: Union[str, Path]) -> Tuple[int, str, str, Dict, str]:
        """Calculate credit score using small business financial ratios"""
        try:
            # Calculate key small business metrics
            total_revenue = data['Credit'].sum()
            total_expenses = data['Debit'].sum()
            net_cash_flow = total_revenue - total_expenses
            average_daily_balance = data['Balance'].mean()
            monthly_transactions = len(data) / (data['Date'].max() - data['Date'].min()).days * 30
            
            # Calculate important financial ratios
            operating_cash_ratio = net_cash_flow / total_expenses if total_expenses != 0 else 0
            revenue_growth = (data.groupby(data['Date'].dt.month)['Credit'].sum().pct_change().mean()) if len(data) > 30 else 0
            debt_service_ratio = total_expenses / total_revenue if total_revenue != 0 else float('inf')            

            # Financial summary with business name
            financial_summary = f"""
            Business Name: {business_name}
            
            Key Metrics:
            Total Revenue: {total_revenue:,.2f}
            Total Expenses: {total_expenses:,.2f}
            Net Cash Flow: {net_cash_flow:,.2f}
            Average Daily Balance: {average_daily_balance:,.2f}
            Monthly Transaction Volume: {monthly_transactions:.0f}
            
            Financial Ratios:
            Operating Cash Ratio: {operating_cash_ratio:.2f}
            Revenue Growth Rate: {revenue_growth:.1%}
            Debt Service Ratio: {debt_service_ratio:.2f}
            """
            
            # Generate risk analysis based on small business metrics
            risk_factors = self._calculate_risk_factors(
                operating_cash_ratio,
                revenue_growth,
                debt_service_ratio,
                monthly_transactions,
                average_daily_balance
            )
            
            text_file = self.extract_text_from_pdf(file_path)

            # Calculate base score
            base_score = 300 + (
                (max(min(operating_cash_ratio * 100, 100), 0) * 0.3) +
                (max(min(revenue_growth * 100, 50), -50) + 50) * 0.3 +
                (max(min((1 - debt_service_ratio) * 100, 100), 0) * 0.4)
            ) * 5.5
            
            final_score = self._adjust_score_with_ai(base_score, risk_factors)
            rating = self._get_rating(final_score)
            
            # Changed 'file_path=text_file' to 'file_path=file_path'
            processed_purpose = self._process_audio_purpose(
                business_name=business_name, 
                purpose=audio_purpose_path, 
                file_path=file_path
            )
            # print(restrutured_purpose)

            detailed_analysis = self._generate_small_business_analysis(
                business_name,
                financial_summary,
                risk_factors,
                final_score,
                processed_purpose
            )
            
            return final_score, rating, detailed_analysis, risk_factors, processed_purpose
            
        except Exception as e:
            raise ValueError(f"Error calculating score: {str(e)}")

    def extract_text_from_pdf(self, file_path: Union[str, Path]) -> str:
        """Extract text content from PDF files and format it for CSV parsing"""
        text = ""
        try:
            # Convert to Path object if string
            pdf_path = Path(file_path) if isinstance(file_path, str) else file_path
            
            with pdf_path.open('rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    # Clean and format the text for CSV parsing
                    lines = page_text.split('\n')
                    for line in lines:
                        # Remove any leading/trailing whitespace
                        line = line.strip()
                        if not line:
                            continue  # Skip empty lines
                        # Replace multiple spaces or tabs with a single comma
                        line = re.sub(r'[\s\t]+', ',', line)
                        text += line + '\n'  # Add newline to separate records

            return text.strip()

        except Exception as e:
            raise ValueError(f"Error processing PDF: {str(e)}")

    def calculate_score(self, data: pd.DataFrame, business_name: str, purpose: Union[str, Path], file_path: Union[str, Path], audio_purpose_path: Union[str, Path]) -> Tuple[int, str, str, Dict, str]:
        """Calculate credit score using small business financial ratios"""
        try:
            # Calculate key small business metrics
            total_revenue = data['Credit'].sum()
            total_expenses = data['Debit'].sum()
            net_cash_flow = total_revenue - total_expenses
            average_daily_balance = data['Balance'].mean()
            monthly_transactions = len(data) / (data['Date'].max() - data['Date'].min()).days * 30
            
            # Calculate important financial ratios
            operating_cash_ratio = net_cash_flow / total_expenses if total_expenses != 0 else 0
            revenue_growth = (data.groupby(data['Date'].dt.month)['Credit'].sum().pct_change().mean()) if len(data) > 30 else 0
            debt_service_ratio = total_expenses / total_revenue if total_revenue != 0 else float('inf')        

            # Financial summary with business name
            financial_summary = f"""
            Business Name: {business_name}
            
            Key Metrics:
            Total Revenue: {total_revenue:,.2f}
            Total Expenses: {total_expenses:,.2f}
            Net Cash Flow: {net_cash_flow:,.2f}
            Average Daily Balance: {average_daily_balance:,.2f}
            Monthly Transaction Volume: {monthly_transactions:.0f}
            
            Financial Ratios:
            Operating Cash Ratio: {operating_cash_ratio:.2f}
            Revenue Growth Rate: {revenue_growth:.1%}
            Debt Service Ratio: {debt_service_ratio:.2f}
            """
            
            # Generate risk analysis based on small business metrics
            risk_factors = self._calculate_risk_factors(
                operating_cash_ratio,
                revenue_growth,
                debt_service_ratio,
                monthly_transactions,
                average_daily_balance
            )
            
            text_file = self.extract_text_from_pdf(file_path)

            # Calculate base score
            base_score = 300 + (
                (max(min(operating_cash_ratio * 100, 100), 0) * 0.3) +
                (max(min(revenue_growth * 100, 50), -50) + 50) * 0.3 +
                (max(min((1 - debt_service_ratio) * 100, 100), 0) * 0.4)
            ) * 5.5
            
            final_score = self._adjust_score_with_ai(base_score, risk_factors)
            rating = self._get_rating(final_score)
            
            # Changed 'file_path=text_file' to 'file_path=file_path'
            processed_purpose = self._process_audio_purpose(
                business_name=business_name, 
                purpose=audio_purpose_path, 
                file_path=file_path
            )

            detailed_analysis = self._generate_small_business_analysis(
                business_name,
                financial_summary,
                risk_factors,
                final_score,
                processed_purpose
            )
            
            return final_score, rating, detailed_analysis, risk_factors, processed_purpose
            
        except Exception as e:
            raise ValueError(f"Error calculating score: {str(e)}")


    def analyze_financial_data(self, file_path: Union[str, Path], 
                             business_name: str, 
                             audio_purpose_path: Union[str, Path],
                             save_output: bool = True) -> Dict:
        """Complete financial analysis workflow with proper path handling"""
        try:
            # Convert to Path objects if strings
            file_path = Path(file_path) if isinstance(file_path, str) else file_path
            audio_purpose_path = Path(audio_purpose_path) if isinstance(audio_purpose_path, str) else audio_purpose_path
            
            print(f"Analyzing data for: {business_name}")
            
            # Verify files exist
            if not file_path.exists():
                raise FileNotFoundError(f"Financial data file not found: {file_path}")
            if not audio_purpose_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_purpose_path}")
            
            df, csv_path = self.convert_to_dataframe(file_path)
            plot = self.plot_the_dataframe(csv_path)
            score, rating, analysis, risk_factors, processed_purpose = self.calculate_score(
                data=df,
                business_name=business_name,
                purpose=audio_purpose_path,
                file_path=file_path,
                audio_purpose_path=audio_purpose_path
            )
            
            results = {
                'business_name': business_name,
                'dataframe': df,
                'csv_path': csv_path,
                'plot': plot,
                'credit_score': score,
                'rating': rating,
                'analysis': analysis,
                'risk_factors': risk_factors,
                'processed_purpose': processed_purpose
            }
            
            if save_output:
                filename_base = f"{business_name.replace(' ', '_')}_{file_path.stem}"
                saved_paths = self.save_outputs(results, filename_base)
                results['saved_paths'] = saved_paths
                
                print(f"\nAnalysis outputs for {business_name} saved to:")
                for output_type, path in saved_paths.items():
                    print(f"- {output_type}: {path}")
            
            return results
            
        except Exception as e:
            raise ValueError(f"Error in analysis workflow: {str(e)}")

# Usage example
if __name__ == "__main__":
    # File paths
    file_path = Path(r"C:\Users\Okeoma\Desktop\finance_analyst\DOC-20241101-WA0021..pdf")
    audio_path = Path(r"C:\Users\Okeoma\OneDrive\Documents\Sound Recordings\Recording.m4a")

    # Initialize scorer
    scorer = AICreditScorer()
    
    # Run analysis
    results = scorer.analyze_financial_data(
        file_path=file_path,
        business_name="Okeoma's Business",
        audio_purpose_path=audio_path
    )

    # Print results
    print(f"\nBusiness: {results['business_name']}")
    print(f"Credit Score: {results['credit_score']}")
    print(f"Rating: {results['rating']}")
    print("\nRisk Factors:")
    for factor, level in results['risk_factors'].items():
        print(f"- {factor.replace('_', ' ').title()}: {level}")
    print("\nAnalysis:")
    print(results['analysis'])