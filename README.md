# AI-Enhanced Credit Scoring System

A comprehensive financial analysis tool that combines traditional credit scoring methods with AI-powered insights to provide detailed credit assessments and recommendations.

## 🌟 Features

- **Automated Document Processing**
  - Support for PDF, CSV, Excel financial statements
  - AI-powered text extraction and structuring
  - Automatic data type conversion and validation

- **Advanced Analytics**
  - Credit score calculation using multiple weighted factors
  - Risk analysis across five key dimensions
  - Trend analysis and pattern detection
  - Historical data comparisons

- **Interactive Dashboard**
  - Real-time visualization of financial metrics
  - Transaction analysis graphs
  - Risk factor radar charts
  - Detailed financial summaries

- **AI-Powered Insights**
  - Natural language analysis of financial patterns
  - Personalized recommendations
  - Risk factor identification
  - Actionable improvement strategies

- **Comprehensive Reporting**
  - Detailed PDF reports
  - Interactive visualizations
  - Historical trend analysis
  - Export capabilities in multiple formats

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

Required packages:
- pandas
- numpy
- google-generativeai
- PyPDF2
- streamlit
- plotly
- seaborn
- matplotlib
- python-dotenv

### Environment Setup

1. Create a `.env` file in the root directory
2. Add your Google AI API key:
```
GOOGLE_API_KEY=your_api_key_here
```

### Running the Application

1. Start the Streamlit dashboard:
```bash
streamlit run main.py
```

2. Access the dashboard at `http://localhost:8501`

## 📊 System Architecture

### Core Components

1. **AICreditScorer** (`credit_score.py`)
   - Main scoring engine
   - Document processing
   - AI analysis integration
   - Score calculation

2. **MarkdownConverter** (`markdown_converter.py`)
   - Markdown processing utilities
   - Text formatting
   - Content extraction

3. **CreditScoringDashboard** (`dashboard.py`)
   - Interactive UI
   - Data visualization
   - Report generation
   - User interaction handling

### Scoring Methodology

The system uses a weighted scoring approach across five key factors:

| Factor | Weight |
|--------|---------|
| Income Stability | 25% |
| Debt Management | 25% |
| Payment Behavior | 20% |
| Financial Ratios | 15% |
| Account History | 15% |

Score ranges:
- Excellent: 800-850
- Good: 700-799
- Fair: 600-699
- Poor: 300-599

## 📁 Project Structure

```
credit_scoring_system/
├── credit_analysis_outputs/
│   ├── data/
│   ├── plots/
│   ├── reports/
│   └── models/
├── .env
├── credit_score.py
├── markdown_converter.py
├── dashboard.py
├── requirements.txt
└── README.md
```

## 🔧 Configuration

### Output Directory Structure

- `data/`: Processed CSV files
- `plots/`: Generated visualizations
- `reports/`: Analysis reports
- `models/`: Saved analysis states

### Customization

You can customize the system by modifying:

1. Scoring weights in `AICreditScorer.__init__`
2. Score ranges in `AICreditScorer.__init__`
3. Dashboard styling in `CreditScoringDashboard.setup_styles`
4. Analysis parameters in `calculate_score` method

## 📈 Usage Examples

### Basic Analysis

```python
from credit_score import AICreditScorer

# Initialize scorer
scorer = AICreditScorer()

# Analyze financial data
results = scorer.analyze_financial_data("financial_statement.pdf")

# Access results
print(f"Credit Score: {results['credit_score']}")
print(f"Rating: {results['rating']}")
print(f"Risk Factors: {results['risk_factors']}")
```

### Dashboard Usage

```python
from dashboard import CreditScoringDashboard

# Initialize and run dashboard
dashboard = CreditScoringDashboard()
dashboard.run()
```

## 🛡️ Security Considerations

- API keys are stored in environment variables
- Temporary files are automatically cleaned up
- User data is processed locally
- No data persistence without explicit save
- Input validation for all file uploads

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google AI for providing the Gemini model
- Streamlit for the dashboard framework
- Plotly for interactive visualizations
- Seaborn for statistical visualizations
