# AI-Powered Credit Scoring Dashboard

This project implements a Streamlit-based dashboard for analyzing financial data and generating credit scores using AI.  It leverages Google's Gemini API for natural language processing and data analysis.

## Features

* **Financial Data Upload:** Accepts various file formats (PDF, CSV, XLS, XLSX) containing financial statements.  PDFs are processed using PyPDF2 and then structured using the Gemini API.
* **AI-Driven Analysis:**  Utilizes the Gemini API to:
    * Extract key financial metrics (total credits, total debits, average balance, transaction count).
    * Assess risk factors (income stability, debt management, payment behavior, financial ratios, account history).
    * Generate a detailed credit analysis report including strengths, weaknesses, recommendations, and potential red flags.
* **Credit Score Calculation:** A credit score is calculated based on the extracted financial metrics and risk factors, incorporating weighted averages and adjustments based on AI insights.
* **Interactive Dashboard:** Presents the credit score, rating, transaction analysis, financial summary, risk factor analysis, and detailed analysis in an intuitive Streamlit dashboard.
* **Visualization:**  Includes a Plotly chart visualizing credit and debit trends over time and a polar chart showing risk factor levels.
* **Previous Analyses:**  Stores and allows access to previous analyses for comparison.
* **AI Chatbot Integration (Optional):**  Integrates a Watson Assistant chatbot (requires configuration) for additional support and interaction.

## Technologies Used

* **Python:** The primary programming language.
* **Streamlit:**  For building the interactive web dashboard.
* **Pandas:** For data manipulation and analysis.
* **Plotly:** For creating interactive charts.
* **PyPDF2:** For extracting text from PDF files.
* **Google Generative AI (Gemini):** For natural language processing and analysis.
* **Seaborn & Matplotlib:** For data visualization.
* **Watson Assistant (Optional):** For chatbot integration.


## Setup

1. **Install Dependencies:**

```bash
pip install -r requirements.txt
```

2. **Google Cloud Setup:**
   * Create a Google Cloud project.
   * Enable the Generative AI API.
   * Obtain an API key.
   * Set the `GOOGLE_API_KEY` environment variable:

     ```bash
     export GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
     ```
     (Alternatively, add it to your `.env` file).

3. **(Optional) Watson Assistant Setup:**
   * Create a Watson Assistant service instance.
   * Obtain the integration ID, region, and service instance ID.
   * Update the `watson_script` within the `display_ai_chat` function with your credentials.

4. **Run the Dashboard:**

```bash
streamlit run app.py
```

## Directory Structure

```
├── dashboard.py             # Main Streamlit application
├── credit_score.py    # AI credit scoring logic
├── markdown_converter.py # Markdown to plain text conversion
├── requirements.txt   # Project dependencies
└── credit_analysis_outputs/ # Directory to store analysis outputs
    ├── data/
    ├── plots/
    ├── reports/
    └── models/
```

## Usage

1. Run the Streamlit app.
2. Upload a financial statement (PDF).
3. The dashboard will display the analysis results, including the credit score, rating, charts, and a detailed report.
4. You can also view previous analyses from the sidebar.

##  Potential Improvements

* **Error Handling:**  More robust error handling could be implemented to gracefully handle various issues (e.g., invalid file formats, API errors).
* **Data Validation:** Add more stringent data validation to ensure the uploaded financial data is in the correct format and contains necessary information.
* **Advanced Risk Assessment:** Incorporate more sophisticated risk assessment models and potentially integrate with external credit bureaus.
* **Customizable Weights:** Allow users to customize the weights assigned to different risk factors.
* **Improved AI Prompt Engineering:** Refine the prompts used with the Gemini API for improved accuracy and clarity in the generated analysis.
* **User Authentication:** Add user authentication to protect sensitive data.


## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.


This README provides a comprehensive overview of the AI-powered credit scoring dashboard.  Remember to replace placeholders like `"YOUR_GOOGLE_API_KEY"` with your actual credentials.
