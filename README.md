# Finance Analyst Dashboard

This project implements a Streamlit-based dashboard for analyzing financial data and providing detailed financial and operational overviews for clients. It leverages AI technologies for natural language processing and data analysis.

## Features

* **Financial Data Upload:** Accepts various file formats (PDF) containing financial statements. PDFs are processed and structured for analysis.
* **Business Model Recording:** Allows users to provide a business model description via audio input, which is transcribed and analyzed.
* **AI-Driven Analysis:** Utilizes AI to:
    * Extract key financial metrics (total credits, total debits, average balance, transaction count).
    * Assess risk factors and financial health indicators.
    * Generate a detailed financial analysis report including insights, recommendations, and potential challenges.
* **Visualization:** Provides interactive financial trends visualization using Seaborn and Matplotlib.
* **User-Friendly Interface:** Easy-to-use interface built with Streamlit for uploading data, recording audio, and viewing results.

## Technologies Used

* **Python:** The primary programming language.
* **Streamlit:** For building the interactive web dashboard.
* **Pandas:** For data manipulation and analysis.
* **Plotly:** For creating interactive charts.
* **PyPDF2:** For extracting text from PDF files.
* **Google Generative AI (Gemini):** For natural language processing and analysis.
* **Seaborn & Matplotlib:** For data visualization.

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

3. **Run the Dashboard:**

    ```bash
    streamlit run dashboard.py
    ```

## Directory Structure

```
├── dashboard.py             # Main Streamlit application
├── credit_score.py    # AI credit scoring logic
├── markdown_converter.py # Markdown to plain text conversion
├── requirements.txt   # Project dependencies
└── credit_analysis_outputs/ # Directory to store analysis outputs(directory will be created when you run it locally)
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
* **Partnering with Financial Consultant:** Collaborate with financial consultants to provide expert insights and recommendations, enhancing the overall analysis and ensuring that the generated reports are aligned with industry standards and best practices.

This README provides a comprehensive overview of the AI-powered credit scoring dashboard.  Remember to replace placeholders like `"YOUR_GOOGLE_API_KEY"` with your actual credentials.
