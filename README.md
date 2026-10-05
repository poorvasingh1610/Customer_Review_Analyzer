# Customer Review Analyzer

An LLM-based NLP application that analyzes customer reviews and converts unstructured customer feedback into structured insights and actionable business recommendations.

The system performs overall sentiment analysis, customer satisfaction score estimation, aspect extraction, aspect-level sentiment analysis, review summarization, and business action generation using an LLM API.

## Features

- Overall sentiment classification: Positive, Negative, Neutral, and Mixed
- Customer satisfaction score estimation from 1–5
- Aspect extraction from customer reviews
- Aspect-level sentiment analysis
- Supporting evidence extraction for each aspect
- Review summarization
- Business action generation
- Evaluation against original customer ratings
- Aspect-level sentiment aggregation
- Automated business insights and recommendations
- Interactive Streamlit dashboard
- Real-time analysis of new customer reviews

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Groq API
- OpenAI-compatible API
- Streamlit
- PyYAML
- Matplotlib
- python-dotenv

## Project Pipeline

```text
Customer Reviews
       ↓
Data Cleaning
       ↓
Balanced Sampling
       ↓
LLM-Based Review Analysis
       ↓
JSON Validation & Retry
       ↓
Sentiment + Rating Estimation
       ↓
Aspect Extraction
       ↓
Aspect-Level Sentiment + Evidence
       ↓
Evaluation
       ↓
Aspect Aggregation
       ↓
Business Report & Recommendations
       ↓
Streamlit Dashboard
```

## Dataset

The project uses Amazon food customer reviews.

A balanced sample of 75 reviews was used for the experiment, with 15 reviews from each rating category from 1 to 5 stars.


## Evaluation

The LLM predictions were evaluated against the original customer star ratings as a reference signal.

| Metric | Result |
|---|---:|
| Reviews analyzed | 75 |
| Direct sentiment accuracy | 65.3% |
| Rating-to-sentiment accuracy | 73.3% |
| Mean Absolute Error | 0.47 stars |

## Streamlit Application

The application provides three main sections:

- **Analyze a Review** – Performs real-time LLM-based analysis of a customer review.
- **Results** – Displays evaluation metrics, sentiment distribution, aspect statistics, and confusion matrix.
- **Business Report** – Displays automatically generated customer insights and business recommendations.

## Project Structure

```text
Customer_Review_Analyzer/
│
├── app.py
├── main.py
├── prepare_data.py
├── rebuild.py
├── config.yaml
├── prompts.yaml
├── requirements.txt
├── .gitignore
│
├── data/
│   └── sample_reviews.csv
│
└── output/
    ├── results.csv
    ├── evaluation.csv
    ├── aspects.csv
    ├── metrics.txt
    └── report.md
    ```