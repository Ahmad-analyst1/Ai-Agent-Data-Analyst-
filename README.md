# 🤖 AI Data Analyst Agent

A GitHub-deployable Streamlit AI Data Analyst.

## Features

- CSV and Excel upload
- Dataset profiling
- Descriptive statistics
- Missing-value analysis
- Duplicate detection
- IQR outlier detection
- Interactive Plotly charts
- Correlation analysis
- Automatic insights
- Natural-language AI data Q&A
- CSV export

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## OpenAI setup

For AI Q&A, add this to Streamlit Cloud Secrets:

```toml
OPENAI_API_KEY = "your-api-key"
```

Do not commit API keys to GitHub.

## Streamlit deployment

1. Push this folder to GitHub.
2. Open Streamlit Community Cloud.
3. Select the repository.
4. Select `app.py`.
5. Deploy.
