# Twitter Fake Account Detection

Multi-model machine learning project for detecting fake Twitter accounts using supervised learning algorithms.

## Models

- Single Layer Perceptron (SLP)
- Logistic Regression
- Random Forest
- XGBoost

## Project Structure

```
twitter-fake-account-detection/
├── config/          # Configuration files
├── data/            # Datasets (raw & processed)
├── notebooks/       # EDA, feature engineering, model comparison
├── src/             # Python package (core logic)
├── scripts/         # CLI entry points
├── app/             # Streamlit UI & FastAPI
├── reports/         # Evaluation results & figures
├── tests/           # Unit tests
└── ...
```

## Quick Start

```bash
pip install -r requirements.txt
python -m scripts.run_pipeline
```

## Usage

- **CLI**: `python -m scripts.run_pipeline`
- **Streamlit**: `streamlit run app/streamlit_app.py`
- **API**: `uvicorn app.api:app --reload`

## License

MIT
