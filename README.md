# Twitter Fake Account Detection

Multi-model machine learning project for detecting fake Twitter accounts using supervised learning algorithms. Built with scikit-learn and XGBoost, featuring a Streamlit dashboard and FastAPI endpoint.

## Models

| Model | Category | Library |
|-------|----------|---------|
| Single Layer Perceptron (SLP) | Neural Network | sklearn.neural_network |
| Logistic Regression | Linear Model | sklearn.linear_model |
| Random Forest | Ensemble (Bagging) | sklearn.ensemble |
| XGBoost | Ensemble (Boosting) | xgboost |

## Performance

| Model | Test Accuracy | Precision | Recall | F1-score | ROC AUC |
|-------|:------------:|:---------:|:------:|:--------:|:-------:|
| SLP | 93.43% | 0.89 | 0.95 | 0.92 | 0.979 |
| Logistic Regression | 93.55% | 0.89 | 0.95 | 0.92 | 0.981 |
| Random Forest | 94.43% | 0.92 | 0.94 | 0.93 | 0.986 |
| XGBoost | **94.78%** | **0.92** | **0.95** | **0.94** | **0.988** |

## Features

11 features after preprocessing and feature engineering:

| Feature | Description |
|---------|------------|
| `followers_count` | Number of followers |
| `friends_count` | Number of friends |
| `post_count` | Number of posts |
| `location_available` | Whether location is provided |
| `lang_encode` | Encoded language code |
| `desc_len` | Description length |
| `account_age_days` | Account age in days |
| `follower_friend_ratio` | Followers / (Friends + 1) |
| `screen_name_numeric_ratio` | Ratio of digits in screen name |
| `url_available` | Description contains URL |
| `description_has_url` | Same (future-proof for profile URL) |

## Project Structure

```
twitter-fake-account-detection/
├── config/          # Configuration (config.yaml)
├── data/
│   ├── raw/         # Raw train/test datasets
│   └── processed/   # Preprocessed & scaled data
├── notebooks/       # EDA, feature engineering, model comparison
│   ├── 01-eda.ipynb
│   ├── 02-feature-engineering.ipynb
│   └── 03-model-comparison.ipynb
├── src/             # Python package (core logic)
│   ├── config.py           # YAML config loader
│   ├── data_loader.py      # CSV/XLSX reader
│   ├── preprocessing.py    # Validation, language mapping, feature creation
│   ├── features.py         # Engineered features (ratio, URL, etc.)
│   ├── scaling.py          # MinMaxScaler wrapper
│   ├── model.py            # Train, save, prepare data
│   ├── evaluation.py       # Metrics, CV, feature weights
│   ├── prediction.py       # Predict, risk levels
│   ├── visualization.py    # Plots (confusion matrix, ROC, etc.)
│   └── pipeline.py         # Pipeline orchestrator + model registry
├── scripts/         # CLI entry points
│   ├── run_pipeline.py     # Full pipeline runner
│   ├── train.py            # Multi-model trainer
│   └── predict.py          # CLI predictor
├── app/             # Web interfaces
│   ├── streamlit_app.py    # Multi-model dashboard (4 tabs)
│   ├── api.py              # FastAPI (single & batch predict)
│   └── schemas.py          # Pydantic models
├── reports/         # Evaluation results & figures
│   └── figures/            # Confusion matrices, ROC curves, etc.
├── models/          # Trained .pkl artifacts
├── tests/           # Unit tests (pytest)
│   ├── conftest.py
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_prediction.py
├── Makefile         # Automation shortcuts
├── pyproject.toml   # Project metadata & dependencies
├── requirements.txt # Dependencies
└── .gitignore
```

## Setup

```bash
# Clone & install
git clone https://github.com/RajendraF1/twitter-fake-account-detection.git
cd twitter-fake-account-detection
pip install -r requirements.txt

# Or using Make
make install
```

## Usage

### CLI — Run full pipeline (train SLP)
```bash
python -m scripts.run_pipeline
# or
make run
```

### CLI — Train all models
```bash
python -m scripts.train
# Train a specific model
python -m scripts.train --model random_forest
# or
make train
```

### CLI — Predict
```bash
python -m scripts.predict --input data/raw/test_data_eps1.xlsx --model xgboost --output hasil.csv
```

### Streamlit Dashboard
```bash
streamlit run app/streamlit_app.py
# or
make streamlit
```
Opens at `http://localhost:8501` with 4 tabs: Dashboard, Model Detail, Single Predict, Batch Predict.

### FastAPI
```bash
uvicorn app.api:app --reload
# or
make api
```
API docs at `http://localhost:8000/docs`
- `POST /predict` — single JSON prediction
- `POST /predict/batch` — file upload prediction

### Tests
```bash
python -m pytest tests/ -v
# or
make test
```

## Tech Stack

- **Python 3.10+**
- **scikit-learn** — SLP, Logistic Regression, Random Forest
- **XGBoost** — Gradient boosting
- **Streamlit** — Dashboard UI
- **FastAPI** — REST API
- **pandas / openpyxl** — Data processing
- **matplotlib / seaborn** — Visualization
- **joblib** — Model serialization
- **PyYAML** — Configuration
- **pytest** — Unit testing

## License

MIT
