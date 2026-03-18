# AutoML Lite

A Python web application that lets a user upload a tabular dataset (CSV) and automatically performs data validation, preprocessing, EDA, model training, and evaluation.

## Features
- CSV Data Upload and automatic schema inference.
- Configurable targets and task types (Classification/Regression).
- EDA summaries and plots.
- Machine Learning Model Zoo using scikit-learn.
- HTML Report generation.

## Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Or install via pip with the package: `pip install -e .`

## Running
Run the FastAPI development server:
```bash
uvicorn backend.main:app --reload
```
Navigate to `http://127.0.0.1:8000` in your browser.
