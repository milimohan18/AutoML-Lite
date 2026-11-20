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

## Publishing to WordPress
A finished run's HTML report can be published as a WordPress post with the **Publish to WordPress** button on the run page. It uses the WordPress REST API.

1. Start your WordPress site in [LocalWP](https://localwp.com/) and note its site domain (e.g. `http://automl-lite.local`).
2. In WordPress admin, go to **Users > Profile > Application Passwords**, enter a name (e.g. `AutoML Lite`) and click **Add New Application Password**. Copy the password; it is shown only once.
3. Copy `.env.example` to `.env` and fill in your values:
   ```
   WP_SITE_URL=http://automl-lite.local
   WP_USERNAME=your-wordpress-username
   WP_APP_PASSWORD=xxxx xxxx xxxx xxxx xxxx xxxx
   ```
   `.env` is git-ignored, so your credentials are never committed.
4. Restart the server, open a completed run and click **Publish to WordPress**. The post link appears when it's done.

Notes:
- The API endpoint is `POST /api/runs/{run_id}/publish`. Publishing the same run again returns the existing post instead of creating a duplicate.
- Only the report's tables and text are published. The chart images are local files, so they are left out of the post.
- If you get a 404 from WordPress, set **Settings > Permalinks** to anything other than "Plain" so `/wp-json/` URLs work.
