import json
from pathlib import Path
from jinja2 import Template

html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>AutoML Lite Report</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 1000px; margin: 0 auto; padding: 20px; }
        h1, h2, h3 { color: #333; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        img { max-width: 100%; height: auto; border: 1px solid #ddd; margin-top: 10px; }
    </style>
</head>
<body>
    <h1>AutoML Lite Report</h1>
    
    <h2>Run Metadata</h2>
    <ul>
        <li><strong>Task Type:</strong> {{ run.task_type }}</li>
        <li><strong>Dataset Name:</strong> {{ run.filename }}</li>
        <li><strong>Target Column:</strong> {{ run.target_column }}</li>
        <li><strong>Status:</strong> {{ run.status }}</li>
    </ul>

    <h2>Leaderboard</h2>
    <table>
        <tr>
            <th>Model</th>
            <th>CV Score ({{ best_model_info.primary_metric }})</th>
            <th>Training Time (s)</th>
        </tr>
        {% for model, metrics in leaderboard.items() %}
        <tr>
            <td>{{ model }}</td>
            <td>{{ "%.4f"|format(metrics.cv_score) }}</td>
            <td>{{ "%.2f"|format(metrics.train_time_seconds) }}</td>
        </tr>
        {% endfor %}
    </table>

    <h2>Best Model details</h2>
    <p><strong>{{ best_model_info.name }}</strong></p>
    <ul>
        <li>Validation CV Score: {{ "%.4f"|format(best_model_info.cv_score) }}</li>
    </ul>
    <h3>Test Set Metrics</h3>
    <ul>
        {% for metric, value in best_model_info.test_metrics.items() %}
        <li>{{ metric }}: {{ "%.4f"|format(value) }}</li>
        {% endfor %}
    </ul>

    <h2>Data Overview & EDA</h2>
    <img src="target_distribution.png" alt="Target Distribution">
    <img src="correlation_heatmap.png" alt="Feature Correlation Heatmap" onerror="this.style.display='none'">
</body>
</html>
"""

def generate_html_report(run_data: dict, leaderboard: dict, best_model_info: dict, run_dir: Path):
    template = Template(html_template)
    html_content = template.render(
        run=run_data,
        leaderboard=leaderboard,
        best_model_info=best_model_info
    )
    report_path = run_dir / 'report.html'
    with open(report_path, 'w') as f:
        f.write(html_content)

def save_metrics_json(leaderboard: dict, best_model_info: dict, run_dir: Path):
    metrics = {
        "leaderboard": leaderboard,
        "best_model": best_model_info
    }
    with open(run_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
