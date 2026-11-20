"""
Analytics router — raw SQL endpoints for BI reporting.

Demonstrates explicit SQL proficiency using sqlalchemy.text():
  - AVG, COUNT, MIN, MAX, SUM, ROUND, CAST
  - GROUP BY, ORDER BY, HAVING
  - CTEs (WITH), subqueries
  - Window functions: RANK() OVER (PARTITION BY ...)
  - CASE WHEN conditional logic
  - Date arithmetic with julianday()
  - JSON extraction with json_extract() and json_each()
  - COALESCE, SUBSTR
  - Formatted Excel export with openpyxl
"""

from io import BytesIO
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.db import get_db

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


# ---------------------------------------------------------------------------
# Query 1 — Global Model Leaderboard: AVG / GROUP BY / ORDER BY
# ---------------------------------------------------------------------------
@router.get("/leaderboard")
def global_leaderboard(db: Session = Depends(get_db)):
    """
    Across all completed runs, which algorithm has the best average score?

    SQL concepts: CTE, json_each, json_extract, CAST, AVG, COUNT, MIN, MAX,
                  ROUND, GROUP BY, ORDER BY
    """
    sql = text("""
        WITH model_scores AS (
            SELECT
                r.id                                                        AS run_id,
                r.task_type,
                j.key                                                       AS model_name,
                CAST(json_extract(j.value, '$.cv_score') AS REAL)           AS cv_score,
                CAST(json_extract(j.value, '$.train_time_seconds') AS REAL) AS train_time
            FROM runs r,
                 json_each(r.metrics_json, '$.leaderboard') AS j
            WHERE r.status = 'COMPLETED'
        )
        SELECT
            model_name,
            COUNT(*)                        AS times_trained,
            ROUND(AVG(cv_score), 4)         AS avg_cv_score,
            ROUND(MIN(cv_score), 4)         AS min_cv_score,
            ROUND(MAX(cv_score), 4)         AS max_cv_score,
            ROUND(AVG(train_time), 2)       AS avg_train_time_sec
        FROM model_scores
        GROUP BY model_name
        ORDER BY avg_cv_score DESC
    """)
    rows = db.execute(sql).mappings().all()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Query 2 — Per-Run Model Ranking: Window Function RANK()
# ---------------------------------------------------------------------------
@router.get("/run-rankings")
def per_run_rankings(db: Session = Depends(get_db)):
    """
    Within each run, rank the models by CV score.

    SQL concepts: CTE, RANK() OVER (PARTITION BY ... ORDER BY ...), window functions
    """
    sql = text("""
        WITH model_scores AS (
            SELECT
                r.id                                                        AS run_id,
                r.task_type,
                j.key                                                       AS model_name,
                CAST(json_extract(j.value, '$.cv_score') AS REAL)           AS cv_score
            FROM runs r,
                 json_each(r.metrics_json, '$.leaderboard') AS j
            WHERE r.status = 'COMPLETED'
        )
        SELECT
            run_id,
            model_name,
            cv_score,
            RANK() OVER (PARTITION BY run_id ORDER BY cv_score DESC) AS rank_in_run
        FROM model_scores
        ORDER BY run_id, rank_in_run
    """)
    rows = db.execute(sql).mappings().all()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Query 3 — Run Duration Analytics: CASE WHEN / julianday
# ---------------------------------------------------------------------------
@router.get("/run-durations")
def run_duration_analytics(db: Session = Depends(get_db)):
    """
    How long do runs take? Classify each run into speed tiers.

    SQL concepts: CASE WHEN, julianday date arithmetic, ROUND, filtering
    """
    sql = text("""
        SELECT
            id,
            task_type,
            status,
            ROUND(
                (julianday(completed_at) - julianday(started_at)) * 86400, 1
            ) AS duration_seconds,
            CASE
                WHEN (julianday(completed_at) - julianday(started_at)) * 86400 > 60
                    THEN 'SLOW'
                WHEN (julianday(completed_at) - julianday(started_at)) * 86400 > 10
                    THEN 'NORMAL'
                ELSE 'FAST'
            END AS speed_tier,
            best_metric_name,
            ROUND(best_metric_score, 4) AS best_score
        FROM runs
        WHERE status = 'COMPLETED'
          AND started_at IS NOT NULL
          AND completed_at IS NOT NULL
        ORDER BY duration_seconds DESC
    """)
    rows = db.execute(sql).mappings().all()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Query 4 — Dataset Profiling Summary: json_extract / COALESCE
# ---------------------------------------------------------------------------
@router.get("/dataset-profiles")
def dataset_profiles(db: Session = Depends(get_db)):
    """
    Quick overview of every dataset processed by the system.

    SQL concepts: json_extract on JSON columns, COALESCE, SUBSTR, ORDER BY
    """
    sql = text("""
        SELECT
            id                                                                AS run_id,
            SUBSTR(filename, -40)                                             AS filename_tail,
            target_column,
            task_type,
            CAST(json_extract(dataset_summary, '$.n_rows') AS INTEGER)        AS n_rows,
            CAST(json_extract(dataset_summary, '$.n_columns') AS INTEGER)     AS n_cols,
            COALESCE(ROUND(best_metric_score, 4), 0)                          AS best_score,
            created_at
        FROM runs
        WHERE status = 'COMPLETED'
        ORDER BY created_at DESC
    """)
    rows = db.execute(sql).mappings().all()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Query 5 — Task-Type Breakdown: GROUP BY / HAVING / conditional SUM
# ---------------------------------------------------------------------------
@router.get("/task-type-summary")
def task_type_summary(db: Session = Depends(get_db)):
    """
    For each task type, show aggregate stats (only if enough data).

    SQL concepts: GROUP BY, HAVING, COUNT, SUM with CASE, AVG, conditional aggregation
    """
    sql = text("""
        SELECT
            task_type,
            COUNT(*)                                                        AS total_runs,
            SUM(CASE WHEN status = 'COMPLETED' THEN 1 ELSE 0 END)          AS completed,
            SUM(CASE WHEN status = 'FAILED' THEN 1 ELSE 0 END)             AS failed,
            ROUND(AVG(best_metric_score), 4)                                AS avg_best_score
        FROM runs
        GROUP BY task_type
        HAVING COUNT(*) >= 1
        ORDER BY total_runs DESC
    """)
    rows = db.execute(sql).mappings().all()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Excel Export — Formatted .xlsx with styled headers and embedded bar chart
# ---------------------------------------------------------------------------
@router.get("/export/excel")
def export_leaderboard_excel(db: Session = Depends(get_db)):
    """
    Exports the global model leaderboard as a professionally formatted .xlsx
    file with styled headers, auto-width columns, and an embedded bar chart.

    Demonstrates: Microsoft Excel proficiency via openpyxl
    """
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.chart import BarChart, Reference
    from openpyxl.utils import get_column_letter

    # Re-use the global leaderboard SQL
    sql = text("""
        WITH model_scores AS (
            SELECT
                j.key                                                       AS model_name,
                CAST(json_extract(j.value, '$.cv_score') AS REAL)           AS cv_score,
                CAST(json_extract(j.value, '$.train_time_seconds') AS REAL) AS train_time
            FROM runs r,
                 json_each(r.metrics_json, '$.leaderboard') AS j
            WHERE r.status = 'COMPLETED'
        )
        SELECT
            model_name,
            COUNT(*)                        AS times_trained,
            ROUND(AVG(cv_score), 4)         AS avg_cv_score,
            ROUND(MIN(cv_score), 4)         AS min_cv_score,
            ROUND(MAX(cv_score), 4)         AS max_cv_score,
            ROUND(AVG(train_time), 2)       AS avg_train_time_sec
        FROM model_scores
        GROUP BY model_name
        ORDER BY avg_cv_score DESC
    """)
    rows = db.execute(sql).mappings().all()

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Model Leaderboard"

    # ── Header styling ──────────────────────────────────────────────
    headers = ["Model", "Times Trained", "Avg CV Score", "Min CV Score",
               "Max CV Score", "Avg Train Time (s)"]
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    header_font = Font(color="FFFFFF", bold=True, size=11, name="Calibri")
    thin_border = Border(
        left=Side(style="thin"), right=Side(style="thin"),
        top=Side(style="thin"), bottom=Side(style="thin")
    )

    for col_idx, header_text in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col_idx, value=header_text)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = thin_border

    # ── Data rows ───────────────────────────────────────────────────
    data_font = Font(size=10, name="Calibri")
    for row_idx, row_data in enumerate(rows, 2):
        values = [
            row_data["model_name"],
            row_data["times_trained"],
            row_data["avg_cv_score"],
            row_data["min_cv_score"],
            row_data["max_cv_score"],
            row_data["avg_train_time_sec"],
        ]
        for col_idx, val in enumerate(values, 1):
            cell = ws.cell(row=row_idx, column=col_idx, value=val)
            cell.font = data_font
            cell.border = thin_border
            if col_idx >= 3:
                cell.number_format = "0.0000" if col_idx <= 5 else "0.00"
                cell.alignment = Alignment(horizontal="center")

    # ── Auto-width columns ──────────────────────────────────────────
    for col_idx in range(1, len(headers) + 1):
        max_len = max(
            len(str(ws.cell(row=r, column=col_idx).value or ""))
            for r in range(1, len(rows) + 2)
        )
        ws.column_dimensions[get_column_letter(col_idx)].width = max_len + 4

    # ── Embedded bar chart ──────────────────────────────────────────
    if len(rows) > 0:
        chart = BarChart()
        chart.type = "col"
        chart.title = "Average CV Score by Model"
        chart.y_axis.title = "Avg CV Score"
        chart.x_axis.title = "Model"
        chart.style = 10
        chart.width = 20
        chart.height = 12

        data_ref = Reference(ws, min_col=3, min_row=1, max_row=len(rows) + 1)
        cats_ref = Reference(ws, min_col=1, min_row=2, max_row=len(rows) + 1)
        chart.add_data(data_ref, titles_from_data=True)
        chart.set_categories(cats_ref)
        chart.shape = 4
        ws.add_chart(chart, "H2")

    # ── Save to bytes and stream ────────────────────────────────────
    buf = BytesIO()
    wb.save(buf)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=automl_leaderboard.xlsx"}
    )
