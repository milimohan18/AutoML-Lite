import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-GUI backend: charts are drawn in background threads
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path
from typing import Dict, Any, List

def generate_eda(df: pd.DataFrame, target_column: str, task_type: str, run_dir: Path) -> Dict[str, Any]:
    """Generates EDA artifacts and returns a JSON-serializable summary."""
    sns.set_theme(style="whitegrid")
    eda_summary = {}

    # 1. Dataset Overview
    eda_summary["n_rows"] = int(len(df))
    eda_summary["n_columns"] = int(len(df.columns))
    
    dtypes_counts = df.dtypes.value_counts().to_dict()
    eda_summary["column_types"] = {str(k): int(v) for k, v in dtypes_counts.items()}
    
    # 2. Missing Values
    missing_series = df.isnull().sum()
    missing_series = missing_series[missing_series > 0].sort_values(ascending=False)
    
    if not missing_series.empty:
        eda_summary["missing_values"] = missing_series.to_dict()
        plt.figure(figsize=(10, 6))
        sns.barplot(x=missing_series.values, y=missing_series.index)
        plt.title('Missing Values per Column')
        plt.xlabel('Count')
        plt.tight_layout()
        plt.savefig(run_dir / 'missing_values.png')
        plt.close()
    else:
        eda_summary["missing_values"] = {}
        
    # 3. Target Distribution
    plt.figure(figsize=(8, 5))
    if task_type == 'classification':
        target_counts = df[target_column].value_counts()
        eda_summary['target_distribution'] = target_counts.to_dict()
        sns.barplot(x=target_counts.index, y=target_counts.values)
        plt.title(f'Target Distribution: {target_column}')
        plt.xlabel('Class')
        plt.ylabel('Count')
    else:
        target_desc = df[target_column].describe().to_dict()
        eda_summary['target_distribution'] = target_desc
        sns.histplot(df[target_column], kde=True)
        plt.title(f'Target Distribution: {target_column}')
        plt.xlabel(target_column)
    
    plt.tight_layout()
    plt.savefig(run_dir / 'target_distribution.png')
    plt.close()
    
    # 4. Feature Statistics Pipeline
    num_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if target_column in num_features: num_features.remove(target_column)
    if target_column in cat_features: cat_features.remove(target_column)
    
    eda_summary["num_features_stat"] = df[num_features].describe().to_dict() if num_features else {}
    eda_summary["cat_features_stat"] = {
        col: {"n_unique": int(df[col].nunique()), "top": str(df[col].mode()[0]) if not df[col].empty else None}
        for col in cat_features
    }
    
    # 5. Correlation Heatmap (Pearson)
    corr_cols = num_features.copy()
    if target_column in df.select_dtypes(include=['int64', 'float64']).columns.tolist():
        corr_cols.append(target_column)
    
    if len(corr_cols) >= 2:
        corr_matrix = df[corr_cols].corr()
        eda_summary["correlation_matrix"] = {
            str(k): {str(k2): round(v2, 3) for k2, v2 in v.items()}
            for k, v in corr_matrix.to_dict().items()
        }

        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
                    cmap="RdBu_r", center=0, square=True,
                    linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title("Feature Correlation Matrix (Pearson)")
        plt.tight_layout()
        plt.savefig(run_dir / 'correlation_heatmap.png', dpi=150)
        plt.close()
    else:
        eda_summary["correlation_matrix"] = {}

    return eda_summary
