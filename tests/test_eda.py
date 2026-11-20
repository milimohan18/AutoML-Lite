import threading

import matplotlib
import numpy as np
import pandas as pd

from automl_lite.eda import generate_eda


def test_matplotlib_uses_non_gui_backend():
    assert matplotlib.get_backend().lower() == "agg"


def test_generate_eda_in_background_thread(tmp_path):
    # Runs are processed in background threads, where GUI backends like TkAgg crash
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "size": rng.normal(size=50),
        "texture": rng.normal(size=50),
        "target": rng.choice(["benign", "malignant"], size=50),
    })
    df.loc[:4, "texture"] = np.nan  # so the missing-values chart is drawn too

    result, errors = {}, []

    def worker():
        try:
            result["summary"] = generate_eda(df, "target", "classification", tmp_path)
        except Exception as e:
            errors.append(e)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=60)

    assert not errors, f"generate_eda failed in a background thread: {errors[0]!r}"
    assert "summary" in result
    assert (tmp_path / "target_distribution.png").exists()
    assert (tmp_path / "missing_values.png").exists()
