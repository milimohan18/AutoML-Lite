import pandas as pd
from automl_lite.data_loader import infer_task_type, load_data

def test_load_data_strips_column_whitespace(tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text("mean area ,  perimeter,target\n1,2,a\n3,4,b\n")

    df = load_data(str(csv))
    assert list(df.columns) == ["mean area", "perimeter", "target"]

def test_infer_task_type_classification():
    df = pd.DataFrame({
        'feature': [1, 2, 3, 4],
        'target': ['cat', 'dog', 'cat', 'dog']
    })
    
    task_type = infer_task_type(df, 'target')
    assert task_type == 'classification'

def test_infer_task_type_regression():
    target_vals = list(range(20))
    df = pd.DataFrame({
        'target': target_vals
    })
    
    task_type = infer_task_type(df, 'target')
    assert task_type == 'regression'

if __name__ == "__main__":
    test_infer_task_type_classification()
    test_infer_task_type_regression()
    print("All basic tests passed!")
