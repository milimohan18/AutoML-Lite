import pandas as pd
from automl_lite.data_loader import infer_task_type

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
