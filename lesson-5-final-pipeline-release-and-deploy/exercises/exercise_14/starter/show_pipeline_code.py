import os

# Auto-detect base directory (run this from anywhere inside 'starter')
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__))
)

# Files for full pipeline release audit
FILES_TO_INSPECT = {
    'main_pipeline': os.path.join(BASE_DIR, 'main.py'),
    'train_val_test_split_run': os.path.join(BASE_DIR, 'segregate', 'run.py'),
    'train_val_test_split_MLproject': os.path.join(BASE_DIR, 'segregate', 'MLproject'),
    'train_random_forest_run': os.path.join(BASE_DIR, 'random_forest', 'run.py'),
    'train_random_forest_MLproject': os.path.join(BASE_DIR, 'random_forest', 'MLproject'),
    'test_regression_model_run': os.path.join(BASE_DIR, 'test_regression_model', 'run.py'),
    'test_regression_model_MLproject': os.path.join(BASE_DIR, 'test_regression_model', 'MLproject'),
    'check_data_run': os.path.join(BASE_DIR, 'check_data', 'run.py'),
    'check_data_tests': os.path.join(BASE_DIR, 'check_data', 'test_data.py'),
    'basic_cleaning_run': os.path.join(BASE_DIR, 'basic_cleaning', 'run.py'),
    'preprocess_run': os.path.join(BASE_DIR, 'preprocess', 'run.py'),
    'config_yaml': os.path.join(BASE_DIR, 'config.yaml')
}

# Loop and print file contents
for name, path in FILES_TO_INSPECT.items():
    print(f"\n{'='*80}")
    print(f"{name.upper()} => {path}")
    print(f"{'='*80}")
    
    if os.path.exists(path):
        with open(path, 'r') as file:
            print(file.read())
    else:
        print(f"⚠ File not found: {path}")