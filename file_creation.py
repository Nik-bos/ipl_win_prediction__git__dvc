import os

def create_files():
    original_data = 'original_data'
    raw_data = os.path.join('data', 'raw')
    processed_data = os.path.join('data', 'processed')
    notebooks = 'notebooks'
    models = 'models'
    src = 'src'
    scores = 'reports'

    directories = [original_data, raw_data, processed_data, notebooks, models, src, scores]

    # Creating folders.
    # Git doesn't track empty folders, so inside each folder, we are creating .gitkeep file.
    for dir in directories:
        os.makedirs(dir, exist_ok = True)
        print(f"Directory created successfully: {dir}")
        
        # Creating .gitkeep file in each folder bcz git does not track empty folders
        gitkeep_path = os.path.join(dir, '.gitkeep')
        with open(gitkeep_path, 'w') as f:
            pass

        print(f'File .gitkeep created successfully: {gitkeep_path}')
        print("="*30)

    # =============== Creating Files ===============
    files = [
        'dvc.yaml',
        "params.yaml",
        ".gitignore",
        "README.md",
        os.path.join('src', '__init__.py'),
        os.path.join('reports', 'model_parameters.json'),
        os.path.join('reports', 'model_scores.json')
        ]

    for file in files:
        with open(file, 'w') as f:
            pass

        print(f"File created successfully: {file}")


create_files()