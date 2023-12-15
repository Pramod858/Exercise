import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

list_of_files = [
    "config.yaml",
    "requirements.txt",
    "run_deployment.py",
    "run_pipeline.py",
    "streamlit_app.py",
    "__init__.py",
    "materializer/custom_materializer.py",
    "model/data_cleaning.py",
    "model/evaluation.py",
    "model/model_dev.py",
    "pipelines/deployment_pipeline.py",
    "pipelines/training_pipeline.py",
    "pipelines/utils.py",
    "saved_model/model.pkl",
    "steps/clean_data.py",
    "steps/config.py",
    "steps/evaluation.py",
    "steps/ingest_data.py",
    "steps/model_train.py",
    "tests/data_test.py",
    "tests/__init__.py",
]

for filepath in list_of_files:
    filepath = Path(filepath)

    if not os.path.exists(filepath):
        if filepath.is_dir():  # Check if it's a directory
            os.makedirs(filepath)
            logging.info(f"Creating directory: {filepath}")
        else:  # It's a file
            filedir = filepath.parent
            if filedir and not os.path.exists(filedir):
                os.makedirs(filedir)
                logging.info(f"Creating directory: {filedir}")
            with open(filepath, "w") as f:
                pass
            logging.info(f"Creating empty file: {filepath.name}")
    else:
        logging.info(f"{filepath.name} already exists")
