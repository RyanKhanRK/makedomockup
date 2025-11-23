import mlflow
from flask import Flask
from flask_cors import CORS
import subprocess
import sys

# Install flask-cors if needed
try:
    from flask_cors import CORS
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "flask-cors"])
    from flask_cors import CORS

# Start MLflow with CORS enabled
if __name__ == "__main__":
    import os
    os.system("mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns")
