# MLflow Extended Dashboard

**Extended Version of MLflow for Model Management and Experiment Tracking**

Senior Project by **Ryan Khan** (64070503446)  
King Mongkut's University of Technology Thonburi

---

## 📋 Project Overview

This project extends MLflow's capabilities with:
- ✅ **Enhanced Dashboard UI** - Modern, interactive interface
- ✅ **SHAP Integration** - Feature importance & explainability analysis
- ✅ **Fairness Analysis** - Bias detection across subgroups
- ✅ **Run Comparison** - Side-by-side experiment comparison
- ✅ **Model Registry** - Version tracking & deployment management

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip
- Web browser

### Installation & Setup

#### Option 1: Automated Setup (Recommended)

**Linux/Mac:**
```bash
chmod +x SETUP_AND_RUN.sh
./SETUP_AND_RUN.sh
```

**Windows:**
```cmd
SETUP_AND_RUN.bat
```

#### Option 2: Manual Setup

1. **Install dependencies:**
```bash
pip install mlflow scikit-learn shap pandas numpy matplotlib seaborn flask flask-cors
```

2. **Start MLflow server:**
```bash
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

3. **Run experiments:**
```bash
# Titanic experiments (UC1 & UC2)
python titanic_shap_example.py

# Iris experiments (UC3)
python iris_shap_example.py
```

4. **Open dashboard:**
```bash
# Just open the HTML file in your browser
open mlflow-dashboard.html  # Mac
start mlflow-dashboard.html # Windows
xdg-open mlflow-dashboard.html # Linux
```

---

## 📁 Project Structure

```
mlflow-extended-dashboard/
├── mlflow-dashboard.html          # Main dashboard (React app)
├── shap_mlflow_integration.py    # SHAP logging module
├── titanic_shap_example.py        # UC1 & UC2 experiments
├── iris_shap_example.py           # UC3 experiments
├── SETUP_AND_RUN.sh               # Linux/Mac setup script
├── SETUP_AND_RUN.bat              # Windows setup script
├── README.md                      # This file
├── mlruns/                        # MLflow artifacts (auto-generated)
├── mlflow.db                      # SQLite database (auto-generated)
└── data/                          # Dataset directory (optional)
```

---

## 🎯 Use Cases (From Report)

### UC1: Titanic Binary Classification (Baseline Features)
- **Model:** Logistic Regression
- **Features:** Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
- **Goal:** Establish baseline performance

### UC2: Titanic Binary Classification (Engineered Features)
- **Model:** Random Forest
- **Features:** UC1 features + FamilySize, Title, AgeBin, IsAlone
- **Goal:** Demonstrate impact of feature engineering

### UC3: Iris Multiclass Classification
- **Model:** Decision Tree / Random Forest
- **Features:** Sepal length/width, Petal length/width
- **Goal:** Show multiclass classification with SHAP

---

## 🔍 Features

### 1. Enhanced Dashboard UI
- Modern, responsive design
- Real-time sync with MLflow backend
- Experiment filtering and search
- Status indicators (running/finished/failed)

### 2. SHAP Integration
- **Global Importance:** Bar plots showing feature rankings
- **Local Explanations:** Force plots for individual predictions
- **Dependence Plots:** Feature interactions
- **Waterfall Charts:** Decision path visualization

### 3. Fairness Analysis
- Gender-based performance (Titanic)
- Species-level accuracy (Iris)
- Group comparison visualizations
- Bias metrics logging

### 4. Run Comparison
- Side-by-side metric comparison (up to 3 runs)
- Parameter difference highlighting
- Visual performance charts

### 5. Model Registry
- Version tracking
- Deployment stage management
- Model metadata

---

## 📊 Dashboard Views

### Experiments Tab
- Overview metrics (total runs, accuracy, models)
- Filterable run table
- Compare mode toggle
- Search by run ID or model name

### Run Details Tab
- Full metrics display
- Parameters listing
- SHAP plots (if available)
- Fairness analysis (if available)
- Artifact downloads

### Model Registry Tab
- Registered models listing
- Version history
- Deployment stages
- Quick access to MLflow UI

---

## 🔧 API Integration

The dashboard connects to MLflow's REST API:

```javascript
// Fetch experiments
POST http://localhost:5000/api/2.0/mlflow/experiments/search

// Fetch runs
POST http://localhost:5000/api/2.0/mlflow/runs/search

// Get run details
GET http://localhost:5000/api/2.0/mlflow/runs/get?run_id={id}

// List artifacts
GET http://localhost:5000/api/2.0/mlflow/artifacts/list?run_id={id}

// Download artifact
GET http://localhost:5000/get-artifact?path={path}&run_uuid={id}
```

---

## 🐍 Using SHAP Integration Module

### Basic Usage

```python
from shap_mlflow_integration import log_model_with_shap
from sklearn.ensemble import RandomForestClassifier

# Train your model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Log with SHAP analysis
log_model_with_shap(
    model=model,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    feature_names=['feature1', 'feature2', ...],
    model_name='my_experiment',
    model_type='tree'  # or 'kernel', 'linear'
)
```

### Advanced Usage

```python
from shap_mlflow_integration import SHAPMLflowLogger

# Create logger
shap_logger = SHAPMLflowLogger(
    model=model,
    X_train=X_train,
    X_test=X_test,
    feature_names=feature_names
)

# Compute SHAP values
shap_logger.compute_shap_values(model_type='tree')

# Log specific plots
shap_logger.log_summary_plot(plot_type='bar')
shap_logger.log_feature_importance()
shap_logger.log_dependence_plots(top_n=5)
shap_logger.log_force_plots(num_samples=3)

# Or log everything at once
shap_logger.log_all_plots(model_type='tree')
```

---

## 🔒 CORS Issues?

If you encounter CORS errors, create a proxy server:

```python
# proxy_server.py
from flask import Flask, request, Response
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

MLFLOW_URL = "http://localhost:5000"

@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def proxy(path):
    url = f"{MLFLOW_URL}/{path}"
    resp = requests.request(
        method=request.method,
        url=url,
        headers={k: v for k, v in request.headers if k != 'Host'},
        data=request.get_data(),
        params=request.args
    )
    return Response(resp.content, resp.status_code, resp.raw.headers.items())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
```

Run: `python proxy_server.py`

Then update dashboard to use `http://localhost:5001`

---

## 📈 Example Output

### Metrics Logged
- accuracy
- precision
- recall
- f1_score
- per-class metrics (multiclass)
- SHAP feature importance scores
- fairness metrics (group-wise accuracy)

### Artifacts Logged
- `confusion_matrix.png`
- `shap_summary_bar.png`
- `shap_summary_dot.png`
- `shap_feature_importance.csv`
- `shap_dependence_{feature}.png`
- `shap_force_plot_sample_{i}.png`
- `shap_values.npy` (raw SHAP arrays)
- `per_class_metrics.csv`
- `fairness_gender_analysis.png`

---

## 🧪 Testing

### Load Testing (from Report)
Performance tested with Locust:
- Simulated 20 concurrent users
- Average response time: ~350-400ms
- No failed requests
- Artifact downloads: <800ms

---

## 📝 System Architecture

```
┌─────────────┐
│  Data Input │ (Titanic, Iris)
└──────┬──────┘
       │
       ▼
┌──────────────┐
│ Model Training│ (Zeppelin/Python)
└──────┬───────┘
       │
       ▼
┌───────────────┐
│ MLflow Logging│ (Params, Metrics, Artifacts)
└──────┬────────┘
       │
       ├──────────────┐
       ▼              ▼
┌────────────┐  ┌──────────────┐
│ SHAP       │  │ Fairness     │
│ Analysis   │  │ Evaluation   │
└─────┬──────┘  └──────┬───────┘
      │                │
      └────────┬───────┘
               ▼
     ┌──────────────────┐
     │ MLflow Database  │
     │ + Artifacts      │
     └────────┬─────────┘
              │
              ▼
     ┌──────────────────┐
     │   REST API       │
     └────────┬─────────┘
              │
              ▼
     ┌──────────────────┐
     │ Dashboard UI     │
     │ (React + Charts) │
     └──────────────────┘
```

---

## 🛠️ Troubleshooting

### Problem: "Failed to fetch experiments"
**Solution:** Make sure MLflow server is running on port 5000
```bash
mlflow server --host 0.0.0.0 --port 5000
```

### Problem: "No runs found"
**Solution:** Run the example scripts first
```bash
python titanic_shap_example.py
python iris_shap_example.py
```

### Problem: CORS errors in browser console
**Solution:** Use the proxy server (see CORS section above)

### Problem: SHAP plots not showing
**Solution:** Check that artifacts were logged properly in MLflow UI

### Problem: Import errors
**Solution:** Install all dependencies
```bash
pip install mlflow scikit-learn shap pandas numpy matplotlib seaborn
```

---

## 🚀 Deployment

### Local Development
```bash
mlflow server --host 0.0.0.0 --port 5000
```

### Remote Server
```bash
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri postgresql://user:pass@host/db \
  --default-artifact-root s3://my-bucket/mlruns
```

### Docker (Future Work)
```dockerfile
FROM python:3.10
RUN pip install mlflow scikit-learn shap
EXPOSE 5000
CMD ["mlflow", "server", "--host", "0.0.0.0"]
```

---

## 📚 References

1. MLflow Documentation: https://mlflow.org/docs/latest/
2. SHAP Documentation: https://shap.readthedocs.io/
3. Scikit-learn: https://scikit-learn.org/
4. React + Recharts: https://recharts.org/

---

## 👨‍💻 Author

**Ryan Khan**  
Student ID: 64070503446  
King Mongkut's University of Technology Thonburi  
Bachelor of Engineering (Computer Engineering)

**Advisor:** Dr. Aye Hninn Khine

---

## 📄 License

This project is for academic purposes as part of a senior project requirement.

---

## 🎓 Academic Context

This project fulfills the requirements for:
- **Course:** Senior Project (Term 1 & 2)
- **Program:** Bachelor of Engineering (Computer Engineering)
- **Faculty:** Engineering
- **University:** King Mongkut's University of Technology Thonburi
- **Academic Year:** 2024

---

## 🔮 Future Work (Term 2)

- [ ] Full frontend React implementation
- [ ] Cloud deployment (Docker + AWS/GCP)
- [ ] Extended fairness metrics (AIF360/Fairlearn)
- [ ] Real-time model monitoring
- [ ] Multi-user authentication
- [ ] Advanced visualization (Plotly.js)
- [ ] Model deployment integration
- [ ] Automated reporting

---

## ⭐ Acknowledgments

Special thanks to:
- Dr. Aye Hninn Khine (Project Advisor)
- Dr. Priyakorn Pusawiro (Resource Support)
- Project Committee Members
- KMUTT Faculty of Engineering

---

**Last Updated:** November 2024  
**Version:** 1.0 (Term 1 Complete)
