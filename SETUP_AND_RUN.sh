#!/bin/bash

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  MLflow Extended Dashboard - Complete Setup                  ║"
echo "║  Senior Project: Ryan Khan (64070503446)                      ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Install dependencies
echo -e "${YELLOW}Step 1: Installing Python dependencies...${NC}"
pip install mlflow scikit-learn shap pandas numpy matplotlib seaborn flask flask-cors

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Dependencies installed${NC}\n"
else
    echo -e "${RED}✗ Failed to install dependencies${NC}"
    exit 1
fi

# Step 2: Create directory structure
echo -e "${YELLOW}Step 2: Creating project structure...${NC}"
mkdir -p mlruns
mkdir -p artifacts
mkdir -p data

echo -e "${GREEN}✓ Directories created${NC}\n"

# Step 3: Download Titanic dataset (optional)
echo -e "${YELLOW}Step 3: Checking for datasets...${NC}"
if [ ! -f "titanic.csv" ]; then
    echo "  ℹ Titanic dataset not found. Scripts will use seaborn's built-in dataset."
else
    echo -e "${GREEN}✓ Titanic dataset found${NC}"
fi
echo ""

# Step 4: Start MLflow server
echo -e "${YELLOW}Step 4: Starting MLflow tracking server...${NC}"
echo "  Starting on http://localhost:5000"
echo "  Press Ctrl+C to stop"
echo ""

# Kill any existing MLflow servers
pkill -f "mlflow server" 2>/dev/null

# Start MLflow in background
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns &
MLFLOW_PID=$!

# Wait for MLflow to start
echo "  Waiting for MLflow to start..."
sleep 5

# Check if MLflow is running
if ps -p $MLFLOW_PID > /dev/null; then
    echo -e "${GREEN}✓ MLflow server started (PID: $MLFLOW_PID)${NC}\n"
else
    echo -e "${RED}✗ Failed to start MLflow server${NC}"
    exit 1
fi

# Step 5: Run experiments
echo -e "${YELLOW}Step 5: Running ML experiments...${NC}"
echo ""

read -p "Run Titanic experiments (UC1 & UC2)? [Y/n] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
    echo "  Running Titanic experiments..."
    python titanic_shap_example.py
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Titanic experiments complete${NC}\n"
    else
        echo -e "${RED}✗ Titanic experiments failed${NC}\n"
    fi
fi

read -p "Run Iris experiments (UC3)? [Y/n] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
    echo "  Running Iris experiments..."
    python iris_shap_example.py
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Iris experiments complete${NC}\n"
    else
        echo -e "${RED}✗ Iris experiments failed${NC}\n"
    fi
fi

# Step 6: Open dashboard
echo -e "${YELLOW}Step 6: Opening dashboard...${NC}"
echo ""
echo "  Dashboard: mlflow-dashboard.html"
echo "  MLflow UI: http://localhost:5000"
echo ""

# Detect OS and open dashboard
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    open mlflow-dashboard.html
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    xdg-open mlflow-dashboard.html 2>/dev/null || echo "Please open mlflow-dashboard.html in your browser"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    start mlflow-dashboard.html
else
    echo "Please open mlflow-dashboard.html in your browser"
fi

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                     SETUP COMPLETE! ✅                        ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "Access your dashboard:"
echo "  • Extended Dashboard: mlflow-dashboard.html"
echo "  • MLflow UI: http://localhost:5000"
echo ""
echo "MLflow server is running in background (PID: $MLFLOW_PID)"
echo "To stop: kill $MLFLOW_PID"
echo ""
echo "Next steps:"
echo "  1. Explore experiments in the dashboard"
echo "  2. Check SHAP plots in Run Details"
echo "  3. Compare different runs"
echo "  4. View fairness analysis"
echo ""
