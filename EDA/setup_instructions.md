# Demand Forecasting EDA Setup Instructions

## Prerequisites
- Python 3.8+ installed
- pip package manager working
- Dataset files already placed in `data/` folder:
  - `data/train.csv`
  - `data/test.csv`
  - `data/sample_submission.csv`

## Quick Setup

### 1. Install Dependencies
```bash
# Install all required packages from requirements.txt
pip install -r requirements.txt
```

**Alternative (if requirements.txt doesn't work):**
```bash
# Install packages individually
pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn plotly jupyter ipykernel
```

### 2. Setup Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install requirements in virtual environment
pip install -r requirements.txt
```

### 3. Setup Jupyter Kernel (For Notebook Usage)
```bash
# Install the current environment as a Jupyter kernel
python -m ipykernel install --user --name=demand-forecasting --display-name "Demand Forecasting"

# Start Jupyter Notebook
jupyter notebook
```

### 4. Run the Analysis

Choose one of the following options:

#### Option A: Run the Python Script
```bash
# Run the EDA script directly
python demand_forecasting_eda.py
```

This will:
- Generate detailed analysis output to console
- Save 8 high-quality graphs to `output_graphs/` folder
- Create comprehensive statistical reports

#### Option B: Use Jupyter Notebook
```bash
# Start Jupyter Notebook
jupyter notebook

# Then:
# 1. Open demand_forecasting_eda.ipynb in your browser
# 2. Select kernel: "Demand Forecasting" (if you set it up)
# 3. Run all cells (Cell → Run All)
```

**Note:** If you didn't set up the custom kernel, just use the default Python 3 kernel.

## Project Files

### Core Analysis Files
- **`demand_forecasting_eda.py`** - Enhanced EDA script with advanced statistical analysis
- **`demand_forecasting_eda.ipynb`** - Jupyter notebook version with interactive cells
- **`requirements.txt`** - All required Python packages
- **`setup_instructions.md`** - This setup guide

### Output Files (Generated)
- **`output_graphs/`** - Folder containing 8 high-quality visualization files:
  - `01_sales_distribution_analysis.png`
  - `02_temporal_patterns_analysis.png`
  - `03_store_item_patterns_analysis.png`
  - `04_store_item_heatmap.png`
  - `05_correlation_matrix.png`
  - `06_seasonal_decomposition.png`
  - `07_advanced_statistical_analysis.png`
  - `08_outlier_detection_analysis.png`

## Enhanced EDA Features

- Statistical Analysis
- Comprehensive Outlier Detection



## Dataset Structure

Expected files in `data/` folder:
- **`data/train.csv`** - Training data with sales history
- **`data/test.csv`** - Test data for predictions  