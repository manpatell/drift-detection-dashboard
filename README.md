# 🛡️ CreditGuard — ML Monitoring Platform

A **production-grade MLOps dashboard** for real-time model drift detection, data quality monitoring, and model performance tracking — built for a credit-risk loan default scoring system.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.20%2B-3F4F75?logo=plotly&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-1.11%2B-8CAAE6?logo=scipy&logoColor=white)

---

## Business Context

CreditGuard is a FinTech ML platform that scores loan applications in real time. As market conditions shift (economic downturns, changing demographics, policy changes), the input feature distributions drift away from the training data — degrading model performance silently without a monitoring layer.

This dashboard provides that monitoring layer.

---

## Features

### 🏠 Overview Dashboard
- System health KPIs — accuracy, F1, AUC-ROC, precision, recall vs. baseline
- Gauge charts with SLA thresholds
- Feature drift status heatmap
- PSI bar chart across all monitored features
- 60-day performance & drift trend lines
- Active alerts feed with CSV export
- Data snapshot tabs (reference vs. production)

### 📊 Data Drift Analysis
- Statistical comparison of every feature: KS Test, PSI, Jensen-Shannon Divergence, Wasserstein Distance, Chi-Square
- Colour-coded drift summary table (Critical / Warning / Stable)
- Per-feature deep dive: overlapping histograms, box plots, descriptive stats
- All-features distribution grid (numeric + categorical)

### 🎯 Model Performance
- Baseline vs. production metrics table with delta colouring
- 5 performance gauges (Accuracy, AUC-ROC, F1, Precision, Recall)
- Confusion matrix comparison (reference vs. production)
- ROC curve overlay with AUC comparison
- Prediction score distribution shift
- Feature importance chart (mean impurity decrease)
- 60-day performance trend lines

### 🔍 Feature Analysis
- Descriptive statistics comparison
- Correlation matrix heatmaps + delta correlation drift
- Interactive scatter plots with marginal histograms
- Outlier analysis (Z-score method)
- Missing value analysis
- Value distributions grouped by categorical features (box plots)

### 🚨 Alerts & Configuration
- Active alert cards with severity, PSI, test statistics
- Configurable PSI / KS thresholds via sidebar sliders
- 60-day alert timeline with 7-day moving average
- Notification channel status (Email, Slack, PagerDuty, Webhook)
- Alert policy table (condition → severity → SLA response time)

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI Framework | Streamlit 1.32+ |
| Visualisation | Plotly 5.20+ |
| ML Model | scikit-learn GradientBoostingClassifier |
| Statistical Tests | SciPy (KS, Chi-Square), custom PSI / JS / Wasserstein |
| Data Layer | pandas 2.0+, NumPy 1.26+ |
| Language | Python 3.10+ |

---

## Project Structure

```
drift-detection-dashboard/
├── app.py                     # Home / Overview dashboard
├── pages/
│   ├── 1_Data_Drift.py        # Statistical drift analysis
│   ├── 2_Model_Performance.py # Model monitoring & ROC curves
│   ├── 3_Feature_Analysis.py  # Correlations, outliers, interactions
│   └── 4_Alerts.py            # Alert management & configuration
├── utils/
│   ├── data_generator.py      # Synthetic credit-risk dataset
│   ├── drift_metrics.py       # KS, PSI, JS, Wasserstein, Chi-Square
│   ├── model_utils.py         # GBM training & production evaluation
│   └── visualization.py      # Plotly chart factory
├── config.py                  # Thresholds, feature lists, palette
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/manpatell/drift-detection-dashboard.git
cd drift-detection-dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Drift Detection Methods

| Method | Feature Type | Interpretation |
|---|---|---|
| **KS Test** | Numeric | p < 0.05 → distributions differ significantly |
| **PSI** | Both | < 0.10 stable · 0.10–0.20 moderate · > 0.20 high drift |
| **Jensen-Shannon Divergence** | Both | 0–1 bounded; > 0.10 indicates meaningful shift |
| **Wasserstein Distance** | Numeric | Earth mover's distance, normalised by reference σ |
| **Chi-Square Test** | Categorical | p < 0.05 → category proportions have changed |

---

## MLOps Concepts Demonstrated

- **Data drift** — distributional shift in input features
- **Concept drift** — change in the relationship between inputs and outputs
- **Population Stability Index (PSI)** — industry-standard metric from credit risk
- **Statistical hypothesis testing** — KS, Chi-Square applied to production monitoring
- **Model degradation tracking** — continuous evaluation of deployed model performance
- **Alerting and SLA management** — threshold-based notifications with severity levels
- **Reference/production split** — training distribution as the monitoring baseline
