[README.md](https://github.com/user-attachments/files/25714503/README.md)
# 🌾 CropIQ — Crop Yield Prediction & Explainability Dashboard

> An interactive Streamlit dashboard that predicts crop yields using a Random Forest Regressor, with real-time SHAP-powered explainability and a human-readable Trust Panel designed for non-technical rural stakeholders.

---

## 📸 Overview

CropIQ bridges the gap between high-accuracy machine learning and real-world agricultural decision-making. Farmers and agronomists can adjust field conditions via intuitive sliders and instantly see:

- A **predicted yield** in tonnes per hectare
- A **visual breakdown** of which factors are helping or hurting the yield
- A **plain-language Human Summary** — no data science background required

---

## ✨ Features

### 🤖 Predictive Model
- **Random Forest Regressor** (200 trees, max depth 12) trained on a 2,000-sample synthetic agriculture dataset
- Achieves ~0.97 R² while remaining fully interpretable — the ideal balance between deep learning accuracy and linear model transparency
- Trained features: Nitrogen, Phosphorus, Potassium, Rainfall, Temperature, Soil pH, Humidity, Solar Radiation, Irrigation, Organic Matter

### 🔍 SHAP Explainability
- Uses `shap.TreeExplainer` for exact, per-prediction feature attributions (runs in O(TLD) — fast enough for real-time inference)
- Every slider change triggers a fresh SHAP calculation, so explanations are always tied to the current inputs

### 🗣️ Translation Layer
- A dedicated `generate_human_summary()` function intercepts raw SHAP floats and converts them into agronomic prose
- Uses a `FEATURE_CONTEXT` dictionary that maps each feature × impact direction to domain-aware language (e.g. nitrogen excess → *"nitrogen excess causing soil acidification"*, not just *"Nitrogen: +0.42"*)

### 🛡️ Trust Panel
- **Influence Breakdown chart** — horizontal bar chart with green (positive) and red (negative) bars per feature
- **Tag clouds** — at-a-glance positive/negative influence badges
- **Human Summary box** — narrative explanation with key strength, main concern, and a concrete recommendation
- **SHAP impact table** — signed per-feature contribution values for technical users

### 📊 Global Insights
- Collapsible **Global Feature Importance** panel with a colour-graded bar chart
- Model architecture comparison table (Deep Learning vs Random Forest vs Linear Model)
- Sample dataset explorer (50-row random sample)

---

## 🏗️ Architecture

```
crop_yield_dashboard.py
│
├── generate_dataset()        # Synthetic agriculture data (domain-driven formula)
├── train_model()             # RandomForestRegressor + SHAP TreeExplainer
├── generate_human_summary()  # Translation Layer: SHAP → plain English
├── plot_trust_panel()        # Per-prediction SHAP bar chart
├── plot_global_importance()  # Model-wide feature importance chart
└── main()                    # Streamlit UI layout & interactivity
```

### Why Random Forest over Deep Learning?

| Property | Deep Learning | **Random Forest** | Linear Model |
|---|---|---|---|
| Accuracy | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| Interpretable | ❌ | ✅ via SHAP | ✅ native |
| Handles non-linearity | ✅ | ✅ | ❌ |
| Stakeholder trust | ❌ | ✅ | ✅ |
| Training data needed | Large | Moderate | Small |

Random Forest sits in the **Rashomon zone** — matching deep learning accuracy on tabular data while remaining auditable via SHAP.

---

## 📦 Installation

### Prerequisites
- Python 3.9+

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/cropiq.git
cd cropiq

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the dashboard
streamlit run crop_yield_dashboard.py
```

---

## 📋 Requirements

Create a `requirements.txt` with:

```
streamlit>=1.32.0
scikit-learn>=1.4.0
shap>=0.45.0
pandas>=2.0.0
numpy>=1.26.0
matplotlib>=3.8.0
```

---

## 🌱 Input Features

| Feature | Unit | Range | Description |
|---|---|---|---|
| Nitrogen | kg/ha | 0 – 140 | Primary macronutrient for leaf growth |
| Phosphorus | kg/ha | 0 – 100 | Supports root development and energy transfer |
| Potassium | kg/ha | 0 – 120 | Disease resistance and stem strength |
| Rainfall | mm/year | 200 – 1500 | Annual precipitation |
| Temperature | °C | 10 – 40 | Growing season average |
| Soil pH | — | 4.5 – 9.0 | Nutrient availability proxy |
| Humidity | % | 20 – 95 | Ambient relative humidity |
| Solar Radiation | MJ/m²/day | 10 – 30 | Photosynthesis driver |
| Irrigation | mm/season | 0 – 500 | Supplemental water input |
| Organic Matter | % | 0.5 – 6.0 | Soil health and water retention |

---

## 🗂️ Project Structure

```
cropiq/
├── crop_yield_dashboard.py   # Main application
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 🔮 Potential Extensions

- **CSV upload** — let users import their own field data for batch predictions
- **SHAP beeswarm plot** — global SHAP summary across the full dataset
- **Multi-crop mode** — switch between wheat, maize, rice yield models
- **Time-series inputs** — incorporate seasonal weather forecast data
- **Export report** — generate a PDF summary of the Trust Panel per field

---

## ⚠️ Disclaimer

This dashboard uses **synthetic data** generated from a domain-informed mathematical formula. It is intended for demonstration and educational purposes only. Do not use predictions for real agricultural or commercial decisions without validation against actual field data.

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*Built with ❤️ using Streamlit · scikit-learn · SHAP · Matplotlib*
