import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CropIQ — Yield Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background-color: #f8f9f2; }

    .hero-box {
        background: linear-gradient(135deg, #2d6a4f 0%, #52b788 100%);
        border-radius: 16px;
        padding: 28px 36px;
        color: white;
        margin-bottom: 24px;
    }
    .hero-box h1 { font-size: 2rem; margin: 0 0 6px 0; }
    .hero-box p  { margin: 0; opacity: 0.88; font-size: 1.05rem; }

    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        border-left: 5px solid #52b788;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 12px;
    }
    .metric-card h3 { margin: 0 0 4px 0; font-size: 0.82rem; color: #666; text-transform: uppercase; letter-spacing: 0.06em; }
    .metric-card p  { margin: 0; font-size: 1.8rem; font-weight: 700; color: #2d6a4f; }

    .trust-panel {
        background: white;
        border-radius: 16px;
        padding: 28px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-top: 20px;
    }
    .trust-panel h2 { color: #1b4332; margin-top: 0; }

    .summary-box {
        background: #d8f3dc;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 20px 0;
        border-left: 5px solid #2d6a4f;
        font-size: 1.05rem;
        color: #1b4332;
        line-height: 1.7;
    }

    .pos-tag {
        display: inline-block;
        background: #d8f3dc;
        color: #1b4332;
        border-radius: 20px;
        padding: 4px 14px;
        margin: 4px;
        font-size: 0.88rem;
        font-weight: 600;
    }
    .neg-tag {
        display: inline-block;
        background: #ffe5d9;
        color: #9d0208;
        border-radius: 20px;
        padding: 4px 14px;
        margin: 4px;
        font-size: 0.88rem;
        font-weight: 600;
    }

    .sidebar-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #2d6a4f;
        margin-bottom: 16px;
    }
    .stSlider > label { font-weight: 600; }

    /* hide default streamlit footer */
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────────
@st.cache_data
def generate_dataset(n=2000, seed=42):
    rng = np.random.default_rng(seed)

    nitrogen      = rng.uniform(0, 140, n)          # kg/ha
    phosphorus    = rng.uniform(0, 100, n)           # kg/ha
    potassium     = rng.uniform(0, 120, n)           # kg/ha
    rainfall      = rng.uniform(200, 1500, n)        # mm/year
    temperature   = rng.uniform(10, 40, n)           # °C
    soil_ph       = rng.uniform(4.5, 9.0, n)
    humidity      = rng.uniform(20, 95, n)           # %
    solar_rad     = rng.uniform(10, 30, n)           # MJ/m²/day
    irrigation    = rng.uniform(0, 500, n)           # mm/season
    organic_matter= rng.uniform(0.5, 6.0, n)        # %

    # Yield (tonnes/ha) — domain-driven synthetic formula
    optimal_ph_bonus = np.exp(-0.5 * ((soil_ph - 6.5) / 0.8) ** 2)
    optimal_temp_bonus = np.exp(-0.5 * ((temperature - 25) / 5) ** 2)
    water = rainfall + irrigation

    yield_ = (
        2.0
        + 0.03  * nitrogen
        + 0.025 * phosphorus
        + 0.018 * potassium
        + 0.002 * water
        + 4.5   * optimal_ph_bonus
        + 3.0   * optimal_temp_bonus
        + 0.02  * humidity
        + 0.08  * solar_rad
        + 0.4   * organic_matter
        + rng.normal(0, 0.35, n)
    )
    yield_ = np.clip(yield_, 0.5, 18)

    df = pd.DataFrame({
        "Nitrogen (kg/ha)":       nitrogen,
        "Phosphorus (kg/ha)":     phosphorus,
        "Potassium (kg/ha)":      potassium,
        "Rainfall (mm)":          rainfall,
        "Temperature (°C)":       temperature,
        "Soil pH":                soil_ph,
        "Humidity (%)":           humidity,
        "Solar Radiation":        solar_rad,
        "Irrigation (mm)":        irrigation,
        "Organic Matter (%)":     organic_matter,
        "Yield (t/ha)":           yield_,
    })
    return df


# ─────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────
@st.cache_resource
def train_model(df):
    X = df.drop(columns=["Yield (t/ha)"])
    y = df["Yield (t/ha)"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "R²": round(r2_score(y_test, y_pred), 3),
        "MAE": round(mean_absolute_error(y_test, y_pred), 3),
        "Test samples": len(X_test),
    }
    # Build SHAP explainer once
    explainer = shap.TreeExplainer(model)
    return model, explainer, X_train, metrics


# ─────────────────────────────────────────────
# HUMAN SUMMARY TRANSLATION LAYER
# ─────────────────────────────────────────────
FEATURE_CONTEXT = {
    "Nitrogen (kg/ha)": {
        "nutrient": True,
        "high_pos": "excellent nitrogen levels powering leaf growth",
        "low_neg":  "nitrogen deficiency limiting photosynthesis",
        "high_neg": "nitrogen excess causing soil acidification",
    },
    "Phosphorus (kg/ha)": {
        "nutrient": True,
        "high_pos": "optimal phosphorus supporting root development",
        "low_neg":  "low phosphorus stunting root systems",
        "high_neg": "phosphorus surplus reducing micronutrient uptake",
    },
    "Potassium (kg/ha)": {
        "nutrient": True,
        "high_pos": "strong potassium levels boosting disease resistance",
        "low_neg":  "potassium shortage weakening crop stems",
        "high_neg": "excess potassium disrupting calcium balance",
    },
    "Rainfall (mm)": {
        "high_pos": "generous rainfall keeping soil moisture ideal",
        "low_neg":  "insufficient rainfall causing drought stress",
        "high_neg": "excessive rainfall risking waterlogging",
    },
    "Temperature (°C)": {
        "high_pos": "temperatures in the sweet spot for growth",
        "low_neg":  "cold temperatures slowing metabolic activity",
        "high_neg": "heat stress reducing pollination success",
    },
    "Soil pH": {
        "high_pos": "near-neutral pH unlocking optimal nutrient availability",
        "low_neg":  "acidic soil locking away key nutrients",
        "high_neg": "alkaline soil blocking iron and manganese uptake",
    },
    "Humidity (%)": {
        "high_pos": "balanced humidity reducing plant water stress",
        "low_neg":  "dry air increasing evapotranspiration losses",
        "high_neg": "high humidity increasing fungal disease risk",
    },
    "Solar Radiation": {
        "high_pos": "abundant sunlight driving strong photosynthesis",
        "low_neg":  "poor light conditions reducing energy production",
        "high_neg": "extreme radiation causing leaf scorch",
    },
    "Irrigation (mm)": {
        "high_pos": "well-managed irrigation supplementing water needs",
        "low_neg":  "under-irrigation leaving crops water-stressed",
        "high_neg": "over-irrigation waterlogging root zones",
    },
    "Organic Matter (%)": {
        "high_pos": "rich organic matter improving soil structure and fertility",
        "low_neg":  "low organic matter limiting soil water retention",
        "high_neg": "unusually high organic matter causing nitrogen immobilisation",
    },
}

def generate_human_summary(shap_values, feature_names, feature_values, predicted_yield):
    """
    Translation Layer: converts raw SHAP values into a stakeholder-friendly narrative.
    """
    shap_df = pd.DataFrame({
        "feature": feature_names,
        "shap":    shap_values,
        "value":   feature_values,
    }).sort_values("shap", ascending=False)

    positives = shap_df[shap_df["shap"] > 0.05]
    negatives = shap_df[shap_df["shap"] < -0.05]

    rating = "thriving 🌱" if predicted_yield > 8 else ("growing well 🌾" if predicted_yield > 5 else "under stress ⚠️")

    lines = [f"**Your field is {rating}**, with a predicted yield of **{predicted_yield:.2f} t/ha**.\n"]

    if not positives.empty:
        top_pos = positives.iloc[0]
        ctx = FEATURE_CONTEXT.get(top_pos["feature"], {})
        desc = ctx.get("high_pos", f"favourable {top_pos['feature']}")
        lines.append(f"🟢 **Key strength:** This field is benefiting most from {desc}.")

    if not negatives.empty:
        top_neg = negatives.iloc[0]
        ctx = FEATURE_CONTEXT.get(top_neg["feature"], {})
        # Determine high vs low context
        median_val = 50  # rough mid-range fallback
        desc = ctx.get("low_neg", f"suboptimal {top_neg['feature']}")
        lines.append(f"🔴 **Main concern:** Yield is being held back by {desc}.")

    if len(positives) > 1:
        other_pos = ", ".join(positives["feature"].iloc[1:4].tolist())
        lines.append(f"✅ **Also helping:** {other_pos}.")

    lines.append(
        f"\n💡 **Recommendation:** Focus on addressing the limiting factor above "
        f"to unlock the most yield gain per unit of input."
    )
    return "\n\n".join(lines)


# ─────────────────────────────────────────────
# TRUST PANEL CHART
# ─────────────────────────────────────────────
def plot_trust_panel(shap_values, feature_names):
    shap_df = pd.DataFrame({"feature": feature_names, "shap": shap_values})
    shap_df = shap_df.sort_values("shap")

    colors = ["#e63946" if v < 0 else "#2d6a4f" for v in shap_df["shap"]]
    labels = [f.replace(" (kg/ha)", "").replace(" (mm)", "").replace(" (%)", "").replace(" (°C)", "") for f in shap_df["feature"]]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8f9f2")

    bars = ax.barh(labels, shap_df["shap"], color=colors, height=0.6, edgecolor="white", linewidth=0.8)
    ax.axvline(0, color="#333", linewidth=1.2, linestyle="--", alpha=0.5)

    for bar, val in zip(bars, shap_df["shap"]):
        x = val + (0.02 if val >= 0 else -0.02)
        ha = "left" if val >= 0 else "right"
        ax.text(x, bar.get_y() + bar.get_height()/2, f"{val:+.2f}",
                va="center", ha=ha, fontsize=8.5, color="#333", fontweight="600")

    ax.set_xlabel("Impact on Yield (t/ha)", fontsize=10, color="#555")
    ax.set_title("", fontsize=1)
    ax.tick_params(axis="y", labelsize=9.5, colors="#333")
    ax.tick_params(axis="x", labelsize=8.5, colors="#888")
    ax.spines[["top","right","left"]].set_visible(False)
    ax.spines["bottom"].set_color("#ddd")
    ax.grid(axis="x", linestyle=":", alpha=0.4, color="#aaa")

    pos_patch = mpatches.Patch(color="#2d6a4f", label="Positive influence")
    neg_patch = mpatches.Patch(color="#e63946", label="Negative influence")
    ax.legend(handles=[pos_patch, neg_patch], loc="lower right", fontsize=8.5,
              framealpha=0.9, edgecolor="#ddd")

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# GLOBAL FEATURE IMPORTANCE CHART
# ─────────────────────────────────────────────
def plot_global_importance(model, feature_names):
    imp = model.feature_importances_
    df_imp = pd.DataFrame({"feature": feature_names, "importance": imp}).sort_values("importance")
    labels = [f.replace(" (kg/ha)", "").replace(" (mm)", "").replace(" (%)", "").replace(" (°C)", "") for f in df_imp["feature"]]

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8f9f2")

    cmap = plt.cm.YlGn
    norm_vals = (df_imp["importance"] - df_imp["importance"].min()) / (df_imp["importance"].max() - df_imp["importance"].min())
    bar_colors = [cmap(0.35 + 0.65 * v) for v in norm_vals]

    ax.barh(labels, df_imp["importance"], color=bar_colors, height=0.65, edgecolor="white")
    ax.set_xlabel("Mean Decrease in Impurity", fontsize=9, color="#555")
    ax.tick_params(labelsize=9, colors="#333")
    ax.spines[["top","right","left"]].set_visible(False)
    ax.spines["bottom"].set_color("#ddd")
    ax.grid(axis="x", linestyle=":", alpha=0.4, color="#aaa")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():
    df = generate_dataset()
    model, explainer, X_train, metrics = train_model(df)
    features = [c for c in df.columns if c != "Yield (t/ha)"]

    # ── HERO ──────────────────────────────────
    st.markdown("""
    <div class="hero-box">
        <h1>🌾 CropIQ — Yield Prediction & Explainability Dashboard</h1>
        <p>Enter your field conditions, get an AI-powered yield forecast, and understand <em>exactly</em> why — in plain language.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── SIDEBAR: Field Inputs ──────────────────
    st.sidebar.markdown('<div class="sidebar-header">🌱 Your Field Conditions</div>', unsafe_allow_html=True)

    nitrogen     = st.sidebar.slider("Nitrogen (kg/ha)",      0,   140,  80)
    phosphorus   = st.sidebar.slider("Phosphorus (kg/ha)",    0,   100,  50)
    potassium    = st.sidebar.slider("Potassium (kg/ha)",     0,   120,  60)
    rainfall     = st.sidebar.slider("Rainfall (mm/year)",  200,  1500, 750)
    temperature  = st.sidebar.slider("Temperature (°C)",      10,   40,  25)
    soil_ph      = st.sidebar.slider("Soil pH",             4.5,  9.0,  6.5)
    humidity     = st.sidebar.slider("Humidity (%)",          20,   95,  65)
    solar_rad    = st.sidebar.slider("Solar Radiation (MJ/m²/day)", 10, 30, 20)
    irrigation   = st.sidebar.slider("Irrigation (mm/season)", 0,  500, 150)
    organic_mat  = st.sidebar.slider("Organic Matter (%)",   0.5,  6.0,  3.0)

    input_data = pd.DataFrame([[
        nitrogen, phosphorus, potassium, rainfall, temperature,
        soil_ph, humidity, solar_rad, irrigation, organic_mat
    ]], columns=features)

    # ── PREDICTION ────────────────────────────
    predicted_yield = model.predict(input_data)[0]

    # ── SHAP values for this prediction ───────
    shap_values = explainer.shap_values(input_data)[0]

    # ── TOP ROW METRICS ───────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <h3>Predicted Yield</h3><p>{predicted_yield:.2f} t/ha</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <h3>Model R²</h3><p>{metrics['R²']}</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <h3>Mean Abs Error</h3><p>{metrics['MAE']} t/ha</p></div>""", unsafe_allow_html=True)
    with col4:
        top_driver = features[np.argmax(np.abs(shap_values))]
        short_name = top_driver.split(" (")[0]
        st.markdown(f"""<div class="metric-card">
            <h3>Top Driver</h3><p style="font-size:1.2rem">{short_name}</p></div>""", unsafe_allow_html=True)

    st.divider()

    # ── TRUST PANEL ───────────────────────────
    st.markdown('<div class="trust-panel">', unsafe_allow_html=True)
    st.markdown("## 🔍 Trust Panel — What's Driving This Yield?")

    left, right = st.columns([1.1, 1])

    with left:
        st.markdown("### 📊 Influence Breakdown")
        fig_trust = plot_trust_panel(shap_values, features)
        st.pyplot(fig_trust)

        # Tag clouds
        pos_features = [(features[i], shap_values[i]) for i in range(len(features)) if shap_values[i] > 0.05]
        neg_features = [(features[i], shap_values[i]) for i in range(len(features)) if shap_values[i] < -0.05]
        pos_features.sort(key=lambda x: -x[1])
        neg_features.sort(key=lambda x: x[1])

        if pos_features:
            tags = " ".join([f'<span class="pos-tag">✅ {f.split(" (")[0]}</span>' for f, _ in pos_features[:5]])
            st.markdown(f"**Positive Influences:** {tags}", unsafe_allow_html=True)
        if neg_features:
            tags = " ".join([f'<span class="neg-tag">⚠️ {f.split(" (")[0]}</span>' for f, _ in neg_features[:5]])
            st.markdown(f"**Negative Influences:** {tags}", unsafe_allow_html=True)

    with right:
        st.markdown("### 📝 Human Summary")
        summary = generate_human_summary(shap_values, features, input_data.values[0], predicted_yield)
        st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)

        st.markdown("### 📈 What the AI Says About Your Inputs")
        display_df = pd.DataFrame({
            "Feature": [f.split(" (")[0] for f in features],
            "Your Value": [f"{v:.1f}" for v in input_data.values[0]],
            "SHAP Impact": [f"{'+' if s>0 else ''}{s:.3f} t/ha" for s in shap_values],
        })
        st.dataframe(display_df, hide_index=True, use_container_width=True,
                     column_config={
                         "SHAP Impact": st.column_config.TextColumn("SHAP Impact"),
                     })

    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # ── GLOBAL IMPORTANCE ─────────────────────
    with st.expander("🌐 Global Feature Importance (across all 2,000 simulated fields)", expanded=False):
        col_a, col_b = st.columns([1, 1.5])
        with col_a:
            st.markdown("""
**Why Random Forest?**

Random Forest sits in the ideal "Rashomon zone" between black-box deep learning (high accuracy, zero interpretability) and simple linear models (transparent, lower accuracy).

| Property | Deep Learning | **Random Forest** | Linear Model |
|---|---|---|---|
| Accuracy | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| Interpretable | ❌ | ✅ via SHAP | ✅ native |
| Non-linear | ✅ | ✅ | ❌ |
| Stakeholder trust | ❌ | ✅ | ✅ |

SHAP (SHapley Additive exPlanations) provides theoretically grounded per-prediction attributions with the `TreeExplainer` running in **O(TLD)** — fast enough for real-time inference.
            """)
        with col_b:
            fig_global = plot_global_importance(model, features)
            st.pyplot(fig_global)

    # ── DATA EXPLORER ─────────────────────────
    with st.expander("📂 Explore the Synthetic Dataset (sample)", expanded=False):
        st.dataframe(df.sample(50, random_state=7).round(2), use_container_width=True)

    st.caption("CropIQ · Powered by RandomForest + SHAP · Synthetic data only — for demonstration purposes.")

if __name__ == "__main__":
    main()
