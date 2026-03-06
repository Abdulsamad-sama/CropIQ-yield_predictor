import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from io import BytesIO
import base64
import warnings
warnings.filterwarnings("ignore")


# ------------------------------------
# PAGE CONFIG
# ------------------------------------
st.set_page_config(
    page_title="CropIQ v2",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded")

# ------------------------------------
# CUSTOM CSS
# ------------------------------------
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
    .metric-card h3 { margin: 0 0 4px 0; 
        font-size: 0.82rem; 
        color: #666; 
        text-transform: uppercase; 
        letter-spacing: 0.06em;
    }
    .metric-card p  { 
        margin: 0; 
        font-size: 1.8rem; 
        font-weight: 700; 
        color: #2d6a4f; 
    }
    .trust-panel { 
        background: white; 
        border-radius: 16px; 
        padding: 28px; 
        box-shadow: 0 4px 20px rgba(0,0,0,0.08); 
        margin-top: 20px; 
    }
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
    .upload-zone { 
        background: #f0fdf4; 
        border: 2px dashed #52b788; 
        border-radius: 12px; 
        padding: 30px; 
        text-align: center; 
        color: #666; 
        margin: 12px 0; 
    }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

CROP_PROFILES = {
    "🌾 Wheat":   {"color":"#e9c46a","optimal_temp":18,"temp_spread":5,"optimal_ph":6.5,"ph_spread":0.7,"base_yield":2.5,"n_coeff":0.032,"p_coeff":0.022,"k_coeff":0.015,"water_coeff":0.0018,"hum_coeff":0.018,"sol_coeff":0.07,"om_coeff":0.35,"ph_bonus":3.5,"temp_bonus":2.5,"yield_range":(1,12),"noise":0.3,"description":"Cool-season cereal. Thrives in 15-24°C with moderate rainfall."},
    "🌽 Maize":   {"color":"#f4a261","optimal_temp":28,"temp_spread":5,"optimal_ph":6.2,"ph_spread":0.8,"base_yield":3.0,"n_coeff":0.042,"p_coeff":0.028,"k_coeff":0.020,"water_coeff":0.0022,"hum_coeff":0.022,"sol_coeff":0.09,"om_coeff":0.45,"ph_bonus":4.0,"temp_bonus":3.5,"yield_range":(1,18),"noise":0.4,"description":"Warm-season crop. High nitrogen demand. Sensitive to drought."},
    "🍚 Rice":    {"color":"#90e0ef","optimal_temp":30,"temp_spread":4,"optimal_ph":6.0,"ph_spread":0.9,"base_yield":2.0,"n_coeff":0.028,"p_coeff":0.020,"k_coeff":0.016,"water_coeff":0.0028,"hum_coeff":0.030,"sol_coeff":0.06,"om_coeff":0.38,"ph_bonus":3.8,"temp_bonus":3.2,"yield_range":(1,14),"noise":0.35,"description":"Flooded paddy crop. High humidity and water requirements."},
    "🫘 Soybean": {"color":"#95d5b2","optimal_temp":25,"temp_spread":5,"optimal_ph":6.8,"ph_spread":0.6,"base_yield":1.5,"n_coeff":0.015,"p_coeff":0.030,"k_coeff":0.022,"water_coeff":0.0020,"hum_coeff":0.020,"sol_coeff":0.08,"om_coeff":0.50,"ph_bonus":4.2,"temp_bonus":2.8,"yield_range":(0.5,8),"noise":0.28,"description":"Legume. Nitrogen-fixing. Sensitive to soil pH extremes."},
}

FEATURES = [
    "Nitrogen (kg/ha)","Phosphorus (kg/ha)","Potassium (kg/ha)",
    "Rainfall (mm)","Temperature (°C)","Soil pH",
    "Humidity (%)","Solar Radiation","Irrigation (mm)","Organic Matter (%)",
]

FEATURE_RANGES = {
    "Nitrogen (kg/ha)":(0,140,80),"Phosphorus (kg/ha)":(0,100,50),"Potassium (kg/ha)":(0,120,60),
    "Rainfall (mm)":(200,1500,750),"Temperature (°C)":(10,40,25),"Soil pH":(4.5,9.0,6.5),
    "Humidity (%)":(20,95,65),"Solar Radiation":(10,30,20),"Irrigation (mm)":(0,500,150),"Organic Matter (%)":(0.5,6.0,3.0),
}

FEATURE_CONTEXT = {
    "Nitrogen (kg/ha)":   {"high_pos":"excellent nitrogen levels powering leaf growth","low_neg":"nitrogen deficiency limiting photosynthesis","high_neg":"nitrogen excess causing soil acidification"},
    "Phosphorus (kg/ha)": {"high_pos":"optimal phosphorus supporting root development","low_neg":"low phosphorus stunting root systems","high_neg":"phosphorus surplus reducing micronutrient uptake"},
    "Potassium (kg/ha)":  {"high_pos":"strong potassium levels boosting disease resistance","low_neg":"potassium shortage weakening crop stems","high_neg":"excess potassium disrupting calcium balance"},
    "Rainfall (mm)":      {"high_pos":"generous rainfall keeping soil moisture ideal","low_neg":"insufficient rainfall causing drought stress","high_neg":"excessive rainfall risking waterlogging"},
    "Temperature (°C)":   {"high_pos":"temperatures in the sweet spot for growth","low_neg":"cold temperatures slowing metabolic activity","high_neg":"heat stress reducing pollination success"},
    "Soil pH":            {"high_pos":"near-neutral pH unlocking optimal nutrient availability","low_neg":"acidic soil locking away key nutrients","high_neg":"alkaline soil blocking iron and manganese uptake"},
    "Humidity (%)":       {"high_pos":"balanced humidity reducing plant water stress","low_neg":"dry air increasing evapotranspiration losses","high_neg":"high humidity increasing fungal disease risk"},
    "Solar Radiation":    {"high_pos":"abundant sunlight driving strong photosynthesis","low_neg":"poor light conditions reducing energy production","high_neg":"extreme radiation causing leaf scorch"},
    "Irrigation (mm)":    {"high_pos":"well-managed irrigation supplementing water needs","low_neg":"under-irrigation leaving crops water-stressed","high_neg":"over-irrigation waterlogging root zones"},
    "Organic Matter (%)": {"high_pos":"rich organic matter improving soil structure","low_neg":"low organic matter limiting water retention","high_neg":"excess organic matter causing nitrogen immobilisation"},
}

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


# ── DATA (FIXED: explicit column mapping, no locals() dict comprehension) ──
@st.cache_data
def generate_dataset(crop_name: str, n: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    p   = CROP_PROFILES[crop_name]

    nitrogen       = rng.uniform(0,   140, n)
    phosphorus     = rng.uniform(0,   100, n)
    potassium      = rng.uniform(0,   120, n)
    rainfall       = rng.uniform(200, 1500, n)
    temperature    = rng.uniform(10,  40,  n)
    soil_ph        = rng.uniform(4.5, 9.0, n)
    humidity       = rng.uniform(20,  95,  n)
    solar_rad      = rng.uniform(10,  30,  n)
    irrigation     = rng.uniform(0,   500, n)
    organic_matter = rng.uniform(0.5, 6.0, n)

    optimal_ph_bonus   = np.exp(-0.5 * ((soil_ph    - p["optimal_ph"])   / p["ph_spread"])   ** 2)
    optimal_temp_bonus = np.exp(-0.5 * ((temperature - p["optimal_temp"]) / p["temp_spread"]) ** 2)

    yield_ = (
        p["base_yield"]
        + p["n_coeff"]     * nitrogen
        + p["p_coeff"]     * phosphorus
        + p["k_coeff"]     * potassium
        + p["water_coeff"] * (rainfall + irrigation)
        + p["ph_bonus"]    * optimal_ph_bonus
        + p["temp_bonus"]  * optimal_temp_bonus
        + p["hum_coeff"]   * humidity
        + p["sol_coeff"]   * solar_rad
        + p["om_coeff"]    * organic_matter
        + rng.normal(0, p["noise"], n)
    )
    lo, hi = p["yield_range"]
    yield_ = np.clip(yield_, lo, hi)

    return pd.DataFrame({
        "Nitrogen (kg/ha)":   nitrogen,
        "Phosphorus (kg/ha)": phosphorus,
        "Potassium (kg/ha)":  potassium,
        "Rainfall (mm)":      rainfall,
        "Temperature (°C)":   temperature,
        "Soil pH":            soil_ph,
        "Humidity (%)":       humidity,
        "Solar Radiation":    solar_rad,
        "Irrigation (mm)":    irrigation,
        "Organic Matter (%)": organic_matter,
        "Yield (t/ha)":       yield_,
    })

# ------------------------------------
# MODEL TRAINING
# ------------------------------------
@st.cache_resource
def train_model(crop_name: str):
    df = generate_dataset(crop_name)
    X  = df.drop(columns=["Yield (t/ha)"])
    y  = df["Yield (t/ha)"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {"R2": round(r2_score(y_test, y_pred), 3), "MAE": round(mean_absolute_error(y_test, y_pred), 3)}
    return model, shap.TreeExplainer(model), X_train, metrics

# ------------------------------------
# HUMAN SUMMARY TRANSLATION LAYER
# ------------------------------------
def generate_human_summary(shap_values, feature_names, predicted_yield, crop_name):
    df = pd.DataFrame({"feature": feature_names, "shap": shap_values}).sort_values("shap", ascending=False)
    pos = df[df["shap"] >  0.05]
    neg = df[df["shap"] < -0.05]
    crop_short = crop_name.split(" ")[1]
    rating = "thriving" if predicted_yield > 8 else ("growing well" if predicted_yield > 4 else "under stress")
    lines = [f"**Your {crop_short} field is {rating}**, with a predicted yield of **{predicted_yield:.2f} t/ha**.\n"]
    if not pos.empty:
        ctx = FEATURE_CONTEXT.get(pos.iloc[0]["feature"], {})
        lines.append(f"🟢 **Key strength:** {ctx.get('high_pos','favourable conditions').capitalize()}.")
    if not neg.empty:
        ctx = FEATURE_CONTEXT.get(neg.iloc[0]["feature"], {})
        lines.append(f"🔴 **Main concern:** {ctx.get('low_neg','suboptimal conditions').capitalize()}.")
    if len(pos) > 1:
        others = ", ".join([f.split(" (")[0] for f in pos["feature"].iloc[1:4].tolist()])
        lines.append(f"✅ **Also helping:** {others}.")
    lines.append("💡 **Recommendation:** Address the limiting factor above to unlock the most yield gain per input unit.")
    return "\n\n".join(lines)


def generate_seasonal_yield(crop_name, monthly_temp, monthly_rain, monthly_solar,
                             nitrogen, phosphorus, potassium, soil_ph,
                             humidity, irrigation, organic_matter):
    model, _, _, _ = train_model(crop_name)
    records = []
    for i, month in enumerate(MONTHS):
        row  = pd.DataFrame([[nitrogen, phosphorus, potassium, monthly_rain[i], monthly_temp[i],
                               soil_ph, humidity, monthly_solar[i], irrigation/12, organic_matter]], columns=FEATURES)
        pred = model.predict(row)[0]
        records.append({"Month": month, "Est. Monthly Yield": round(pred, 2),
                        "Temperature (°C)": monthly_temp[i], "Rainfall (mm)": monthly_rain[i],
                        "Solar Radiation": monthly_solar[i]})
    return pd.DataFrame(records)


def shorten(name):
    return name.replace(" (kg/ha)","").replace(" (mm)","").replace(" (%)","").replace(" (°C)","")

# ------------------------------------
# TRUST PANEL CHART
# ------------------------------------
def plot_trust_panel(shap_values, feature_names):
    df     = pd.DataFrame({"feature":[shorten(f) for f in feature_names],"shap":shap_values}).sort_values("shap")
    colors = ["#e63946" if v < 0 else "#2d6a4f" for v in df["shap"]]
    fig, ax = plt.subplots(figsize=(7,4.2)); fig.patch.set_facecolor("#fff"); ax.set_facecolor("#f8f9f2")
    bars = ax.barh(df["feature"], df["shap"], color=colors, height=0.6, edgecolor="white", linewidth=0.8)
    ax.axvline(0, color="#333", linewidth=1.1, linestyle="--", alpha=0.5)
    for bar, val in zip(bars, df["shap"]):
        ax.text(val+(0.02 if val>=0 else -0.02), bar.get_y()+bar.get_height()/2,
                f"{val:+.2f}", va="center", ha="left" if val>=0 else "right", fontsize=8.5, fontweight="600")
    ax.set_xlabel("Impact on Yield (t/ha)", fontsize=10, color="#555")
    ax.tick_params(axis="y", labelsize=9.5); ax.tick_params(axis="x", labelsize=8.5)
    ax.spines[["top","right","left"]].set_visible(False); ax.spines["bottom"].set_color("#ddd")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.legend(handles=[mpatches.Patch(color="#2d6a4f",label="Positive"),mpatches.Patch(color="#e63946",label="Negative")],
              loc="lower right", fontsize=8.5, framealpha=0.9, edgecolor="#ddd")
    plt.tight_layout(); return fig

# ------------------------------------
# TIME SERIES CHART
# ------------------------------------
def plot_time_series(ts_df, crop_name):
    color = CROP_PROFILES[crop_name]["color"]
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(8,5.5),sharex=True); fig.patch.set_facecolor("#fff")
    ax1.fill_between(ts_df["Month"], ts_df["Est. Monthly Yield"], alpha=0.25, color=color)
    ax1.plot(ts_df["Month"], ts_df["Est. Monthly Yield"], color=color, linewidth=2.5, marker="o", markersize=5)
    ax1.set_ylabel("Est. Monthly Yield (t/ha)", fontsize=9); ax1.set_facecolor("#f8f9f2")
    ax1.spines[["top","right"]].set_visible(False); ax1.grid(axis="y", linestyle=":", alpha=0.4)
    ax2r = ax2.twinx()
    ax2.bar(ts_df["Month"], ts_df["Rainfall (mm)"], color="#90e0ef", alpha=0.7, label="Rainfall")
    ax2r.plot(ts_df["Month"], ts_df["Temperature (°C)"], color="#e63946", linewidth=2, marker="s", markersize=4, label="Temp")
    ax2.set_ylabel("Rainfall (mm)", fontsize=9, color="#90e0ef"); ax2r.set_ylabel("Temp (°C)", fontsize=9, color="#e63946")
    ax2.set_facecolor("#f8f9f2"); ax2.spines[["top","right"]].set_visible(False)
    h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax2r.get_legend_handles_labels()
    ax2.legend(h1+h2, l1+l2, fontsize=8, loc="upper right")
    plt.tight_layout(); fig.suptitle("Seasonal Weather & Yield Forecast", fontsize=11, fontweight="bold", y=1.01)
    return fig

# ------------------------------------
# MULTI CROP COMPARISON
# ------------------------------------
def plot_multi_crop_comparison(input_data):
    crop_names = list(CROP_PROFILES.keys())
    yields = [round(train_model(c)[0].predict(input_data)[0], 2) for c in crop_names]
    labels = [c.split(" ")[1] for c in crop_names]
    colors = [CROP_PROFILES[c]["color"] for c in crop_names]
    fig, ax = plt.subplots(figsize=(6,3.5)); fig.patch.set_facecolor("#fff"); ax.set_facecolor("#f8f9f2")
    bars = ax.bar(labels, yields, color=colors, edgecolor="white", linewidth=1.2, width=0.55)
    for bar, val in zip(bars, yields):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1, f"{val:.2f}",
                ha="center", va="bottom", fontweight="700", fontsize=10)
    ax.set_ylabel("Predicted Yield (t/ha)", fontsize=9.5)
    ax.spines[["top","right","left"]].set_visible(False); ax.spines["bottom"].set_color("#ddd")
    ax.tick_params(labelsize=10); ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_title("Yield Comparison — Same Field Conditions", fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig, dict(zip(crop_names, yields))

# ------------------------------------
# GLOBAL FEATURE IMPORTANCE CHART
# ------------------------------------
def plot_global_importance(model, feature_names):
    imp    = model.feature_importances_
    df_imp = pd.DataFrame({"feature":[shorten(f) for f in feature_names],"importance":imp}).sort_values("importance")
    norm   = (df_imp["importance"]-df_imp["importance"].min())/(df_imp["importance"].max()-df_imp["importance"].min())
    fig, ax = plt.subplots(figsize=(6,4)); fig.patch.set_facecolor("#fff"); ax.set_facecolor("#f8f9f2")
    ax.barh(df_imp["feature"], df_imp["importance"], color=[plt.cm.YlGn(0.35+0.65*v) for v in norm], height=0.65, edgecolor="white")
    ax.set_xlabel("Mean Decrease in Impurity", fontsize=9, color="#555")
    ax.spines[["top","right","left"]].set_visible(False); ax.spines["bottom"].set_color("#ddd")
    ax.grid(axis="x", linestyle=":", alpha=0.4); plt.tight_layout()
    return fig

# ------------------------------------
# PDF REPORT GENERATION FEATURE
# ------------------------------------
def generate_pdf_report(crop_name, predicted_yield, shap_values, feature_names,
                         input_data, summary_text, metrics, ts_df=None):
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.units import cm

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm, leftMargin=2*cm, rightMargin=2*cm)
    S   = getSampleStyleSheet()
    title_s = ParagraphStyle("T",  parent=S["Title"],   fontSize=22, textColor=rl_colors.HexColor("#2d6a4f"), spaceAfter=6)
    h2_s    = ParagraphStyle("H2", parent=S["Heading2"],fontSize=13, textColor=rl_colors.HexColor("#1b4332"), spaceAfter=4)
    body_s  = ParagraphStyle("B",  parent=S["Normal"],  fontSize=10, leading=15, spaceAfter=6)
    small_s = ParagraphStyle("Sm", parent=S["Normal"],  fontSize=8.5,textColor=rl_colors.HexColor("#666666"))

    story = []
    story.append(Paragraph("CropIQ — Field Report", title_s))
    story.append(Paragraph(f"Crop: {crop_name}  |  Predicted Yield: <b>{predicted_yield:.2f} t/ha</b>"
                            f"  |  R2: {metrics['R2']}  |  MAE: {metrics['MAE']} t/ha", body_s))
    story.append(HRFlowable(width="100%", thickness=1.5, color=rl_colors.HexColor("#52b788"), spaceAfter=10))

    story.append(Paragraph("Human Summary", h2_s))
    story.append(Paragraph(summary_text.replace("**","").replace("\n\n","<br/><br/>"), body_s))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("Field Input Conditions", h2_s))
    tdata = [["Feature","Your Value","SHAP Impact (t/ha)"]]
    for feat, val, sv in zip(feature_names, input_data.values[0], shap_values):
        tdata.append([feat, f"{val:.1f}", f"{'+' if sv>0 else ''}{sv:.3f}"])
    t = Table(tdata, colWidths=[7*cm, 4*cm, 5*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),rl_colors.HexColor("#2d6a4f")),("TEXTCOLOR",(0,0),(-1,0),rl_colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),9),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[rl_colors.HexColor("#f8f9f2"),rl_colors.white]),
        ("GRID",(0,0),(-1,-1),0.5,rl_colors.HexColor("#dddddd")),("ALIGN",(1,0),(-1,-1),"CENTER"),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
    ]))
    story.append(t); story.append(Spacer(1,0.5*cm))

    story.append(Paragraph("Trust Panel — Influence Breakdown", h2_s))
    fig    = plot_trust_panel(shap_values, feature_names)
    ibuf   = BytesIO(); fig.savefig(ibuf, format="png", dpi=130, bbox_inches="tight"); ibuf.seek(0); plt.close(fig)
    story.append(RLImage(ibuf, width=15*cm, height=7*cm)); story.append(Spacer(1, 0.4*cm))

    if ts_df is not None:
        story.append(Paragraph("Seasonal Forecast", h2_s))
        tsdata = [["Month","Est. Monthly Yield (t/ha)","Temp (°C)","Rainfall (mm)"]]
        for _, row in ts_df.iterrows():
            tsdata.append([row["Month"],f"{row['Est. Monthly Yield']:.2f}",f"{row['Temperature (°C)']:.1f}",f"{row['Rainfall (mm)']:.0f}"])
        tst = Table(tsdata, colWidths=[3*cm,5.5*cm,3.5*cm,4*cm])
        tst.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),rl_colors.HexColor("#52b788")),("TEXTCOLOR",(0,0),(-1,0),rl_colors.white),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),9),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[rl_colors.HexColor("#f0fdf4"),rl_colors.white]),
            ("GRID",(0,0),(-1,-1),0.5,rl_colors.HexColor("#dddddd")),("ALIGN",(1,0),(-1,-1),"CENTER"),
            ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
        ]))
        story.append(tst); story.append(Spacer(1,0.4*cm))

    story.append(HRFlowable(width="100%",thickness=0.5,color=rl_colors.HexColor("#cccccc"),spaceAfter=6))
    story.append(Paragraph("CropIQ · RandomForest + SHAP · Synthetic data — for demonstration only.", small_s))
    doc.build(story); buf.seek(0); return buf


CSV_TEMPLATE = pd.DataFrame([{f: round(FEATURE_RANGES[f][2], 1) for f in FEATURES}] * 3)

# ------------------------------------
# BATCH PREDICTION FEATURE
# ------------------------------------
def run_batch_predictions(df_upload, crop_name):
    missing = [f for f in FEATURES if f not in df_upload.columns]
    if missing:
        return None, f"Missing columns: {', '.join(missing)}"
    model, _, _, _ = train_model(crop_name)
    preds  = model.predict(df_upload[FEATURES])
    df_out = df_upload.copy()
    df_out["Predicted Yield (t/ha)"] = np.round(preds, 3)
    df_out["Yield Rating"] = pd.cut(preds, bins=[0,4,7,11,999],
                                    labels=["⚠️ Low","🌱 Moderate","🌾 Good","⭐ Excellent"])
    return df_out, None

# ------------------------------------
# MAIN APP
# ------------------------------------
def main():
    # ---HERO -------------------------=
    st.markdown("""
    <div class="hero-box">
        <h1>🌾 CropIQ v2 — Yield Prediction & Explainability</h1>
        <p> Enter your field conditions, get an AI-powered yield forecast, and understand <em>exactly</em> why — in plain language.</p>
        <p>Multi-crop forecasting · SHAP Trust Panel · Seasonal Time-Series · CSV Batch Mode · PDF Export</p>
    </div>""", unsafe_allow_html=True)

    # --- SIDEBAR: Field Inputs -----------------
    st.sidebar.markdown("### 🌿 Crop & Field Settings")
    crop_name = st.sidebar.selectbox("Select Crop", list(CROP_PROFILES.keys()), index=0)
    st.sidebar.markdown(f"*{CROP_PROFILES[crop_name]['description']}*")
    st.sidebar.divider()

    # ---upload csv sidebar-----------------
    st.sidebar.markdown("#### 📂 Upload your field condition CSV")

    sidebar_csv = st.sidebar.file_uploader(
        "Upload single-field CSV", type=["csv"], label_visibility="collapsed"
    )

    # Defaults — overridden by CSV if uploaded
    defaults = {f: float(FEATURE_RANGES[f][2]) for f in FEATURES}

    if sidebar_csv is not None:
        try:
            df_side = pd.read_csv(sidebar_csv)
            missing = [f for f in FEATURES if f not in df_side.columns]
            if missing:
                st.sidebar.error(f"Missing columns: {', '.join(missing)}")
            else:
                row = df_side.iloc[0]
                for f in FEATURES:
                    lo, hi, _ = FEATURE_RANGES[f]
                    defaults[f] = float(np.clip(row[f], lo, hi))
                if len(df_side) > 1:
                    st.sidebar.info(f"CSV has {len(df_side)} rows — showing row 1. Use the Batch tab for all rows.")
                else:
                    st.sidebar.success("Field values loaded from CSV.")
        except Exception as e:
            st.sidebar.error(f"Could not read CSV: {e}")

    tmpl_buf = BytesIO()
    CSV_TEMPLATE.iloc[:1].to_csv(tmpl_buf, index=False)
    st.sidebar.divider()

    # ---Manual input of features---------------
    st.sidebar.markdown("**Manual input of field conditions**")
    nitrogen    = st.sidebar.slider("Nitrogen (kg/ha)",             0,   140, 80)
    phosphorus  = st.sidebar.slider("Phosphorus (kg/ha)",           0,   100, 50)
    potassium   = st.sidebar.slider("Potassium (kg/ha)",            0,   120, 60)
    rainfall    = st.sidebar.slider("Rainfall (mm/year)",         200,  1500,750)
    temperature = st.sidebar.slider("Temperature (°C)",            10,    40, 25)
    soil_ph     = st.sidebar.slider("Soil pH",                    4.5,   9.0, 6.5)
    humidity    = st.sidebar.slider("Humidity (%)",                20,    95, 65)
    solar_rad   = st.sidebar.slider("Solar Radiation (MJ/m2/day)", 10,    30, 20)
    irrigation  = st.sidebar.slider("Irrigation (mm/season)",       0,   500,150)
    organic_mat = st.sidebar.slider("Organic Matter (%)",          0.5,   6.0, 3.0)

    input_data = pd.DataFrame([[nitrogen,phosphorus,potassium,rainfall,temperature,
                                 soil_ph,humidity,solar_rad,irrigation,organic_mat]], columns=FEATURES)

    model, explainer, X_train, metrics = train_model(crop_name)
    predicted_yield = model.predict(input_data)[0]
    shap_values     = explainer.shap_values(input_data)[0]

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-card"><h3>Predicted Yield</h3><p>{predicted_yield:.2f} t/ha</p></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><h3>Model R2</h3><p>{metrics["R2"]}</p></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card"><h3>Mean Abs Error</h3><p>{metrics["MAE"]} t/ha</p></div>', unsafe_allow_html=True)
    with c4:
        top = FEATURES[np.argmax(np.abs(shap_values))].split(" (")[0]
        st.markdown(f'<div class="metric-card"><h3>Top SHAP Driver</h3><p style="font-size:1.2rem">{top}</p></div>', unsafe_allow_html=True)

    st.divider()
    # ---Carousel-----------------
    tab1,tab2,tab3,tab4 = st.tabs(["🔍 Trust Panel","📂 CSV Batch Upload","📅 Time-Series Forecast","🌽 Multi-Crop Compare"])

    # ---Trust Panel------------
    with tab1:
        st.markdown('<div class="trust-panel">', unsafe_allow_html=True)
        st.markdown("## 🔍 Trust Panel")
        left, right = st.columns([1.1, 1])
        with left:
            st.markdown("### 📊 Influence Breakdown")
            st.pyplot(plot_trust_panel(shap_values, FEATURES))
            pos_f = sorted([(FEATURES[i],shap_values[i]) for i in range(len(FEATURES)) if shap_values[i]>0.05],  key=lambda x:-x[1])
            neg_f = sorted([(FEATURES[i],shap_values[i]) for i in range(len(FEATURES)) if shap_values[i]<-0.05], key=lambda x: x[1])
            if pos_f: st.markdown("**Positive:** "+" ".join([f'<span class="pos-tag">✅ {f.split(" (")[0]}</span>' for f,_ in pos_f[:5]]), unsafe_allow_html=True)
            if neg_f: st.markdown("**Negative:** "+" ".join([f'<span class="neg-tag">⚠️ {f.split(" (")[0]}</span>' for f,_ in neg_f[:5]]), unsafe_allow_html=True)
        with right:
            st.markdown("### 📝 Human Summary")
            summary = generate_human_summary(shap_values, FEATURES, predicted_yield, crop_name)
            st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)
            st.markdown("### 📈 SHAP Impact Table")
            st.dataframe(pd.DataFrame({
                "Feature":    [f.split(" (")[0] for f in FEATURES],
                "Value":      [f"{v:.1f}" for v in input_data.values[0]],
                "SHAP (t/ha)":[f"{'+' if s>0 else ''}{s:.3f}" for s in shap_values],
            }), hide_index=True, use_container_width=True)
            st.markdown("### 📄 Export Report")
            if st.button("⬇️ Generate PDF Report", type="primary"):
                with st.spinner("Building PDF..."):
                    ts_df_exp = st.session_state.get("ts_df", None)
                    pdf_buf = generate_pdf_report(crop_name, predicted_yield, shap_values, FEATURES,
                                                  input_data, summary, metrics, ts_df=ts_df_exp)
                b64  = base64.b64encode(pdf_buf.read()).decode()
                href = (f'<a href="data:application/pdf;base64,{b64}" download="CropIQ_Report.pdf" '
                        f'style="background:#2d6a4f;color:white;padding:10px 20px;border-radius:8px;'
                        f'text-decoration:none;font-weight:600;">📥 Download PDF</a>')
                st.markdown(href, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        with st.expander("🌐 Global Feature Importance", expanded=False):
            ca, cb = st.columns([1,1.5])
            with ca:
                st.markdown("""
**Why Random Forest?**

| Property | Deep Learning | **Random Forest** | Linear |
|---|---|---|---|
| Accuracy | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| Interpretable | ❌ | ✅ SHAP | ✅ |
| Non-linear | ✅ | ✅ | ❌ |
""")
            with cb: st.pyplot(plot_global_importance(model, FEATURES))

# ---CSV Batch Field Upload--------------------
    with tab2:
        st.markdown("## 📂 CSV Batch Field Upload")
        tmpl_buf = BytesIO(); CSV_TEMPLATE.to_csv(tmpl_buf, index=False)
        st.download_button("⬇️ Download CSV Template", tmpl_buf.getvalue(), "cropiq_template.csv", "text/csv")
        st.markdown('<div class="upload-zone">📁 Drag & drop your CSV, or click to browse</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload field CSV", type=["csv"], label_visibility="collapsed")
        if uploaded:
            df_raw = pd.read_csv(uploaded)
            st.markdown(f"**{len(df_raw)} fields loaded.** Preview:"); st.dataframe(df_raw.head(5), use_container_width=True)
            df_result, err = run_batch_predictions(df_raw, crop_name)
            if err:
                st.error(f"❌ {err}")
            else:
                st.success(f"✅ Predictions complete for {len(df_result)} fields.")
                r1,r2,r3 = st.columns(3)
                r1.metric("Avg Yield",  f"{df_result['Predicted Yield (t/ha)'].mean():.2f} t/ha")
                r2.metric("Best Field", f"{df_result['Predicted Yield (t/ha)'].max():.2f} t/ha")
                r3.metric("Worst Field",f"{df_result['Predicted Yield (t/ha)'].min():.2f} t/ha")
                fig_d, ax = plt.subplots(figsize=(7,3)); fig_d.patch.set_facecolor("#fff"); ax.set_facecolor("#f8f9f2")
                ax.hist(df_result["Predicted Yield (t/ha)"],bins=20,color="#52b788",edgecolor="white",linewidth=0.8)
                ax.set_xlabel("Predicted Yield (t/ha)",fontsize=10); ax.set_ylabel("Fields",fontsize=10)
                ax.set_title("Yield Distribution",fontweight="bold"); ax.spines[["top","right"]].set_visible(False); plt.tight_layout()
                st.pyplot(fig_d)
                st.dataframe(df_result, use_container_width=True, hide_index=True)
                out_buf = BytesIO(); df_result.to_csv(out_buf, index=False)
                st.download_button("⬇️ Download Results CSV", out_buf.getvalue(), "cropiq_results.csv", "text/csv", type="primary")
        else:
            st.info("Required columns: " + ", ".join(FEATURES))

# ---Seasonal Time-Series Forecast----------------
    with tab3:
        st.markdown("## Seasonal Time-Series Forecast")
        st.markdown("Enter a location to fetch real climate data from **Open-Meteo**.")

        loc_col, btn_col = st.columns([3, 1])
        with loc_col:
            location_query = st.text_input("Location", placeholder="e.g. Lagos, Nairobi, Mumbai",
                                           label_visibility="collapsed")
        with btn_col:
            fetch_clicked = st.button("Fetch Weather", type="primary", use_container_width=True)

        @st.cache_data(ttl=3600)
        def geocode_location(query):
            import urllib.request, json, urllib.parse
            url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(query)}&count=1&language=en&format=json"
            with urllib.request.urlopen(url, timeout=8) as r:
                data = json.loads(r.read())
            if not data.get("results"):
                return None
            res = data["results"][0]
            return {"name": res["name"], "country": res.get("country", ""), "lat": res["latitude"],
                    "lon": res["longitude"]}

        @st.cache_data(ttl=3600)
        def fetch_climate_data(lat, lon):
            import urllib.request, json
            from datetime import date
            from collections import defaultdict
            year = date.today().year - 1
            params = (f"latitude={lat}&longitude={lon}"
                      f"&start_date={year}-01-01&end_date={year}-12-31"
                      f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,shortwave_radiation_sum"
                      f"&timezone=auto")
            with urllib.request.urlopen(f"https://archive-api.open-meteo.com/v1/archive?{params}", timeout=15) as r:
                data = json.loads(r.read())
            daily = data["daily"]
            m_temp, m_rain, m_solar = defaultdict(list), defaultdict(list), defaultdict(list)
            for d, tmax, tmin, p, s in zip(daily["time"], daily["temperature_2m_max"],
                                           daily["temperature_2m_min"], daily["precipitation_sum"],
                                           daily["shortwave_radiation_sum"]):
                mo = int(d[5:7]) - 1
                if tmax is not None and tmin is not None: m_temp[mo].append((tmax + tmin) / 2)
                if p is not None: m_rain[mo].append(p)
                if s is not None: m_solar[mo].append(s)
            return (
                [round(sum(m_temp[i]) / max(len(m_temp[i]), 1), 1) for i in range(12)],
                [round(sum(m_rain[i]), 0) for i in range(12)],
                [round(sum(m_solar[i]) / max(len(m_solar[i]), 1), 1) for i in range(12)],
            )

        if fetch_clicked and location_query.strip():
            with st.spinner(f"Locating '{location_query}'..."):
                geo = geocode_location(location_query.strip())
            if not geo:
                st.error("Location not found. Try a different city name.")
            else:
                with st.spinner(f"Fetching climate for {geo['name']}, {geo['country']}..."):
                    try:
                        mt, mr, ms = fetch_climate_data(geo["lat"], geo["lon"])
                        st.session_state["weather_data"] = {"temp": mt, "rain": mr, "solar": ms}
                        st.session_state["weather_location"] = f"{geo['name']}, {geo['country']}"
                        st.success(f"Loaded climate data for **{geo['name']}, {geo['country']}**")
                    except Exception as e:
                        st.error(f"Failed to fetch weather: {e}")

        if "weather_data" in st.session_state:
            wd = st.session_state["weather_data"]
            monthly_temp, monthly_rain, monthly_solar = wd["temp"], wd["rain"], wd["solar"]
            st.info(f"Weather source: {st.session_state.get('weather_location', '')} via Open-Meteo archive")
            with st.expander("Raw Climate Data", expanded=False):
                st.dataframe(pd.DataFrame({
                    "Month": MONTHS, "Avg Temp (C)": monthly_temp,
                    "Rainfall (mm)": monthly_rain, "Solar Rad": monthly_solar,
                }), hide_index=True, use_container_width=True)
            ts_df = generate_seasonal_yield(crop_name, monthly_temp, monthly_rain, monthly_solar,
                                            nitrogen, phosphorus, potassium, soil_ph, humidity, irrigation, organic_mat)
            st.session_state["ts_df"] = ts_df
            st.pyplot(plot_time_series(ts_df, crop_name))
            ca, cb, cc = st.columns(3)
            ca.metric("Peak Month", ts_df.loc[ts_df["Est. Monthly Yield"].idxmax(), "Month"])
            cb.metric("Peak Yield", f"{ts_df['Est. Monthly Yield'].max():.2f} t/ha")
            cc.metric("Season Avg", f"{ts_df['Est. Monthly Yield'].mean():.2f} t/ha")
            st.dataframe(ts_df.set_index("Month").T.round(2), use_container_width=True)
        else:
            st.info("Enter a city name and click Fetch Weather to load real climate data.")

# ---Multi-Crop Yield Comparison------------------
    with tab4:
        st.markdown("## 🌽 Multi-Crop Yield Comparison")
        with st.spinner("Running all 4 crop models..."):
            fig_cmp, yield_map = plot_multi_crop_comparison(input_data)
        st.pyplot(fig_cmp)
        best = max(yield_map, key=yield_map.get)
        st.success(f"**Best crop for these conditions:** {best} — **{yield_map[best]:.2f} t/ha**")
        st.dataframe(pd.DataFrame([{
            "Crop":crop,"Predicted Yield (t/ha)":f"{yld:.2f}",
            "Optimal Temp":CROP_PROFILES[crop]["optimal_temp"],
            "Optimal pH":CROP_PROFILES[crop]["optimal_ph"],
            "Notes":CROP_PROFILES[crop]["description"]
        } for crop,yld in yield_map.items()]), hide_index=True, use_container_width=True)
        st.markdown("### SHAP Drivers per Crop")
        for col_idx, (crop, _) in zip(st.columns(4), yield_map.items()):
            m, exp, _, _ = train_model(crop)
            sv = exp.shap_values(input_data)[0]
            with col_idx:
                st.markdown(f"**{crop}**")
                st.dataframe(pd.DataFrame({
                    "Feature":[f.split(" (")[0] for f in FEATURES],
                    "SHAP":[f"{'+' if s>0 else ''}{s:.2f}" for s in sv],
                }).sort_values("SHAP",ascending=False), hide_index=True, use_container_width=True, height=280)


    st.caption("CropIQ v2 · RandomForest + SHAP · Synthetic data — demonstration only.")

if __name__ == "__main__":
    main()
