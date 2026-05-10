"""
app.py
======
Streamlit Crime Intelligence Dashboard
Run: streamlit run app.py
"""

import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st
import folium
from folium.plugins import HeatMap, HeatMapWithTime
from streamlit_folium import st_folium
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title  = "LAPD Crime Intelligence",
    page_icon   = "🚔",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# Custom CSS for dark theme consistency
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .metric-card {
        background: #1e2329;
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #e84545;
        margin: 5px 0;
    }
    .risk-high   { color: #e84545; font-weight: bold; font-size: 1.3em; }
    .risk-medium { color: #ffa500; font-weight: bold; font-size: 1.3em; }
    .risk-low    { color: #4caf50; font-weight: bold; font-size: 1.3em; }
    h1, h2, h3 { color: #ffffff; }
    .stTabs [data-baseweb="tab"] { color: #aaaaaa; }
    .stTabs [aria-selected="true"] { color: #e84545; border-bottom-color: #e84545; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CACHED DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading crime data...")
def load_clean_data():
    path = DATA_DIR / "clean_crime.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_resource(show_spinner="Loading prediction models...")
def load_models(granularity="month"):
    models = {}
    try:
        models["hotspot"]  = joblib.load(MODEL_DIR / f"hotspot_model_{granularity}.pkl")
        models["features"] = joblib.load(MODEL_DIR / f"hotspot_features_{granularity}.pkl")
        # Verify feature count matches at startup — catches stale pkl files
        n_model = models["hotspot"].n_features_in_
        n_saved = len(models["features"])
        if n_model != n_saved:
            import streamlit as _st
            _st.warning(
                f"⚠️ Feature count mismatch: model expects {n_model} features, "
                f"saved list has {n_saved}. Re-run training notebook to fix."
            )
        else:
            print(f"✅ Hotspot model OK: {n_model} features.")
    except FileNotFoundError:
        pass
    try:
        models["crime_type"] = joblib.load(MODEL_DIR / "crime_type_model.pkl")
        models["le"]         = joblib.load(MODEL_DIR / "crime_type_label_encoder.pkl")
        models["ct_features"]= joblib.load(MODEL_DIR / "crime_type_features.pkl")
    except FileNotFoundError:
        pass
    return models


@st.cache_data
def load_centroids():
    path = DATA_DIR / "area_centroids.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_data
def load_forecasts():
    path = DATA_DIR / "area_forecasts.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_data
def load_timeseries():
    path = DATA_DIR / "monthly_timeseries.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_data
def load_clusters():
    path = DATA_DIR / "area_clusters.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_data
def load_agg(granularity="month"):
    path = DATA_DIR / f"agg_{granularity}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


# ══════════════════════════════════════════════════════════════════════════════
#  PREDICT HEATMAP  (inline — avoids re-importing model_hotspot in prod)
# ══════════════════════════════════════════════════════════════════════════════
def predict_area_scores(model, agg_df, feature_cols,
                        target_year, target_period, granularity="month"):
    from model_hotspot import predict_heatmap
    return predict_heatmap(model, agg_df, feature_cols,
                           target_year, target_period, granularity)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/"
                     "LAPD_badge.svg/200px-LAPD_badge.svg.png", width=80)
    st.sidebar.title("🚔 Crime Intelligence")
    st.sidebar.markdown("---")

    granularity = st.sidebar.radio("Time Granularity", ["Month", "Week"],
                                   horizontal=True).lower()

    st.sidebar.markdown("### 🗓️ Prediction Window")
    target_year = st.sidebar.selectbox("Year", list(range(2020, 2027)),
                                       index=4)

    if granularity == "month":
        month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]
        sel_period  = st.sidebar.selectbox("Month", month_names, index=0)
        target_period = month_names.index(sel_period) + 1
    else:
        target_period = st.sidebar.slider("Week", 1, 52, 1)

    predict_btn = st.sidebar.button("🔮 Predict Crime Hotspots",
                                    type="primary", use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Area Filter")
    area_filter = st.sidebar.slider("Highlight Top N Areas", 1, 21, 5)

    return granularity, target_year, target_period, predict_btn, area_filter


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
def render_heatmap_tab(df, models, centroids, agg_df,
                       granularity, target_year, target_period,
                       predict_clicked, area_filter):
    st.header("🗺️ Crime Hotspot Heatmap")

    if not predict_clicked:
        st.info("👈 Configure prediction parameters in the sidebar and click **Predict Crime Hotspots**")

    # Always show current historical heatmap
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}" if df is not None else "N/A")
    with col2:
        if df is not None:
            st.metric("Years Covered",
                      f"{int(df['year'].min())}–{int(df['year'].max())}")
    with col3:
        if df is not None:
            st.metric("Unique Areas", df["area"].nunique())
    with col4:
        if df is not None and "crime_category" in df.columns:
            top_cat = df["crime_category"].value_counts().index[0]
            st.metric("Top Crime Type", top_cat)

    st.markdown("---")

    if df is None or centroids is None:
        st.error("❌ Data not found. Run `python train_all.py <csv>` first.")
        return

    # ── Decide which scores to show ───────────────────────────────────────────
    if predict_clicked and "hotspot" in models and agg_df is not None:
        with st.spinner("🔮 Generating predictions..."):
            pred_df = predict_area_scores(
                models["hotspot"], agg_df, models["features"],
                target_year, target_period, granularity
            )
            merged = pred_df.merge(centroids, on="area", how="left").dropna(
                subset=["lat","lon"]
            )
            score_col  = "risk_score"
            title_suffix = f"— Predicted ({target_year}, {granularity.capitalize()} {target_period})"
    else:
        # Historical distribution
        hist_scores = df.groupby("area").size().reset_index(name="crime_count")
        mn = hist_scores["crime_count"].min()
        mx = hist_scores["crime_count"].max()
        hist_scores["risk_score"] = (hist_scores["crime_count"] - mn) / (mx - mn + 1e-9)
        merged     = hist_scores.merge(centroids, on="area", how="left").dropna(
            subset=["lat","lon"]
        )
        score_col  = "risk_score"
        title_suffix = "— Historical Distribution"

    # ── Build Folium map ──────────────────────────────────────────────────────
    la_center = [34.052, -118.243]
    m = folium.Map(location=la_center, zoom_start=10,
                   tiles="CartoDB dark_matter")

    # Heat data: [lat, lon, weight]
    heat_data = merged[["lat","lon", score_col]].dropna().values.tolist()
    HeatMap(
        heat_data,
        radius=30,
        blur=20,
        min_opacity=0.3,
        gradient={0.0: "blue", 0.4: "lime", 0.65: "yellow",
                  0.8: "orange", 1.0: "red"},
    ).add_to(m)

    # Circle markers for top N areas
    top_areas = merged.nlargest(area_filter, score_col)
    for _, row in top_areas.iterrows():
        risk = row[score_col]
        color = "#e84545" if risk > 0.7 else "#ffa500" if risk > 0.4 else "#4caf50"
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=10 + risk * 15,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(
                f"<b>Area {int(row['area'])}</b><br>"
                f"Risk Score: {risk:.2f}<br>"
                f"{'Predicted Count: ' + str(int(row.get('predicted_crime_count', 0))) if predict_clicked else ''}",
                max_width=200,
            ),
            tooltip=f"Area {int(row['area'])} | Risk: {risk:.2f}",
        ).add_to(m)

    # Title overlay
    title_html = f"""
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
         background: rgba(14,17,23,0.9); padding: 8px 16px; border-radius: 6px;
         border: 1px solid #e84545; color: white; font-size: 14px; font-weight: bold;
         z-index: 9999;">
         🚔 LAPD Crime Heatmap {title_suffix}
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    st_folium(m, width="100%", height=550, returned_objects=[])

    # ── Risk table ────────────────────────────────────────────────────────────
    if predict_clicked:
        st.markdown("### 🎯 Area Risk Rankings")
        display_df = merged[["area", score_col]].copy()
        if "predicted_crime_count" in merged.columns:
            display_df["Predicted Incidents"] = merged["predicted_crime_count"].astype(int)
        display_df.columns = ["Area", "Risk Score"] + (
            ["Predicted Incidents"] if "predicted_crime_count" in merged.columns else []
        )
        display_df = display_df.sort_values("Risk Score", ascending=False)
        display_df["Risk Level"] = display_df["Risk Score"].apply(
            lambda x: "🔴 HIGH" if x > 0.7 else "🟠 MEDIUM" if x > 0.4 else "🟢 LOW"
        )
        st.dataframe(display_df.style.background_gradient(
            subset=["Risk Score"], cmap="YlOrRd"
        ), use_container_width=True, height=350)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — CRIME INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
def render_insights_tab(df, models):
    st.header("📊 Crime Insights & Risk Factors")

    if df is None:
        st.error("Data not loaded.")
        return

    col_left, col_right = st.columns([1, 1])

    with col_left:
        # ── Top crimes bar ────────────────────────────────────────────────────
        st.subheader("🔝 Top Crime Types")
        if "crime_category" in df.columns:
            cat_counts = df["crime_category"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Count"]
            fig = px.bar(
                cat_counts, x="Count", y="Category", orientation="h",
                color="Count", color_continuous_scale="YlOrRd",
                template="plotly_dark",
            )
            fig.update_layout(showlegend=False, height=350,
                              margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig, use_container_width=True)

        # ── Top 20 crime codes ────────────────────────────────────────────────
        st.subheader("📋 Top 20 Crime Codes")
        top20 = df.groupby("crm_cd").size().nlargest(20).reset_index()
        top20.columns = ["Crime Code", "Count"]
        if "crm_cd_desc" in df.columns:
            desc = df.drop_duplicates("crm_cd")[["crm_cd","crm_cd_desc"]]
            top20 = top20.merge(desc, left_on="Crime Code", right_on="crm_cd", how="left")
            top20["Label"] = top20["Crime Code"].astype(str) + " — " + \
                             top20["crm_cd_desc"].fillna("").str[:25]
        else:
            top20["Label"] = top20["Crime Code"].astype(str)
        fig2 = px.bar(
            top20.sort_values("Count"), x="Count", y="Label", orientation="h",
            color="Count", color_continuous_scale="YlOrRd",
            template="plotly_dark",
        )
        fig2.update_layout(showlegend=False, height=450,
                           margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig2, use_container_width=True)

    with col_right:
        # ── Victim sex breakdown ──────────────────────────────────────────────
        if "vict_sex" in df.columns:
            st.subheader("👥 Victim Sex Distribution")
            sex_map = {"M": "Male", "F": "Female", "X": "Unknown/N-A"}
            sex_data = df["vict_sex"].map(sex_map).value_counts().reset_index()
            sex_data.columns = ["Sex", "Count"]
            fig3 = px.pie(sex_data, names="Sex", values="Count",
                          color_discrete_sequence=["#5C9EFF","#E84545","#888888"],
                          template="plotly_dark", hole=0.4)
            fig3.update_layout(height=300, margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig3, use_container_width=True)

        # ── Victim descent top 8 ──────────────────────────────────────────────
        if "descent_label" in df.columns:
            st.subheader("🌍 Victim Descent (Top 8)")
            descent_data = df["descent_label"].value_counts().head(8).reset_index()
            descent_data.columns = ["Descent", "Count"]
            fig4 = px.bar(
                descent_data, x="Descent", y="Count",
                color="Count", color_continuous_scale="YlOrRd",
                template="plotly_dark",
            )
            fig4.update_layout(showlegend=False, height=300,
                               margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig4, use_container_width=True)

        # ── Hourly heatmap ────────────────────────────────────────────────────
        st.subheader("⏰ Crime Hour × Day of Week Heatmap")
        pivot = df.groupby(["day_of_week","hour"]).size().unstack(fill_value=0)
        pivot.index = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        fig5 = px.imshow(pivot, color_continuous_scale="YlOrRd",
                         aspect="auto", template="plotly_dark",
                         labels={"color": "Incidents"})
        fig5.update_layout(height=280, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig5, use_container_width=True)

    # ── Live crime type predictor ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔮 Live Crime Type Predictor")
    st.caption("Select context → see most likely crime types for that situation")

    if "crime_type" in models:
        pc1, pc2, pc3, pc4, pc5 = st.columns(5)
        with pc1: pred_area = st.selectbox("Area", sorted(df["area"].dropna().unique().astype(int)))
        with pc2: pred_hour = st.slider("Hour", 0, 23, 12)
        with pc3: pred_month= st.slider("Month", 1, 12, 6)
        with pc4: pred_dow  = st.selectbox("Day", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
        with pc5: pred_year = st.selectbox("Year ", list(range(2020, 2027)), index=4)

        dow_map = {"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6}

        from model_crime_type import predict_top_crimes
        top_crimes = predict_top_crimes(
            models["crime_type"], models["le"], models["ct_features"],
            area=pred_area, hour=pred_hour, month=pred_month,
            day_of_week=dow_map[pred_dow], year=pred_year
        )

        fig6 = px.bar(
            top_crimes, x="probability", y="crime_category", orientation="h",
            color="probability", color_continuous_scale="YlOrRd",
            template="plotly_dark",
            labels={"probability": "Probability", "crime_category": "Crime Type"},
        )
        fig6.update_layout(showlegend=False, height=280,
                           margin=dict(l=0,r=0,t=20,b=0),
                           title=f"Top Crime Types — Area {pred_area}, "
                                 f"{pred_dow} {pred_hour:02d}:00")
        st.plotly_chart(fig6, use_container_width=True)
    else:
        st.info("Train the crime type model first: `python train_all.py <csv>`")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — TRENDS
# ══════════════════════════════════════════════════════════════════════════════
def render_trends_tab(df, ts_df, forecasts_df):
    st.header("📈 Crime Trends & Forecasts")

    if df is None:
        st.error("Data not loaded.")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        # ── City-wide trend ───────────────────────────────────────────────────
        st.subheader("🌍 City-Wide Monthly Crime Trend")
        if ts_df is not None:
            city_ts = ts_df.groupby("ds")["y"].sum().reset_index()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=city_ts["ds"], y=city_ts["y"],
                mode="lines+markers", name="Observed",
                line=dict(color="#e84545", width=2),
                marker=dict(size=5),
            ))

            # Add forecast if available
            if forecasts_df is not None and len(forecasts_df) > 0:
                city_fc = forecasts_df.groupby("ds").agg(
                    yhat=("yhat","sum"),
                    yhat_lower=("yhat_lower","sum"),
                    yhat_upper=("yhat_upper","sum"),
                ).reset_index()
                future_fc = city_fc[city_fc["ds"] > city_ts["ds"].max()]
                if len(future_fc) > 0:
                    fig.add_trace(go.Scatter(
                        x=future_fc["ds"], y=future_fc["yhat"],
                        mode="lines", name="Forecast",
                        line=dict(color="gold", width=2, dash="dash"),
                    ))
                    fig.add_trace(go.Scatter(
                        x=pd.concat([future_fc["ds"], future_fc["ds"][::-1]]),
                        y=pd.concat([future_fc["yhat_upper"], future_fc["yhat_lower"][::-1]]),
                        fill="toself", fillcolor="rgba(255,215,0,0.1)",
                        line=dict(color="rgba(255,255,255,0)"),
                        name="95% CI",
                    ))

            fig.update_layout(template="plotly_dark", height=380,
                              margin=dict(l=0,r=0,t=30,b=0),
                              legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)

        # ── YoY change ────────────────────────────────────────────────────────
        st.subheader("📊 Year-over-Year Crime Change")
        if "crime_category" in df.columns:
            yoy = df.groupby(["year","crime_category"]).size().unstack(fill_value=0)
            yoy_pct = yoy.pct_change() * 100
            yoy_pct = yoy_pct.dropna().reset_index().melt(
                id_vars="year", var_name="Category", value_name="Change %"
            )
            fig2 = px.bar(
                yoy_pct, x="year", y="Change %", color="Category",
                barmode="group", template="plotly_dark",
                color_discrete_sequence=px.colors.sequential.YlOrRd,
            )
            fig2.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4)
            fig2.update_layout(height=330, margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig2, use_container_width=True)

    with col2:
        # ── Seasonality ───────────────────────────────────────────────────────
        st.subheader("🗓️ Seasonal Pattern")
        monthly_avg = df.groupby("month").size() / df["year"].nunique()
        month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]
        fig3 = go.Figure(go.Scatterpolar(
            r=monthly_avg.values,
            theta=month_names,
            fill="toself",
            fillcolor="rgba(232,69,69,0.3)",
            line=dict(color="#e84545"),
        ))
        fig3.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            template="plotly_dark", height=320,
            margin=dict(l=20,r=20,t=30,b=0),
        )
        st.plotly_chart(fig3, use_container_width=True)

        # ── Area forecast selector ────────────────────────────────────────────
        st.subheader("📍 Area Forecast")
        if forecasts_df is not None and ts_df is not None:
            sel_area = st.selectbox("Select Area",
                                    sorted(ts_df["area"].unique().astype(int)))
            area_hist = ts_df[ts_df["area"] == sel_area]
            area_fc   = forecasts_df[forecasts_df["area"] == sel_area]

            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(
                x=area_hist["ds"], y=area_hist["y"],
                mode="lines+markers", name="Historical",
                line=dict(color="#5C9EFF", width=2),
            ))
            if len(area_fc) > 0:
                future = area_fc[area_fc["ds"] > area_hist["ds"].max()]
                fig4.add_trace(go.Scatter(
                    x=future["ds"], y=future["yhat"],
                    mode="lines", name="Forecast",
                    line=dict(color="gold", width=2, dash="dash"),
                ))
            fig4.update_layout(template="plotly_dark", height=280,
                               margin=dict(l=0,r=0,t=20,b=0),
                               title=f"Area {sel_area} Crime Forecast")
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("Run `python train_all.py <csv>` to generate forecasts.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — VICTIM PROFILE
# ══════════════════════════════════════════════════════════════════════════════
def render_victim_tab(df):
    st.header("👤 Victim Profile Analysis")

    if df is None:
        st.error("Data not loaded.")
        return

    col1, col2 = st.columns(2)

    with col1:
        # Age distribution (known ages)
        if "vict_age" in df.columns and "age_known" in df.columns:
            st.subheader("📊 Victim Age Distribution (Known Only)")
            known = df[df["age_known"] == 1]["vict_age"].dropna()
            fig = px.histogram(
                known, nbins=40,
                color_discrete_sequence=["#e84545"],
                template="plotly_dark",
                labels={"value": "Age", "count": "Count"},
            )
            fig.add_vline(x=known.median(), line_dash="dash", line_color="gold",
                          annotation_text=f"Median: {known.median():.0f}")
            fig.update_layout(height=300, margin=dict(l=0,r=0,t=30,b=0),
                              showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Age group × crime category
        if "age_group" in df.columns and "crime_category" in df.columns:
            st.subheader("🎯 Age Group × Crime Category")
            age_cat = df[df["age_group"] != "Unknown"].groupby(
                ["age_group", "crime_category"]
            ).size().unstack(fill_value=0)
            age_cat_pct = age_cat.div(age_cat.sum(axis=1), axis=0) * 100
            fig2 = px.imshow(
                age_cat_pct, color_continuous_scale="YlOrRd",
                aspect="auto", template="plotly_dark",
                labels={"color": "% of age group"},
                text_auto=".1f",
            )
            fig2.update_layout(height=300, margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig2, use_container_width=True)

    with col2:
        # Sex × crime category
        if "vict_sex" in df.columns and "crime_category" in df.columns:
            st.subheader("♀♂ Victim Sex by Crime Type")
            sc = df[df["vict_sex"].isin(["M","F"])].groupby(
                ["crime_category","vict_sex"]
            ).size().reset_index(name="count")
            fig3 = px.bar(
                sc, x="crime_category", y="count", color="vict_sex",
                barmode="group", template="plotly_dark",
                color_discrete_map={"M": "#5C9EFF", "F": "#E84545"},
                labels={"vict_sex": "Sex", "crime_category": "Crime Type",
                        "count": "Count"},
            )
            fig3.update_layout(height=300, margin=dict(l=0,r=0,t=30,b=0),
                               xaxis_tickangle=-30)
            st.plotly_chart(fig3, use_container_width=True)

        # Hourly pattern by sex
        if "vict_sex" in df.columns:
            st.subheader("⏰ Crime Time Pattern by Sex")
            hourly_sex = df[df["vict_sex"].isin(["M","F"])].groupby(
                ["hour","vict_sex"]
            ).size().reset_index(name="count")
            # Normalise
            total_by_sex = hourly_sex.groupby("vict_sex")["count"].transform("sum")
            hourly_sex["proportion"] = hourly_sex["count"] / total_by_sex
            fig4 = px.line(
                hourly_sex, x="hour", y="proportion", color="vict_sex",
                template="plotly_dark",
                color_discrete_map={"M": "#5C9EFF", "F": "#E84545"},
                labels={"proportion": "Proportion", "hour": "Hour of Day",
                        "vict_sex": "Sex"},
            )
            fig4.update_layout(height=300, margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig4, use_container_width=True)

    # ── Descent breakdown ─────────────────────────────────────────────────────
    if "descent_label" in df.columns and "crime_category" in df.columns:
        st.subheader("🌍 Victim Descent × Crime Category (Heatmap)")
        top5d = df["descent_label"].value_counts().head(6).index
        dc = df[df["descent_label"].isin(top5d)].groupby(
            ["descent_label","crime_category"]
        ).size().unstack(fill_value=0)
        dc_pct = dc.div(dc.sum(axis=1), axis=0) * 100
        fig5 = px.imshow(
            dc_pct, color_continuous_scale="YlOrRd",
            aspect="auto", template="plotly_dark",
            labels={"color": "% of descent group"},
            text_auto=".1f",
        )
        fig5.update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — MODEL EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
def render_explainability_tab(df, models):
    st.header("🔍 Model Explainability (SHAP & Metrics)")
    st.caption("Feature importance, SHAP values, confusion matrix, and area clusters")

    # ── All plots the notebooks generate — correct filenames ─────────────────
    # Key  = display title
    # Value = list of candidate paths (first one found wins)
    plot_registry = {
        "Feature Importance — Hotspot Model": [
            OUTPUT_DIR / "feature_importance_hotspot.png",
        ],
        "SHAP Summary — Hotspot Model": [
            OUTPUT_DIR / "shap_summary_hotspot.png",
            OUTPUT_DIR / "shap_summary.png",          # legacy name fallback
        ],
        "SHAP Beeswarm — Hotspot Model": [
            OUTPUT_DIR / "shap_beeswarm_hotspot.png",
            OUTPUT_DIR / "shap_beeswarm.png",
        ],
        "SHAP — Crime Type Model": [
            OUTPUT_DIR / "shap_crime_type.png",
        ],
        "Confusion Matrix — Test Set (2024)": [
            OUTPUT_DIR / "confusion_matrix_test.png",
            OUTPUT_DIR / "confusion_matrix.png",
        ],
        "Confusion Matrix — OOF (Train CV)": [
            OUTPUT_DIR / "confusion_matrix_oof.png",
        ],
        "OOF Diagnostics — Hotspot": [
            OUTPUT_DIR / "oof_diagnostics_hotspot.png",
        ],
        "Test Evaluation — Hotspot": [
            OUTPUT_DIR / "test_evaluation_hotspot.png",
        ],
        "Area Clusters (PCA)": [
            OUTPUT_DIR / "area_clusters_pca.png",
        ],
        "CV Metrics — Hotspot": [
            OUTPUT_DIR / "cv_metrics_hotspot.png",
        ],
        "CV Metrics — Crime Type": [
            OUTPUT_DIR / "cv_metrics_crime_type.png",
        ],
        "Per-Class F1 Score": [
            OUTPUT_DIR / "per_class_f1.png",
        ],
    }

    # ── Status banner ─────────────────────────────────────────────────────────
    found   = [(t, paths) for t, paths in plot_registry.items()
               if any(p.exists() for p in paths)]
    missing = [t for t, paths in plot_registry.items()
               if not any(p.exists() for p in paths)]

    col_s, col_m = st.columns(2)
    col_s.metric("✅ Plots Available", len(found))
    col_m.metric("⚠️ Not Yet Generated", len(missing))

    if missing:
        with st.expander("Missing plots — run notebooks to generate"):
            for t in missing:
                st.write(f"• {t}")

    st.markdown("---")

    # ── Live confusion matrix generator (if PNG missing but model loaded) ─────
    cm_paths = plot_registry["Confusion Matrix — Test Set (2024)"]
    cm_exists = any(p.exists() for p in cm_paths)

    if not cm_exists and "crime_type" in models and df is not None:
        st.info("🔄 Confusion matrix PNG not found — generating live from loaded model...")
        try:
            import joblib, numpy as np
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.preprocessing import LabelEncoder
            from sklearn.metrics import confusion_matrix as sk_cm

            le           = models["le"]
            ct_feats     = models["ct_features"]
            crime_model  = models["crime_type"]

            from sklearn.preprocessing import LabelEncoder as _LE
            df_plot = df.copy()
            if "time_of_day" in df_plot.columns:
                df_plot["time_of_day_enc"] = _LE().fit_transform(
                    df_plot["time_of_day"].astype(str))
            else:
                df_plot["time_of_day_enc"] = 0
            if "premis_cd" in df_plot.columns:
                df_plot["premis_cd"] = df_plot["premis_cd"].fillna(-1).astype(int)

            df_test_live = df_plot[df_plot["year"] == 2024].copy()
            if len(df_test_live) > 0:
                X_live = df_test_live[[c for c in ct_feats if c in df_test_live.columns]].fillna(0).astype(float)
                missing_ct = [c for c in ct_feats if c not in df_test_live.columns]
                for c in missing_ct:
                    X_live[c] = 0.0
                X_live = X_live[ct_feats]

                y_live  = le.transform(df_test_live["crime_category"].fillna("Other"))
                preds_l = crime_model.predict(X_live)

                cm   = sk_cm(y_live, preds_l)
                cm_p = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

                plt.style.use("dark_background")
                fig, axes = plt.subplots(1, 2, figsize=(18, 7))
                sns.heatmap(cm,   annot=True, fmt="d",   cmap="YlOrRd",
                            xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[0])
                axes[0].set_title("Confusion Matrix — Counts (2024 Test)", fontsize=13, fontweight="bold")
                axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
                plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=40, ha="right")

                sns.heatmap(cm_p, annot=True, fmt=".1f", cmap="YlOrRd",
                            xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[1])
                axes[1].set_title("Confusion Matrix — % of True (2024 Test)", fontsize=13, fontweight="bold")
                axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")
                plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=40, ha="right")

                plt.tight_layout()
                save_path = OUTPUT_DIR / "confusion_matrix_test.png"
                fig.savefig(save_path, dpi=150)
                plt.close(fig)
                st.success(f"✅ Generated and saved to {save_path.name}")
                st.image(str(save_path), use_container_width=True)
            else:
                st.warning("No 2024 test data found in clean_crime.parquet.")
        except Exception as e:
            st.error(f"Live generation failed: {e}")
        st.markdown("---")

    # ── Render all found plots in 2-col grid ─────────────────────────────────
    items = [(t, next(p for p in paths if p.exists()))
             for t, paths in plot_registry.items()
             if any(p.exists() for p in paths)]

    # Wide plots get their own row
    WIDE_PLOTS = {
        "Confusion Matrix — Test Set (2024)",
        "Confusion Matrix — OOF (Train CV)",
        "Test Evaluation — Hotspot",
    }

    wide  = [(t, p) for t, p in items if t in WIDE_PLOTS]
    narrow= [(t, p) for t, p in items if t not in WIDE_PLOTS]

    # Narrow plots — 2 columns
    cols = st.columns(2)
    for i, (title, path) in enumerate(narrow):
        with cols[i % 2]:
            st.subheader(title)
            st.image(str(path), use_container_width=True)

    # Wide plots — full width
    for title, path in wide:
        st.markdown("---")
        st.subheader(title)
        st.image(str(path), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
def main():
    granularity, target_year, target_period, predict_clicked, area_filter = render_sidebar()

    # Load everything
    df         = load_clean_data()
    models     = load_models(granularity)
    centroids  = load_centroids()
    forecasts  = load_forecasts()
    ts_df      = load_timeseries()
    agg_df     = load_agg(granularity)

    # Top-level title
    st.title("🚔 LAPD Crime Intelligence Dashboard")
    st.caption("Machine Learning–powered crime analysis, prediction & forecasting")

    if df is None:
        st.error("⚠️ No cleaned data found. Run: `python train_all.py <path_to_csv>`")
        st.code("python train_all.py data/raw_crime.csv --granularity month")
        return

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗺️ Heatmap",
        "📊 Crime Insights",
        "📈 Trends",
        "👤 Victim Profile",
        "🔍 Explainability",
    ])

    with tab1:
        render_heatmap_tab(df, models, centroids, agg_df,
                           granularity, target_year, target_period,
                           predict_clicked, area_filter)
    with tab2:
        render_insights_tab(df, models)
    with tab3:
        render_trends_tab(df, ts_df, forecasts)
    with tab4:
        render_victim_tab(df)
    with tab5:
        render_explainability_tab(df, models)


if __name__ == "__main__":
    main()