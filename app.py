import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
import numpy as np
from streamlit_folium import st_folium

# ── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hoorn — Insulation Priority Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #0e0e0e; color: #f2f5fa; }
  [data-testid="stSidebar"] { background-color: #1a1d23; }
  .kpi-card {
    background: #1a1d23;
    border: 1px solid #283442;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin: 4px;
  }
  .kpi-number {
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 6px;
  }
  .kpi-label {
    font-size: 0.75rem;
    color: #9aa3b2;
    letter-spacing: 1px;
    text-transform: uppercase;
  }
  .kpi-high   { color: #E94E1B; }
  .kpi-medium { color: #F7A823; }
  .kpi-low    { color: #52AE32; }
  .kpi-total  { color: #5b8fff; }
  .kpi-gas    { color: #fde0a6; }
  .section-title {
    font-size: 0.7rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #9aa3b2;
    margin-bottom: 12px;
    margin-top: 24px;
  }
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/hoorn_all_scored_v2.csv", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    for col in ["warmte_final", "warmte_predicted", "gas_sjv_m3",
                "gas_per_m2", "Bouwjaar"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    PC4_COORDS = {
        "1621": (52.6450, 5.0550), "1622": (52.6380, 5.0620),
        "1623": (52.6310, 5.0700), "1624": (52.6250, 5.0580),
        "1625": (52.6200, 5.0650), "1626": (52.6150, 5.0720),
        "1627": (52.6350, 5.0400), "1628": (52.6480, 5.0750),
        "1689": (52.6600, 5.0700),
    }
    pc4_col = "PC4" if "PC4" in df.columns else df.columns[0]
    df["lat"] = df[pc4_col].map(lambda x: PC4_COORDS.get(str(x)[:4], (None,None))[0])
    df["lon"] = df[pc4_col].map(lambda x: PC4_COORDS.get(str(x)[:4], (None,None))[1])
    return df

df = load_data()

# ── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏠 Hoorn Dashboard")
    st.markdown("*Insulation Priority Tool*")
    st.divider()
    st.markdown("### Filters")

    priority_opts = ["High", "Medium", "Low"]
    selected_priority = st.multiselect("Priority", priority_opts, default=priority_opts)

    if "Bouwjaar" in df.columns:
        year_min = int(df["Bouwjaar"].min() or 1900)
        year_max = int(df["Bouwjaar"].max() or 2024)
        year_range = st.slider("Building year", year_min, year_max, (year_min, year_max))
    else:
        year_range = (1900, 2024)

    if "gas_sjv_m3" in df.columns:
        gas_max = int(df["gas_sjv_m3"].max() or 5000)
        gas_range = st.slider("Gas consumption (m³/yr)", 0, gas_max, (0, gas_max))
    else:
        gas_range = (0, 99999)

    if "Energieklasse" in df.columns:
        all_labels = sorted(df["Energieklasse"].dropna().unique().tolist())
        selected_labels = st.multiselect("Energy label", all_labels, default=all_labels)
    else:
        selected_labels = []

    st.divider()
    st.markdown(
        "<small style='color:#9aa3b2'>Data: EP Online · Liander · "
        "Municipality of Hoorn<br>Model: LightGBM Regression "
        "(R²=0.953)</small>",
        unsafe_allow_html=True
    )

# ── APPLY FILTERS ──────────────────────────────────────────────────────────
filtered = df[df["priority"].isin(selected_priority)].copy()

if "Bouwjaar" in filtered.columns:
    filtered = filtered[
        filtered["Bouwjaar"].between(year_range[0], year_range[1])
        | filtered["Bouwjaar"].isna()
    ]
if "gas_sjv_m3" in filtered.columns:
    filtered = filtered[
        filtered["gas_sjv_m3"].between(gas_range[0], gas_range[1])
        | filtered["gas_sjv_m3"].isna()
    ]
if selected_labels and "Energieklasse" in filtered.columns:
    filtered = filtered[
        filtered["Energieklasse"].isin(selected_labels)
        | filtered["Energieklasse"].isna()
    ]

# ── HEADER ─────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='font-size:2rem; font-weight:800; margin-bottom:4px;'>"
    "🏠 Hoorn — Insulation Priority Dashboard</h1>"
    "<p style='color:#9aa3b2; margin-top:0;'>Municipality of Hoorn · "
    "Group 4 · Inholland University of Applied Sciences</p>",
    unsafe_allow_html=True
)
st.divider()

# ── KPI CARDS ──────────────────────────────────────────────────────────────
total    = len(filtered)
n_high   = (filtered["priority"] == "High").sum()
n_medium = (filtered["priority"] == "Medium").sum()
n_low    = (filtered["priority"] == "Low").sum()
avg_gas  = df["gas_sjv_m3"].dropna().mean()

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-number kpi-total">{total:,}</div>
        <div class="kpi-label">Total Homes</div>
    </div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-number kpi-high">{n_high:,}</div>
        <div class="kpi-label">High Priority</div>
    </div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-number kpi-medium">{n_medium:,}</div>
        <div class="kpi-label">Medium Priority</div>
    </div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-number kpi-low">{n_low:,}</div>
        <div class="kpi-label">Low Priority</div>
    </div>""", unsafe_allow_html=True)
with k5:
    gas_display = f"{avg_gas:,.0f}" if pd.notna(avg_gas) else "1,157"
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-number kpi-gas">{gas_display}</div>
        <div class="kpi-label">Avg Gas m³/yr</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── ROW 1: MAP + PRIORITY CHART ────────────────────────────────────────────
col_map, col_chart = st.columns([3, 2])

with col_map:
    st.markdown('<div class="section-title">🗺 Neighbourhood Heatmap</div>',
                unsafe_allow_html=True)
    m = folium.Map(location=[52.6403, 5.0598], zoom_start=13,
                   tiles="CartoDB dark_matter")
    COLOR_MAP = {"High": "#E94E1B", "Medium": "#F7A823", "Low": "#52AE32"}

    plot_df = filtered.dropna(subset=["lat","lon"]).copy()
    if len(plot_df) > 2000:
        plot_df = plot_df.sample(2000, random_state=42)

    for _, row in plot_df.iterrows():
        color = COLOR_MAP.get(str(row.get("priority","")), "#888")
        warmte = row.get("warmte_final")
        warmte_str = f"{warmte:.0f}" if pd.notna(warmte) else "N/A"
        row_lat = row["lat"] + np.random.uniform(-0.015, 0.015)
        row_lon = row["lon"] + np.random.uniform(-0.015, 0.015)
        folium.CircleMarker(
            location=[row_lat, row_lon],
            radius=5, color=color, fill=True,
            fill_color=color, fill_opacity=0.7,
            tooltip=f"Priority: {row.get('priority','?')} | Heat: {warmte_str} kWh/m² | "
                    f"Label: {row.get('Energieklasse','?')} | Year: {row.get('Bouwjaar','?')} | "
                    f"Type: {row.get('Gebouwtype','?')}",
        ).add_to(m)

    legend = """
    <div style="position:fixed;bottom:20px;left:20px;z-index:1000;
    background:#1a1d23;padding:12px 16px;border-radius:10px;
    border:1px solid #283442;font-size:12px;color:#f2f5fa;">
    <b style="color:#f2f5fa">Priority</b><br>
    <span style="color:#E94E1B">●</span> High<br>
    <span style="color:#F7A823">●</span> Medium<br>
    <span style="color:#52AE32">●</span> Low
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    st_folium(m, width=700, height=450, returned_objects=[])

with col_chart:
    # Donut chart
    st.markdown('<div class="section-title">📊 Priority Distribution</div>',
                unsafe_allow_html=True)
    priority_counts = filtered["priority"].value_counts().reindex(
        ["High","Medium","Low"], fill_value=0
    )
    fig_donut = go.Figure(go.Pie(
        labels=priority_counts.index,
        values=priority_counts.values,
        hole=0.6,
        marker_colors=["#E94E1B","#F7A823","#52AE32"],
        textinfo="label+percent",
        textfont_size=13,
    ))
    fig_donut.update_layout(
        paper_bgcolor="#1a1d23", plot_bgcolor="#1a1d23",
        font_color="#f2f5fa", showlegend=False,
        margin=dict(t=10,b=10,l=10,r=10), height=220,
        annotations=[dict(
            text=f"<b>{total:,}</b><br>homes",
            x=0.5, y=0.5, font_size=16,
            font_color="#f2f5fa", showarrow=False
        )]
    )
    st.plotly_chart(fig_donut, use_container_width=True)

    # Building era chart
    st.markdown('<div class="section-title">🏗 Building Era vs Priority</div>',
                unsafe_allow_html=True)
    if "Bouwjaar" in filtered.columns:
        era_df = filtered.copy()
        era_df["era"] = pd.cut(
            era_df["Bouwjaar"],
            bins=[0,1945,1975,1992,2012,2100],
            labels=["<1945","1945–75","1975–92","1992–12",">2012"]
        )
        era_grouped = era_df.groupby(["era","priority"],
                                     observed=True).size().reset_index(name="count")
        fig_bar = px.bar(
            era_grouped, x="era", y="count", color="priority",
            color_discrete_map={"High":"#E94E1B","Medium":"#F7A823","Low":"#52AE32"},
            template="plotly_dark",
            labels={"era":"Building era","count":"Homes","priority":"Priority"},
        )
        fig_bar.update_layout(
            paper_bgcolor="#1a1d23", plot_bgcolor="#1a1d23",
            font_color="#f2f5fa", height=220,
            margin=dict(t=10,b=10,l=10,r=10),
            legend=dict(orientation="h",y=-0.2),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# ── ROW 2: HEAT DEMAND + PC4 ───────────────────────────────────────────────
col_heat, col_pc4 = st.columns(2)

with col_heat:
    st.markdown('<div class="section-title">🌡 Heat Demand Distribution</div>',
                unsafe_allow_html=True)
    if "warmte_final" in filtered.columns:
        fig_hist = px.histogram(
            filtered.dropna(subset=["warmte_final"]),
            x="warmte_final", color="priority", nbins=40,
            color_discrete_map={"High":"#E94E1B","Medium":"#F7A823","Low":"#52AE32"},
            template="plotly_dark",
            labels={"warmte_final":"Warmtebehoefte (kWh/m²)"},
        )
        fig_hist.add_vline(x=100, line_dash="dash", line_color="#9aa3b2",
                           opacity=0.6, annotation_text="100 kWh/m² threshold")
        fig_hist.add_vline(x=150, line_dash="dash", line_color="#E94E1B",
                           opacity=0.6, annotation_text="150 kWh/m² high")
        fig_hist.update_layout(
            paper_bgcolor="#1a1d23", plot_bgcolor="#111111",
            font_color="#f2f5fa", height=280,
            margin=dict(t=20,b=20,l=10,r=10),
            legend=dict(orientation="h",y=-0.2),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

with col_pc4:
    st.markdown('<div class="section-title">📍 Heat Demand per Postcode</div>',
                unsafe_allow_html=True)
    if "warmte_final" in filtered.columns and "PC4" in filtered.columns:
        pc4_df = (
            filtered.groupby("PC4")["warmte_final"]
            .agg(["mean","count"]).reset_index()
            .rename(columns={"mean":"avg","count":"n"})
            .query("n >= 5")
            .sort_values("avg", ascending=True)
            .dropna()
        )
        fig_pc4 = px.bar(
            pc4_df, x="avg", y="PC4", orientation="h",
            color="avg",
            color_continuous_scale=["#52AE32","#F7A823","#E94E1B"],
            template="plotly_dark",
            labels={"avg":"Avg heat demand (kWh/m²)","PC4":"Postcode"},
            text="avg",
        )
        fig_pc4.update_traces(
            texttemplate="%{text:.0f}", textposition="outside", textfont_size=11
        )
        fig_pc4.update_layout(
            paper_bgcolor="#1a1d23", plot_bgcolor="#111111",
            font_color="#f2f5fa", height=280,
            margin=dict(t=10,b=10,l=10,r=50),
            coloraxis_showscale=False,
        )
        fig_pc4.update_xaxes(showgrid=False)
        fig_pc4.update_yaxes(showgrid=False)
        st.plotly_chart(fig_pc4, use_container_width=True)

# ── ROW 3: BUILDING TYPE + ENERGY LABEL ───────────────────────────────────
col_type, col_label = st.columns(2)

with col_type:
    st.markdown('<div class="section-title">🏘 Priority by Building Type</div>',
                unsafe_allow_html=True)
    if "Gebouwtype" in filtered.columns:
        type_df = (
            filtered.groupby(["Gebouwtype","priority"], observed=True)
            .size().reset_index(name="n")
        )
        fig_type = px.bar(
            type_df, x="n", y="Gebouwtype", color="priority",
            orientation="h",
            color_discrete_map={"High":"#E94E1B","Medium":"#F7A823","Low":"#52AE32"},
            template="plotly_dark",
            labels={"n":"Homes","Gebouwtype":"","priority":""},
        )
        fig_type.update_layout(
            paper_bgcolor="#1a1d23", plot_bgcolor="#111111",
            font_color="#f2f5fa", height=280,
            margin=dict(t=10,b=10,l=10,r=10),
            legend=dict(orientation="h",y=-0.2),
        )
        fig_type.update_xaxes(showgrid=False)
        fig_type.update_yaxes(showgrid=False, tickfont_size=10)
        st.plotly_chart(fig_type, use_container_width=True)

with col_label:
    st.markdown('<div class="section-title">⚡ Energy Label Distribution</div>',
                unsafe_allow_html=True)
    if "Energieklasse" in filtered.columns:
        label_order  = ["A++++","A+++","A++","A+","A","B","C","D","E","F","G"]
        label_colors = {
            "A++++":"#005f2e","A+++":"#006e35","A++":"#008040",
            "A+":"#009640","A":"#52AE32","B":"#C8D200",
            "C":"#FFED00","D":"#F7A823","E":"#E94E1B",
            "F":"#d73027","G":"#E30613",
        }
        lc = (
            filtered["Energieklasse"].value_counts()
            .reindex(label_order).dropna().reset_index()
        )
        lc.columns = ["Label","Count"]
        fig_lbl = px.bar(
            lc, x="Label", y="Count", color="Label",
            color_discrete_map=label_colors,
            template="plotly_dark",
            labels={"Label":"","Count":"Homes"},
            text="Count",
        )
        fig_lbl.update_traces(
            texttemplate="%{text:,}", textposition="outside",
            textfont_size=10, showlegend=False
        )
        fig_lbl.update_layout(
            paper_bgcolor="#1a1d23", plot_bgcolor="#111111",
            font_color="#f2f5fa", height=280,
            margin=dict(t=30,b=10,l=10,r=10),
            showlegend=False,
        )
        fig_lbl.update_xaxes(showgrid=False)
        fig_lbl.update_yaxes(gridcolor="#283442")
        st.plotly_chart(fig_lbl, use_container_width=True)

# ── ADDRESS TABLE ──────────────────────────────────────────────────────────
st.divider()
st.markdown('<div class="section-title">📋 Address Scorecard</div>',
            unsafe_allow_html=True)

show_cols = [c for c in [
    "Postcode","Huisnummer","Gebouwtype","Energieklasse",
    "Bouwjaar","warmte_final","priority"
] if c in filtered.columns]

table_df = filtered[show_cols].copy().rename(columns={
    "warmte_final":"Heat demand (kWh/m²)",
    "priority":"Priority",
    "Bouwjaar":"Build year",
    "Energieklasse":"Label",
    "Gebouwtype":"Building type",
})
if "Heat demand (kWh/m²)" in table_df.columns:
    table_df["Heat demand (kWh/m²)"] = table_df["Heat demand (kWh/m²)"].round(1)
    table_df = table_df.sort_values("Heat demand (kWh/m²)", ascending=False)

col_tbl, col_dl = st.columns([5,1])
with col_tbl:
    st.dataframe(table_df.head(500), use_container_width=True, height=300,
                 column_config={
                     "Heat demand (kWh/m²)": st.column_config.NumberColumn(
                         format="%.1f kWh/m²"),
                     "Build year": st.column_config.NumberColumn(format="%d"),
                 })
with col_dl:
    st.markdown("<br><br>", unsafe_allow_html=True)
    csv_data = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download CSV", csv_data,
        "hoorn_priority_list.csv", "text/csv",
        use_container_width=True,
    )
    st.markdown(
        f"<small style='color:#9aa3b2'>{len(filtered):,} addresses<br>"
        f"(table shows top 500)</small>",
        unsafe_allow_html=True
    )

# ── FOOTER ─────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center;color:#374151;font-size:10px;"
    "letter-spacing:2px;padding:16px;'>"
    "HOORN INSULATION PRIORITY DASHBOARD · GROUP 4 · INHOLLAND UNIVERSITY · 2026<br>"
    "Data: EP Online · Liander (CC-BY 4.0) · Municipality of Hoorn"
    "</div>",
    unsafe_allow_html=True
)