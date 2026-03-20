"""
app.py — Nassau Candy Distributor: Shipping Route Efficiency Dashboard
A premium Streamlit analytics dashboard with Font Awesome icons.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os

from data_loader import load_and_clean, FACTORY_COORDS
from analytics import (
    get_kpi_cards,
    get_route_summary,
    get_top_bottom_routes,
    get_regional_summary,
    get_state_summary,
    get_ship_mode_summary,
    get_factory_summary,
    get_delay_stats,
    get_monthly_trend,
    train_delay_model,
    detect_anomalies,
    get_customer_impact,
)

# ═══════════════════════════════════════════════
# Page Config
# ═══════════════════════════════════════════════
st.set_page_config(
    page_title="Nassau Candy · Route Efficiency",
    page_icon="NC",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject Font Awesome + custom CSS
FA_CDN = '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">'
st.markdown(FA_CDN, unsafe_allow_html=True)

css_path = os.path.join(os.path.dirname(__file__), "style.css")
if os.path.exists(css_path):
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ── Helper: Font Awesome icon in markdown ──
def fa(icon_class: str, color: str = "#C4B5FD") -> str:
    """Return an FA icon HTML span for use inside st.markdown."""
    return f'<i class="{icon_class}" style="color:{color};margin-right:6px"></i>'


# ═══════════════════════════════════════════════
# Plotly template
# ═══════════════════════════════════════════════
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#C4B5FD"),
    margin=dict(l=40, r=30, t=50, b=40),
    colorway=[
        "#6C63FF", "#48C6EF", "#A855F7", "#F472B6",
        "#34D399", "#FBBF24", "#FB923C", "#38BDF8",
    ],
)

# ═══════════════════════════════════════════════
# Load Data
# ═══════════════════════════════════════════════
@st.cache_data(show_spinner="Loading dataset…")
def get_data():
    return load_and_clean()

df_raw = get_data()

# ═══════════════════════════════════════════════
# Sidebar Filters
# ═══════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        f'<h2>{fa("fa-solid fa-candy-cane","#A855F7")} Filters</h2>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Date range
    min_date = df_raw["Order Date"].min().date()
    max_date = df_raw["Order Date"].max().date()
    date_range = st.date_input(
        "Order Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    st.markdown("")

    # Region
    all_regions = sorted(df_raw["Region"].unique())
    selected_regions = st.multiselect("Region", all_regions, default=all_regions)

    # State
    available_states = sorted(
        df_raw[df_raw["Region"].isin(selected_regions)]["State/Province"].unique()
    )
    selected_states = st.multiselect("State / Province", available_states, default=available_states)

    # Ship mode
    all_modes = sorted(df_raw["Ship Mode"].unique())
    selected_modes = st.multiselect("Ship Mode", all_modes, default=all_modes)

    st.markdown("")

    # Lead time threshold
    max_lt = int(df_raw["Shipping_Lead_Time"].max())
    lt_threshold = st.slider(
        "Delay Threshold (days)",
        min_value=0,
        max_value=max_lt,
        value=min(180, max_lt),
        help="Shipments exceeding this threshold are marked as delayed.",
    )

    st.markdown("---")
    st.caption("Nassau Candy Distributor © 2026")

# ═══════════════════════════════════════════════
# Apply Filters
# ═══════════════════════════════════════════════
df = df_raw.copy()

if len(date_range) == 2:
    start, end = date_range
    df = df[(df["Order Date"].dt.date >= start) & (df["Order Date"].dt.date <= end)]

df = df[df["Region"].isin(selected_regions)]
df = df[df["State/Province"].isin(selected_states)]
df = df[df["Ship Mode"].isin(selected_modes)]

if df.empty:
    st.warning("No data matches the current filter selection. Please adjust your filters.")
    st.stop()

# ═══════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════
st.markdown(
    f'<h1>{fa("fa-solid fa-candy-cane","#6C63FF")} Nassau Candy · Route Efficiency Dashboard</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    "<span style='color:#94A3B8;font-size:0.95rem'>"
    "Factory-to-Customer Shipping Route Efficiency Analysis"
    "</span>",
    unsafe_allow_html=True,
)

# KPI Row
kpis = get_kpi_cards(df)
cols = st.columns(4)
kpi_labels_1 = ["Total Orders", "Total Shipments", "Avg Lead Time", "Unique Routes"]
for i, key in enumerate(kpi_labels_1):
    cols[i].metric(key, kpis[key])

cols2 = st.columns(4)
kpi_labels_2 = ["Median Lead Time", "Total Sales", "Gross Profit", "Unique Customers"]
for i, key in enumerate(kpi_labels_2):
    cols2[i].metric(key, kpis[key])

st.markdown("---")

# ═══════════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Route Efficiency",
    "Geographic Map",
    "Ship Mode Analysis",
    "Route Drill-Down",
    "Advanced Analytics 🔮",
])

# ═══════════════════════════════════════════════
# TAB 1 — Route Efficiency Overview
# ═══════════════════════════════════════════════
with tab1:
    route_summary = get_route_summary(df)
    top_routes, bottom_routes = get_top_bottom_routes(route_summary, 10)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            f'### {fa("fa-solid fa-trophy","#FBBF24")} Top 10 Most Efficient Routes',
            unsafe_allow_html=True,
        )
        fig_top = px.bar(
            top_routes.sort_values("Efficiency_Score"),
            x="Efficiency_Score",
            y="Route",
            orientation="h",
            color="Efficiency_Score",
            color_continuous_scale=["#34D399", "#6C63FF"],
            text="Efficiency_Score",
        )
        fig_top.update_layout(**PLOTLY_LAYOUT, height=450, showlegend=False, coloraxis_showscale=False)
        fig_top.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_top.update_xaxes(title="Efficiency Score", gridcolor="rgba(108,99,255,0.08)")
        fig_top.update_yaxes(title="", gridcolor="rgba(108,99,255,0.08)")
        st.plotly_chart(fig_top, use_container_width=True)

    with c2:
        st.markdown(
            f'### {fa("fa-solid fa-triangle-exclamation","#F472B6")} Top 10 Least Efficient Routes',
            unsafe_allow_html=True,
        )
        fig_bot = px.bar(
            bottom_routes.sort_values("Efficiency_Score", ascending=False),
            x="Efficiency_Score",
            y="Route",
            orientation="h",
            color="Efficiency_Score",
            color_continuous_scale=["#6C63FF", "#F472B6"],
            text="Efficiency_Score",
        )
        fig_bot.update_layout(**PLOTLY_LAYOUT, height=450, showlegend=False, coloraxis_showscale=False)
        fig_bot.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_bot.update_xaxes(title="Efficiency Score", gridcolor="rgba(108,99,255,0.08)")
        fig_bot.update_yaxes(title="", gridcolor="rgba(108,99,255,0.08)")
        st.plotly_chart(fig_bot, use_container_width=True)

    # Regional performance
    st.markdown(
        f'### {fa("fa-solid fa-earth-americas","#48C6EF")} Average Lead Time by Region',
        unsafe_allow_html=True,
    )
    regional = get_regional_summary(df)
    fig_reg = px.bar(
        regional,
        x="Region",
        y="Avg_Lead_Time",
        color="Region",
        text="Avg_Lead_Time",
        color_discrete_sequence=["#6C63FF", "#48C6EF", "#A855F7", "#F472B6"],
    )
    fig_reg.update_layout(**PLOTLY_LAYOUT, height=380, showlegend=False)
    fig_reg.update_traces(texttemplate="%{text:.1f} days", textposition="outside")
    fig_reg.update_xaxes(title="", gridcolor="rgba(108,99,255,0.08)")
    fig_reg.update_yaxes(title="Avg Lead Time (days)", gridcolor="rgba(108,99,255,0.08)")
    st.plotly_chart(fig_reg, use_container_width=True)

    # Factory performance
    st.markdown(
        f'### {fa("fa-solid fa-industry","#A855F7")} Factory Performance',
        unsafe_allow_html=True,
    )
    factory_summary = get_factory_summary(df)
    fig_fac = px.bar(
        factory_summary,
        x="Factory",
        y="Avg_Lead_Time",
        color="Total_Shipments",
        text="Avg_Lead_Time",
        color_continuous_scale=["#48C6EF", "#6C63FF", "#A855F7"],
    )
    fig_fac.update_layout(**PLOTLY_LAYOUT, height=380, coloraxis_colorbar=dict(title="Shipments"))
    fig_fac.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig_fac.update_xaxes(title="", gridcolor="rgba(108,99,255,0.08)")
    fig_fac.update_yaxes(title="Avg Lead Time (days)", gridcolor="rgba(108,99,255,0.08)")
    st.plotly_chart(fig_fac, use_container_width=True)

    # Monthly trend
    st.markdown(
        f'### {fa("fa-solid fa-chart-line","#34D399")} Lead Time Trend Over Time',
        unsafe_allow_html=True,
    )
    trend = get_monthly_trend(df)
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=trend["Order_Month"],
        y=trend["Avg_Lead_Time"],
        mode="lines+markers",
        line=dict(color="#6C63FF", width=3, shape="spline"),
        marker=dict(size=7, color="#48C6EF", line=dict(color="#6C63FF", width=2)),
        name="Avg Lead Time",
        fill="tozeroy",
        fillcolor="rgba(108,99,255,0.08)",
    ))
    fig_trend.update_layout(**PLOTLY_LAYOUT, height=350, showlegend=False)
    fig_trend.update_xaxes(title="", gridcolor="rgba(108,99,255,0.08)")
    fig_trend.update_yaxes(title="Avg Lead Time (days)", gridcolor="rgba(108,99,255,0.08)")
    st.plotly_chart(fig_trend, use_container_width=True)

    # Route data table
    with st.expander("Full Route Performance Table"):
        st.dataframe(
            route_summary.style.format({
                "Avg_Lead_Time": "{:.1f}",
                "Median_Lead_Time": "{:.1f}",
                "Std_Lead_Time": "{:.1f}",
                "Efficiency_Score": "{:.1f}",
                "Total_Sales": "${:,.0f}",
                "Avg_Sales": "${:,.2f}",
                "Total_Profit": "${:,.0f}",
            }),
            use_container_width=True,
            height=500,
        )


# ═══════════════════════════════════════════════
# TAB 2 — Geographic Shipping Map
# ═══════════════════════════════════════════════
with tab2:
    st.markdown(
        f'### {fa("fa-solid fa-map-location-dot","#48C6EF")} Shipping Efficiency by State',
        unsafe_allow_html=True,
    )
    state_summary = get_state_summary(df)
    state_geo = state_summary.dropna(subset=["Lat", "Lon"])

    if not state_geo.empty:
        fig_map = px.scatter_geo(
            state_geo,
            lat="Lat",
            lon="Lon",
            size="Total_Shipments",
            color="Avg_Lead_Time",
            hover_name="State/Province",
            hover_data={
                "Avg_Lead_Time": ":.1f",
                "Total_Shipments": ":,",
                "Total_Sales": ":$,.0f",
                "Lat": False,
                "Lon": False,
            },
            color_continuous_scale=["#34D399", "#FBBF24", "#F472B6"],
            size_max=35,
            scope="north america",
        )

        # Add factory markers
        factory_df = pd.DataFrame([
            {"Factory": name, "Lat": coords[0], "Lon": coords[1]}
            for name, coords in FACTORY_COORDS.items()
        ])
        fig_map.add_trace(go.Scattergeo(
            lat=factory_df["Lat"],
            lon=factory_df["Lon"],
            text=factory_df["Factory"],
            mode="markers+text",
            marker=dict(size=14, color="#6C63FF", symbol="diamond", line=dict(width=2, color="#fff")),
            textposition="top center",
            textfont=dict(size=10, color="#E0E0FF"),
            name="Factories",
            showlegend=True,
        ))

        fig_map.update_layout(
            **PLOTLY_LAYOUT,
            height=600,
            geo=dict(
                bgcolor="rgba(0,0,0,0)",
                lakecolor="rgba(108,99,255,0.08)",
                landcolor="rgba(26,29,35,1)",
                subunitcolor="rgba(108,99,255,0.2)",
                countrycolor="rgba(108,99,255,0.3)",
                showlakes=True,
                showsubunits=True,
                showcountries=True,
            ),
            coloraxis_colorbar=dict(title="Avg Lead<br>Time (days)"),
        )
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("No geographic data available for the current filter selection.")

    # Regional bottleneck analysis
    st.markdown(
        f'### {fa("fa-solid fa-fire","#FB923C")} Regional Bottleneck Analysis',
        unsafe_allow_html=True,
    )
    regional = get_regional_summary(df)
    c1, c2 = st.columns(2)

    with c1:
        fig_bubble = px.scatter(
            regional,
            x="Total_Shipments",
            y="Avg_Lead_Time",
            size="Total_Sales",
            color="Region",
            text="Region",
            size_max=60,
            color_discrete_sequence=["#6C63FF", "#48C6EF", "#A855F7", "#F472B6"],
        )
        fig_bubble.update_layout(**PLOTLY_LAYOUT, height=400, showlegend=False)
        fig_bubble.update_traces(textposition="top center")
        fig_bubble.update_xaxes(title="Total Shipments", gridcolor="rgba(108,99,255,0.08)")
        fig_bubble.update_yaxes(title="Avg Lead Time (days)", gridcolor="rgba(108,99,255,0.08)")
        st.plotly_chart(fig_bubble, use_container_width=True)

    with c2:
        delay_by_region = df.groupby("Region").agg(
            Total=("Row ID", "count"),
            Delayed=("Shipping_Lead_Time", lambda x: (x > lt_threshold).sum()),
        ).reset_index()
        delay_by_region["Delay_%"] = ((delay_by_region["Delayed"] / delay_by_region["Total"]) * 100).round(1)

        fig_delay = px.bar(
            delay_by_region,
            x="Region",
            y="Delay_%",
            color="Region",
            text="Delay_%",
            color_discrete_sequence=["#6C63FF", "#48C6EF", "#A855F7", "#F472B6"],
        )
        fig_delay.update_layout(**PLOTLY_LAYOUT, height=400, showlegend=False)
        fig_delay.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_delay.update_xaxes(title="", gridcolor="rgba(108,99,255,0.08)")
        fig_delay.update_yaxes(title="Delay Frequency (%)", gridcolor="rgba(108,99,255,0.08)")
        st.plotly_chart(fig_delay, use_container_width=True)

    # State heatmap table
    with st.expander("State-Level Performance Table"):
        display_cols = ["State/Province", "Total_Shipments", "Avg_Lead_Time", "Median_Lead_Time", "Std_Lead_Time", "Total_Sales", "Total_Profit"]
        st.dataframe(
            state_summary[display_cols].style.format({
                "Avg_Lead_Time": "{:.1f}",
                "Median_Lead_Time": "{:.1f}",
                "Std_Lead_Time": "{:.1f}",
                "Total_Sales": "${:,.0f}",
                "Total_Profit": "${:,.0f}",
            }),
            use_container_width=True,
            height=500,
        )


# ═══════════════════════════════════════════════
# TAB 3 — Ship Mode Comparison
# ═══════════════════════════════════════════════
with tab3:
    ship_summary = get_ship_mode_summary(df)

    st.markdown(
        f'### {fa("fa-solid fa-truck-fast","#48C6EF")} Ship Mode Performance Overview',
        unsafe_allow_html=True,
    )

    mode_colors = ["#34D399", "#FBBF24", "#FB923C", "#F472B6"]
    m1, m2, m3, m4 = st.columns(4)
    for i, (_, row) in enumerate(ship_summary.iterrows()):
        col = [m1, m2, m3, m4][i % 4]
        col.metric(
            row["Ship Mode"],
            f"{row['Avg_Lead_Time']:.1f} days avg",
            f"{int(row['Total_Shipments']):,} shipments",
        )

    st.markdown("")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            f'### {fa("fa-solid fa-clock","#FBBF24")} Average Lead Time by Ship Mode',
            unsafe_allow_html=True,
        )
        fig_mode = px.bar(
            ship_summary,
            x="Ship Mode",
            y="Avg_Lead_Time",
            color="Ship Mode",
            text="Avg_Lead_Time",
            color_discrete_sequence=["#34D399", "#48C6EF", "#FBBF24", "#F472B6"],
        )
        fig_mode.update_layout(**PLOTLY_LAYOUT, height=400, showlegend=False)
        fig_mode.update_traces(texttemplate="%{text:.1f} days", textposition="outside")
        fig_mode.update_xaxes(title="", gridcolor="rgba(108,99,255,0.08)")
        fig_mode.update_yaxes(title="Avg Lead Time (days)", gridcolor="rgba(108,99,255,0.08)")
        st.plotly_chart(fig_mode, use_container_width=True)

    with c2:
        st.markdown(
            f'### {fa("fa-solid fa-box","#A855F7")} Lead Time Distribution',
            unsafe_allow_html=True,
        )
        fig_box = px.box(
            df,
            x="Ship Mode",
            y="Shipping_Lead_Time",
            color="Ship Mode",
            color_discrete_sequence=["#34D399", "#48C6EF", "#FBBF24", "#F472B6"],
        )
        fig_box.update_layout(**PLOTLY_LAYOUT, height=400, showlegend=False)
        fig_box.update_xaxes(title="", gridcolor="rgba(108,99,255,0.08)")
        fig_box.update_yaxes(title="Lead Time (days)", gridcolor="rgba(108,99,255,0.08)")
        st.plotly_chart(fig_box, use_container_width=True)

    # Cost-time tradeoff
    st.markdown(
        f'### {fa("fa-solid fa-sack-dollar","#34D399")} Cost-Time Tradeoff Analysis',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)

    with c1:
        fig_scatter = px.scatter(
            ship_summary,
            x="Avg_Lead_Time",
            y="Profit_Margin_%",
            size="Total_Shipments",
            color="Ship Mode",
            text="Ship Mode",
            size_max=50,
            color_discrete_sequence=["#34D399", "#48C6EF", "#FBBF24", "#F472B6"],
        )
        fig_scatter.update_layout(**PLOTLY_LAYOUT, height=400, showlegend=False)
        fig_scatter.update_traces(textposition="top center")
        fig_scatter.update_xaxes(title="Avg Lead Time (days)", gridcolor="rgba(108,99,255,0.08)")
        fig_scatter.update_yaxes(title="Profit Margin (%)", gridcolor="rgba(108,99,255,0.08)")
        st.plotly_chart(fig_scatter, use_container_width=True)

    with c2:
        fig_pie = px.pie(
            ship_summary,
            values="Total_Shipments",
            names="Ship Mode",
            color_discrete_sequence=["#6C63FF", "#48C6EF", "#A855F7", "#F472B6"],
            hole=0.45,
        )
        fig_pie.update_layout(**PLOTLY_LAYOUT, height=400)
        fig_pie.update_traces(
            textinfo="percent+label",
            textfont=dict(size=12),
            marker=dict(line=dict(color="#0E1117", width=2)),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Ship mode data table
    with st.expander("Ship Mode Detailed Statistics"):
        st.dataframe(
            ship_summary.style.format({
                "Avg_Lead_Time": "{:.1f}",
                "Median_Lead_Time": "{:.1f}",
                "Std_Lead_Time": "{:.1f}",
                "Total_Sales": "${:,.0f}",
                "Avg_Sales": "${:,.2f}",
                "Total_Cost": "${:,.0f}",
                "Total_Profit": "${:,.0f}",
                "Profit_Margin_%": "{:.1f}%",
            }),
            use_container_width=True,
        )


# ═══════════════════════════════════════════════
# TAB 4 — Route Drill-Down
# ═══════════════════════════════════════════════
with tab4:
    st.markdown(
        f'### {fa("fa-solid fa-magnifying-glass","#6C63FF")} State-Level Route Drill-Down',
        unsafe_allow_html=True,
    )

    # State selector
    drill_states = sorted(df["State/Province"].unique())
    selected_drill_state = st.selectbox("Select a State / Province", drill_states, index=0)

    state_df = df[df["State/Province"] == selected_drill_state]

    if state_df.empty:
        st.warning("No data for this state.")
    else:
        # State KPIs
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Shipments", f"{len(state_df):,}")
        s2.metric("Avg Lead Time", f"{state_df['Shipping_Lead_Time'].mean():.1f} days")
        s3.metric("Total Sales", f"${state_df['Sales'].sum():,.0f}")
        s4.metric("Gross Profit", f"${state_df['Gross Profit'].sum():,.0f}")

        st.markdown("")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown(
                f'#### {fa("fa-solid fa-industry","#A855F7")} Routes Serving This State',
                unsafe_allow_html=True,
            )
            state_routes = get_route_summary(state_df)
            fig_sr = px.bar(
                state_routes.head(15),
                x="Avg_Lead_Time",
                y="Route",
                orientation="h",
                color="Efficiency_Score",
                color_continuous_scale=["#F472B6", "#FBBF24", "#34D399"],
                text="Avg_Lead_Time",
            )
            fig_sr.update_layout(**PLOTLY_LAYOUT, height=450, coloraxis_colorbar=dict(title="Score"))
            fig_sr.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig_sr.update_xaxes(title="Avg Lead Time (days)", gridcolor="rgba(108,99,255,0.08)")
            fig_sr.update_yaxes(title="", gridcolor="rgba(108,99,255,0.08)")
            st.plotly_chart(fig_sr, use_container_width=True)

        with c2:
            st.markdown(
                f'#### {fa("fa-solid fa-chart-pie","#48C6EF")} Ship Mode Split',
                unsafe_allow_html=True,
            )
            mode_split = state_df.groupby("Ship Mode").size().reset_index(name="Count")
            fig_ms = px.pie(
                mode_split,
                values="Count",
                names="Ship Mode",
                color_discrete_sequence=["#6C63FF", "#48C6EF", "#A855F7", "#F472B6"],
                hole=0.45,
            )
            fig_ms.update_layout(**PLOTLY_LAYOUT, height=450)
            fig_ms.update_traces(
                textinfo="percent+label",
                marker=dict(line=dict(color="#0E1117", width=2)),
            )
            st.plotly_chart(fig_ms, use_container_width=True)

        # Shipment timeline (Gantt-style)
        st.markdown(
            f'#### {fa("fa-solid fa-calendar-days","#FBBF24")} Order-Level Shipment Timeline (Recent 50)',
            unsafe_allow_html=True,
        )
        timeline_df = (
            state_df.sort_values("Order Date", ascending=False)
            .head(50)
            .copy()
        )
        timeline_df["Order_Label"] = (
            timeline_df["Order ID"].astype(str).str[:20] + "…"
        )

        fig_gantt = px.timeline(
            timeline_df,
            x_start="Order Date",
            x_end="Ship Date",
            y="Order_Label",
            color="Ship Mode",
            hover_data=["Product Name", "Shipping_Lead_Time", "Sales"],
            color_discrete_sequence=["#6C63FF", "#48C6EF", "#A855F7", "#F472B6"],
        )
        fig_gantt.update_layout(**PLOTLY_LAYOUT, height=600, showlegend=True)
        fig_gantt.update_yaxes(title="", gridcolor="rgba(108,99,255,0.08)", autorange="reversed")
        fig_gantt.update_xaxes(title="", gridcolor="rgba(108,99,255,0.08)")
        st.plotly_chart(fig_gantt, use_container_width=True)

        # Delay stats
        st.markdown(
            f'#### {fa("fa-solid fa-triangle-exclamation","#F472B6")} Delay Analysis',
            unsafe_allow_html=True,
        )
        delay_stats = get_delay_stats(state_df, lt_threshold)
        delayed_count = (state_df["Shipping_Lead_Time"] > lt_threshold).sum()
        total_count = len(state_df)
        delay_pct = (delayed_count / total_count * 100) if total_count > 0 else 0

        d1, d2, d3 = st.columns(3)
        d1.metric("Total Shipments", f"{total_count:,}")
        d2.metric("Delayed Shipments", f"{delayed_count:,}")
        d3.metric("Delay Rate", f"{delay_pct:.1f}%")

        # Order-level data table
        with st.expander("Order-Level Data"):
            display_df = state_df[[
                "Order ID", "Order Date", "Ship Date", "Shipping_Lead_Time",
                "Ship Mode", "Product Name", "Factory", "City", "Sales", "Gross Profit",
            ]].sort_values("Order Date", ascending=False)
            st.dataframe(
                display_df.style.format({
                    "Order Date": lambda x: x.strftime("%d-%b-%Y") if pd.notna(x) else "",
                    "Ship Date": lambda x: x.strftime("%d-%b-%Y") if pd.notna(x) else "",
                    "Sales": "${:,.2f}",
                    "Gross Profit": "${:,.2f}",
                }),
                use_container_width=True,
                height=500,
            )

# ═══════════════════════════════════════════════
# TAB 5 — Advanced Analytics
# ═══════════════════════════════════════════════
with tab5:
    st.markdown(
        f'### {fa("fa-solid fa-wand-magic-sparkles","#C4B5FD")} Predictive Analytics & Anomalies',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown(
            f'#### {fa("fa-solid fa-bullseye","#FBBF24")} Predictive Delay Modeling (Random Forest)',
            unsafe_allow_html=True,
        )
        st.markdown("Identifies which features (Ship Mode, Route, etc.) are the strongest drivers of shipping delays.")
        
        with st.spinner("Training model on current data context..."):
            model_results = train_delay_model(df, lt_threshold)
            
            if model_results:
                msg = f"Model Accuracy: {model_results['accuracy']*100:.1f}%"
                st.success(msg)
                
                feat_imp = model_results["feature_importance"]
                fig_ml = px.bar(
                    feat_imp,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale=["#34D399", "#6C63FF"],
                )
                fig_ml.update_layout(**PLOTLY_LAYOUT, height=350, showlegend=False, coloraxis_showscale=False)
                fig_ml.update_yaxes(title="")
                st.plotly_chart(fig_ml, use_container_width=True)
            else:
                st.info("Not enough variations in the filtered data to train the model (requires both delayed and non-delayed samples).")

    with c2:
        st.markdown(
            f'#### {fa("fa-solid fa-bolt","#48C6EF")} Route Anomaly Detection',
            unsafe_allow_html=True,
        )
        st.markdown("Highlights specific routes that are severely underperforming the global historical average lead time.")
        
        route_sum = get_route_summary(df)
        anomalies = detect_anomalies(route_sum, df)
        
        if not anomalies.empty:
            st.error(f"Found {len(anomalies)} anomalous routes (>1.5 standard deviations from the global mean).")
            display_anomalies = anomalies[["Route", "Avg_Lead_Time", "Deviation_from_Avg", "Total_Shipments"]]
            st.dataframe(
                display_anomalies.style.format({
                    "Avg_Lead_Time": "{:.1f} days",
                    "Deviation_from_Avg": "+{:.1f} days"
                }),
                use_container_width=True,
                height=350,
            )
        else:
            st.success("No anomalous routes detected for current selection. All routes are operating within normal variance.")

    # Customer Impact Section
    st.markdown("---")
    st.markdown(
        f'#### {fa("fa-solid fa-users-viewfinder","#F472B6")} Customer Impact Deep-Dive',
        unsafe_allow_html=True,
    )
    
    impact_df = get_customer_impact(df, lt_threshold)
    
    if not impact_df.empty:
        total_risk = impact_df['Revenue_At_Risk'].sum()
        affected_customers = len(impact_df)
        
        st.error(f"Total Revenue at Risk due to delays: **${total_risk:,.0f}** across **{affected_customers}** customers.")
        
        st.dataframe(
            impact_df.head(50).style.format({
                "Total_Orders": "{:,.0f}",
                "Delayed_Orders": "{:,.0f}",
                "Revenue_At_Risk": "${:,.0f}",
                "Total_Revenue": "${:,.0f}",
                "Delay_Rate_%": "{:.1f}%"
            }).background_gradient(subset=["Revenue_At_Risk"], cmap="Reds"),
            use_container_width=True,
            height=400,
        )
    else:
        st.info("No customers heavily impacted by delays with current filters.")
