"""Shared Plotly theme for the C2C Fashion Marketplace portfolio notebooks."""
import plotly.io as pio
import plotly.graph_objects as go

BRAND_INDIGO = "#2D31FA"
BRAND_ORANGE = "#FF4D00"
INK = "#0A0A0A"
MUTED = "#F5F5F5"

COLORWAY = [BRAND_INDIGO, BRAND_ORANGE, "#00CC96", "#AB63FA",
            "#FFA15A", "#19D3F3", "#FF6692", "#B6E880"]

PORTFOLIO_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Inter, system-ui, sans-serif", size=13, color=INK),
        title_font_size=18,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        colorway=COLORWAY,
        xaxis=dict(gridcolor="#f0f0f0", zerolinecolor="#e0e0e0"),
        yaxis=dict(gridcolor="#f0f0f0", zerolinecolor="#e0e0e0"),
        hoverlabel=dict(font_size=12, bgcolor="white"),
        margin=dict(l=60, r=40, t=60, b=60),
    )
)

pio.templates["portfolio"] = PORTFOLIO_TEMPLATE
pio.templates.default = "plotly_white+portfolio"
