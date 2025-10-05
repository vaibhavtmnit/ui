# dashboard_ui.py
import os
import pandas as pd
from urllib.parse import parse_qs, urlparse

from dash import Dash, dcc, html, dash_table, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

DATA_ROOT = os.path.abspath("./data")

def get_ns_from_href(href: str, default: str = "default") -> str:
    if not href:
        return default
    q = parse_qs(urlparse(href).query or "")
    ns = q.get("ns", [default])[0].strip()
    # basic hardening: allow alnum, dash, underscore only
    return "".join(ch for ch in ns if ch.isalnum() or ch in "-_") or default

def ns_path(ns: str, filename: str) -> str:
    return os.path.join(DATA_ROOT, ns, filename)

def file_mtime(path: str) -> float:
    return os.path.getmtime(path) if os.path.exists(path) else 0.0

def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame()

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

app = Dash(__name__, external_stylesheets=[dbc.themes.LUX], requests_pathname_prefix="/dash/")
app.title = "Merchant Management Dashboard"

HIGHLIGHT_COLUMNS = ["issue", "priority"]
highlight_styles = [
    {"if": {"column_id": c}, "backgroundColor": "#ffecec", "color": "#7f1d1d",
     "fontWeight": "600", "borderLeft": "4px solid #7f1d1d"} for c in HIGHLIGHT_COLUMNS
]

def kpi_card_black(title: str, value: int):
    return dbc.Card(
        dbc.CardBody([html.Div(title, className="small", style={"color":"#000"}),
                      html.H2(f"{value:,}", className="mb-0", style={"color":"#000"})]),
        className="shadow-sm h-100",
        style={"border":"1px solid #e8e8e8","background":"#fff"}
    )

def category_totals_cards(df: pd.DataFrame):
    cols=[]
    for _, row in df.iterrows():
        cols.append(
            dbc.Col(
                dbc.Card(dbc.CardBody([
                    html.Div(str(row["category"]), className="fw-semibold"),
                    html.Div(f"{int(row['issues']):,} issues", className="text-muted"),
                ]), className="shadow-sm h-100",
                style={"borderLeft":"4px solid #212529","background":"#fdfdfd"}),
                xs=12, sm=6, md=4, lg=3, xl=2, className="mb-3"
            )
        )
    return dbc.Row(cols, className="g-3")

def make_trend_figure(mode: str, df1: pd.DataFrame, df2: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if mode == "total":
        fig.add_trace(go.Scatter(x=df1["date"], y=df1["total"], mode="lines+markers",
                                 name="Total", line=dict(width=3, shape="spline"),
                                 marker=dict(symbol="circle-open", size=8)))
        fig.add_trace(go.Scatter(x=df1["date"], y=df1["issues"], mode="lines+markers",
                                 name="Issues", line=dict(width=3, shape="spline"),
                                 marker=dict(symbol="diamond-open", size=8)))
    else:
        for cat, g in df2.groupby("category"):
            fig.add_trace(go.Scatter(x=g["date"], y=g["issues"], mode="lines+markers",
                                     name=str(cat), line=dict(width=3, shape="spline"),
                                     marker=dict(size=6)))
    fig.update_layout(margin=dict(l=20,r=20,t=10,b=10), xaxis_title="Date", yaxis_title="Count",
                      hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      plot_bgcolor="white", paper_bgcolor="white")
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#f0f0f0")
    return fig

app.layout = dbc.Container([
    dcc.Location(id="url"),
    dcc.Store(id="store-mtimes"),  # per-namespace mtimes
    dcc.Store(id="store-df1"), dcc.Store(id="store-df2"),
    dcc.Store(id="store-df3"), dcc.Store(id="store-df4-cols"),
    dcc.Interval(id="poll", interval=10_000, n_intervals=0),

    dbc.Row(id="kpi-row", className="g-3 mt-3"),

    html.Div(className="mt-3"),
    dbc.Card(dbc.CardBody([
        html.H6("Issues by Category (15 days)", className="mb-3"),
        html.Div(id="category-cards")
    ]), className="shadow-sm"),

    html.Div(className="mt-3"),
    dbc.Card(dbc.CardBody([
        html.H6("Trend (15 days)", className="mb-3"),
        dbc.RadioItems(id="trend-mode",
            options=[{"label":"Total","value":"total"},{"label":"By Category","value":"category"}],
            value="total", inline=True, className="mb-2"),
        dcc.Graph(id="trend-fig", config={"displayModeBar": False}, style={"height":"520px"})
    ]), className="shadow-sm"),

    html.Div(className="mt-3"),
    dbc.Card(dbc.CardBody([
        dbc.Row([
            dbc.Col(html.H6("Detailed Records"), className="mb-2", width=8),
            dbc.Col(html.Div(id="download-slot", className="text-end"), width=4),
        ]),
        dash_table.DataTable(
            id="details-table",
            data=[], columns=[],
            page_current=0, page_size=12, page_action="custom",
            sort_action="none", filter_action="none",
            style_table={"overflowX":"auto"},
            style_header={"backgroundColor":"#f8f9fa","fontWeight":"700","border":"0"},
            style_cell={"padding":"14px","border":"0","whiteSpace":"normal","height":"auto",
                        "fontSize":"15px","lineHeight":"1.6"},
            style_data_conditional=highlight_styles,
        )
    ]), className="shadow-sm"),

    html.Footer("Built with Dash + Bootstrap (LUX theme).", className="text-center text-muted my-4")
], fluid=True)

# --- Poll per-namespace files and load into stores ---
@app.callback(
    Output("store-mtimes","data"),
    Output("store-df1","data"),
    Output("store-df2","data"),
    Output("store-df3","data"),
    Output("store-df4-cols","data"),
    Output("download-slot","children"),
    Input("poll","n_intervals"),
    State("store-mtimes","data"),
    State("url","href"),
    prevent_initial_call=False
)
def poll_and_load(_, prev, href):
    ns = get_ns_from_href(href)
    df1_fp, df2_fp, df3_fp, df4_fp = (ns_path(ns,"df1.parquet"),
                                      ns_path(ns,"df2.parquet"),
                                      ns_path(ns,"df3.parquet"),
                                      ns_path(ns,"df4.csv"))
    cur = { "df1": file_mtime(df1_fp), "df2": file_mtime(df2_fp),
            "df3": file_mtime(df3_fp), "df4": file_mtime(df4_fp) }

    # Download button for this namespace
    download_btn = dbc.Button("Download Details (CSV)", color="primary",
                              href=f"/data/{ns}/df4.csv", external_link=True)

    if prev == cur and prev is not None:
        return cur, no_update, no_update, no_update, no_update, download_btn

    df1 = load_parquet(df1_fp)
    df2 = load_parquet(df2_fp)
    df3 = load_parquet(df3_fp)

    # columns for df4 (from header)
    cols = []
    if os.path.exists(df4_fp):
        with open(df4_fp, "r", encoding="utf-8") as f:
            first = f.readline().rstrip("\n")
            cols = first.split(",") if first else []

    return cur, df1.to_dict("records"), df2.to_dict("records"), df3.to_dict("records"), cols, download_btn

@app.callback(
    Output("kpi-row","children"),
    Input("store-df1","data")
)
def render_kpis(df1_data):
    if not df1_data:
        return []
    df1 = pd.DataFrame(df1_data)
    total_number_15d = int(df1["total"].sum())
    total_issues_15d = int(df1["issues"].sum())
    return [dbc.Col(kpi_card_black("Total Number (15 days)", total_number_15d), md=6, xs=12),
            dbc.Col(kpi_card_black("Total Issues (15 days)", total_issues_15d), md=6, xs=12)]

@app.callback(
    Output("category-cards","children"),
    Input("store-df2","data")
)
def render_category_cards(df2_data):
    if not df2_data:
        return html.Div("Waiting for data…", className="text-muted")
    df2 = pd.DataFrame(df2_data)
    agg = df2.groupby("category", as_index=False)["issues"].sum().sort_values("issues", ascending=False)
    return category_totals_cards(agg)

@app.callback(
    Output("trend-fig","figure"),
    Input("trend-mode","value"),
    Input("store-df1","data"),
    Input("store-df2","data"),
)
def update_trend(mode, df1_data, df2_data):
    df1 = pd.DataFrame(df1_data or [])
    df2 = pd.DataFrame(df2_data or [])
    if df1.empty: return go.Figure()
    return make_trend_figure(mode, df1, df2)

@app.callback(
    Output("details-table","columns"),
    Input("store-df4-cols","data")
)
def set_columns(cols):
    if not cols: return []
    return [{"name": c.replace("_"," ").title(), "id": c} for c in cols]

@app.callback(
    Output("details-table", "data"),
    Input("details-table", "page_current"),
    Input("store-df4-cols", "data"),  # NEW: triggers initial load
    State("details-table", "page_size"),
    State("url", "href")
)
def load_df4_page(page_current, cols, page_size, href):
    if not cols or not href:
        return []
    ns = get_ns_from_href(href)
    df = load_csv(ns_path(ns, "df4.csv"))
    if df.empty:
        return []

    # paginate
    page_current = page_current or 0
    page_size = page_size or 12
    start = page_current * page_size
    stop = start + page_size
    sl = df.iloc[start:stop].copy().where(pd.notna(df.iloc[start:stop]), "")

    # normalize rows to the declared columns (fill blanks for missing)
    records = sl.to_dict("records")
    rows = [{c: rec.get(c, "") for c in cols} for rec in records]
    return rows

server = app.server

if __name__ == "__main__":
    app.run(debug=True)
