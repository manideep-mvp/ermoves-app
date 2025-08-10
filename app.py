"""Earnings Moves Analyzer (Finnhub + yfinance + Dash)

Save as app.py. Requires FINNHUB_API_KEY in environment or .env file.
"""

import os
import math
import requests
from functools import lru_cache
from datetime import datetime, date, timedelta

import pandas as pd
import yfinance as yf
import plotly.graph_objs as go

import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
from flask_caching import Cache
from dotenv import load_dotenv

load_dotenv()

# ---------- CONFIG ----------
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
if not FINNHUB_API_KEY:
    FINNHUB_API_KEY = "YOUR_API_KEY"  # replace or set .env

FINNHUB_CAL_URL = "https://finnhub.io/api/v1/calendar/earnings"
FINNHUB_EARNINGS_FOR_SYMBOL = "https://finnhub.io/api/v1/stock/earnings"

# caching config
CACHE_CONFIG = {
    "CACHE_TYPE": "SimpleCache",
    "CACHE_DEFAULT_TIMEOUT": 60 * 10,  # 10 minutes
}

# ---------- HELPERS ----------
def iso_date(d: date):
    return d.strftime("%Y-%m-%d")

def get_week_bounds(some_date: date = None):
    today = some_date or date.today()
    start = today - timedelta(days=today.weekday())  # Monday
    end = start + timedelta(days=4)  # Friday
    return start, end

def request_finnhub(url, params=None):
    params = params or {}
    params["token"] = FINNHUB_API_KEY
    resp = requests.get(url, params=params, timeout=10)
    if resp.status_code != 200:
        raise RuntimeError(f"Finnhub request failed ({resp.status_code}): {resp.text}")
    return resp.json()

@lru_cache(maxsize=64)
def fetch_earnings_week(start_iso: str, end_iso: str):
    data = request_finnhub(FINNHUB_CAL_URL, params={"from": start_iso, "to": end_iso})
    results = data.get("earningsCalendar") or data.get("earnings") or data
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    copy_cols = {}
    if "symbol" in df.columns:
        copy_cols["symbol"] = "symbol"
    elif "ticker" in df.columns:
        copy_cols["symbol"] = "ticker"
    if "date" in df.columns:
        copy_cols["date"] = "date"
    if "hour" in df.columns:
        copy_cols["time"] = "hour"
    if "time" in df.columns:
        copy_cols["time"] = "time"
    if copy_cols:
        df = df.rename(columns={v: k for k, v in copy_cols.items()})
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    if "time" in df.columns:
        df["time"] = df["time"].fillna("TBA")
    else:
        df["time"] = "TBA"
    if "symbol" not in df.columns and "ticker" in df.columns:
        df = df.rename(columns={"ticker": "symbol"})
    keep = [c for c in ["symbol", "date", "time", "company"] if c in df.columns]
    return df[keep].drop_duplicates().reset_index(drop=True)

@lru_cache(maxsize=128)
def fetch_last_price(ticker: str):
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="5d", actions=False)
        if hist.empty:
            return None
        return float(hist["Close"].dropna().iloc[-1])
    except Exception:
        return None

@lru_cache(maxsize=128)
def fetch_historical_prices(ticker: str, period: str = "5y"):
    t = yf.Ticker(ticker)
    df = t.history(period=period, actions=False).copy()
    if df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = None
    return df[["Date", "Open", "High", "Low", "Close", "Volume"]]

@lru_cache(maxsize=128)
def fetch_finnhub_earnings_for_symbol(ticker: str):
    params = {"symbol": ticker}
    try:
        payload = request_finnhub(FINNHUB_EARNINGS_FOR_SYMBOL, params=params)
        if not payload:
            return []
        return payload if isinstance(payload, list) else payload.get("earnings", []) or []
    except Exception:
        return []

def compute_earnings_moves_from_hist(hist_df: pd.DataFrame, earnings_events: list):
    if hist_df.empty or not earnings_events:
        return pd.DataFrame()
    hist = hist_df.set_index("Date")
    results = []
    for ev in earnings_events:
        ev_date = None
        if "date" in ev and ev["date"]:
            try:
                ev_date = pd.to_datetime(ev["date"]).date()
            except Exception:
                pass
        elif "period" in ev and ev["period"]:
            try:
                ev_date = pd.to_datetime(ev["period"]).date()
            except Exception:
                pass
        elif "datetime" in ev and ev["datetime"]:
            try:
                ev_date = pd.to_datetime(ev["datetime"]).date()
            except Exception:
                pass
        if ev_date is None:
            continue
        ev_time = ev.get("time") or ev.get("hour") or ev.get("epsReportTime") or "TBA"
        ev_time = ev_time.upper() if isinstance(ev_time, str) else "TBA"
        try:
            if ev_time.startswith("B"):
                pre_date = prev_trading_date(ev_date, hist)
                post_date = ev_date
            else:
                pre_date = ev_date
                post_date = next_trading_date(ev_date, hist)
        except Exception:
            continue
        if pre_date not in hist.index or post_date not in hist.index:
            continue
        pre_close = hist.at[pre_date, "Close"]
        post_open = hist.at[post_date, "Open"]
        post_close = hist.at[post_date, "Close"]
        post_high = hist.at[post_date, "High"]
        post_low = hist.at[post_date, "Low"]
        pre_vol = hist.at[pre_date, "Volume"] or 0
        post_vol = hist.at[post_date, "Volume"] or 0
        pct_open = safe_percent((post_open - pre_close), pre_close)
        pct_close = safe_percent((post_close - pre_close), pre_close)
        pct_intraday = safe_percent((post_close - post_open), pre_close)
        pct_high = safe_percent((post_high - pre_close), pre_close)
        pct_low = safe_percent((post_low - pre_close), pre_close)
        avg_vol_window = avg_volume_before(pre_date, hist, days=5)
        vol_spike = False
        if avg_vol_window is not None and avg_vol_window > 0 and post_vol > 2 * avg_vol_window:
            vol_spike = True
        elif pre_vol and post_vol > 2 * pre_vol:
            vol_spike = True
        results.append({
            "earnings_date": ev_date.isoformat(),
            "report_time": ev_time,
            "pre_date": pre_date.isoformat(),
            "post_date": post_date.isoformat(),
            "pre_close": float_or_none(pre_close),
            "post_open": float_or_none(post_open),
            "post_close": float_or_none(post_close),
            "%move_at_open": pct_open,
            "%move_at_close": pct_close,
            "%intraday_move": pct_intraday,
            "%highest_move": pct_high,
            "%lowest_move": pct_low,
            "pre_volume": int_or_none(pre_vol),
            "post_volume": int_or_none(post_vol),
            "volume_spike": vol_spike
        })
    return pd.DataFrame(results)

def prev_trading_date(d: date, hist_indexed: pd.DataFrame):
    idxs = [dt for dt in hist_indexed.index if dt < d]
    if not idxs:
        raise KeyError("no previous trading date")
    return max(idxs)

def next_trading_date(d: date, hist_indexed: pd.DataFrame):
    idxs = [dt for dt in hist_indexed.index if dt > d]
    if not idxs:
        raise KeyError("no next trading date")
    return min(idxs)

def safe_percent(numerator, denominator):
    try:
        if denominator == 0 or denominator is None:
            return None
        return round(numerator / denominator * 100, 2)
    except Exception:
        return None

def float_or_none(x):
    try:
        return None if x is None else float(x)
    except Exception:
        return None

def int_or_none(x):
    try:
        return None if x is None else int(x)
    except Exception:
        return None

def avg_volume_before(seed_date: date, hist_indexed: pd.DataFrame, days: int = 5):
    prev_days = sorted([d for d in hist_indexed.index if d < seed_date], reverse=True)[:days]
    if not prev_days:
        return None
    volumes = [hist_indexed.at[d, "Volume"] or 0 for d in prev_days]
    return sum(volumes) / len(volumes)

# ---------- DASH APP ----------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server
cache = Cache(app.server, config=CACHE_CONFIG)

start, end = get_week_bounds()
default_start_iso = iso_date(start)
default_end_iso = iso_date(end)

controls = dbc.Card(
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.Label("Week Start"),
                dcc.DatePickerSingle(id="week-start", date=default_start_iso)
            ], md=3),
            dbc.Col([
                html.Label("Week End"),
                dcc.DatePickerSingle(id="week-end", date=default_end_iso)
            ], md=3),
            dbc.Col([
                html.Label("Min price (USD)"),
                dcc.Input(id="min-price", type="number", value=1, min=0, step=0.01)
            ], md=2),
            dbc.Col([
                html.Label("Min absolute % move (|%)"),
                dcc.Input(id="min-pct", type="number", value=0, min=0, step=0.1)
            ], md=2),
            dbc.Col([
                html.Br(),
                dbc.Button("Load Week", id="load-week", color="primary")
            ], md=2)
        ]),
        html.Div(id="controls-note", style={"marginTop": "8px", "fontSize": "12px", "color": "#666"})
    ]),
    className="mb-3"
)

app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H2("Earnings Moves Analyzer"), width=12)),
    dbc.Row(dbc.Col(html.P("Visualize and analyze stock moves around earnings (Finnhub + yfinance)."), width=12)),
    dbc.Row(dbc.Col(controls, width=12)),
    dbc.Row([
        dbc.Col(html.Div(id="earnings-table-container"), md=6),
        dbc.Col(html.Div(id="ticker-summary"), md=6)
    ]),
    dbc.Row(dbc.Col(html.Div(id="chart-container"), width=12), className="mt-3"),
    dbc.Row([
        dbc.Col(dbc.Button("Download CSV (history)", id="download-csv", color="secondary"), md=2),
        dbc.Col(html.Div(id="download-link"), md=10)
    ], className="mt-2"),
    dcc.Store(id="earnings-week-store"),
    dcc.Store(id="selected-ticker-store"),
    dcc.Store(id="history-data-store")
], fluid=True)

@app.callback(
    Output("earnings-week-store", "data"),
    Output("controls-note", "children"),
    Input("load-week", "n_clicks"),
    State("week-start", "date"),
    State("week-end", "date"),
    State("min-price", "value"),
    prevent_initial_call=False
)
def load_week(n_clicks, week_start, week_end, min_price):
    try:
        if not week_start or not week_end:
            return dash.no_update, "Please select a valid week start and end."
        start_iso = week_start
        end_iso = week_end
        df = fetch_earnings_week(start_iso, end_iso)
        if df.empty:
            return [], f"No earnings found between {start_iso} and {end_iso}."
        df = df.drop_duplicates(subset=["symbol"]).reset_index(drop=True)
        df["last_price"] = df["symbol"].apply(lambda s: fetch_last_price(s) or 0)
        if min_price is not None:
            df = df[df["last_price"] >= float(min_price)]
        df = df.sort_values(by="last_price", ascending=False).reset_index(drop=True)
        note = f"Loaded {len(df)} tickers. Data cached for 10 minutes."
        return df.to_dict("records"), note
    except Exception as e:
        return [], f"Error loading week: {e}"

@app.callback(
    Output("earnings-table-container", "children"),
    Input("earnings-week-store", "data")
)
def render_earnings_table(data):
    if not data:
        return html.Div("No earnings loaded. Pick a week and click 'Load Week'.")
    df = pd.DataFrame(data)
    table = dash_table.DataTable(
        id="earnings-table",
        columns=[
            {"name": "Ticker", "id": "symbol"},
            {"name": "Company", "id": "company"},
            {"name": "Earnings Date", "id": "date", "type": "datetime"},
            {"name": "Time", "id": "time"},
            {"name": "Last Price", "id": "last_price", "type": "numeric", "format": {"specifier": ".2f"}},
        ],
        data=df.to_dict("records"),
        row_selectable="single",
        selected_rows=[],
        style_cell={"textAlign": "left"},
        page_size=12
    )
    return html.Div([
        html.H5("Upcoming earnings (week)"),
        table
    ])

@app.callback(
    Output("selected-ticker-store", "data"),
    Input("earnings-table", "selected_rows"),
    State("earnings-week-store", "data")
)
def select_ticker(selected_rows, week_data):
    if not week_data or selected_rows is None or not selected_rows:
        return None
    ticker = pd.DataFrame(week_data).iloc[selected_rows[0]]["symbol"]
    return {"ticker": ticker}

@app.callback(
    Output("history-data-store", "data"),
    Output("ticker-summary", "children"),
    Input("selected-ticker-store", "data"),
    Input("min-pct", "value"),
    prevent_initial_call=False
)
def load_history(selected_ticker_data, min_pct):
    if not selected_ticker_data:
        return dash.no_update, html.Div("Select a ticker from the table to load historical earnings moves.")
    ticker = selected_ticker_data.get("ticker")
    if not ticker:
        return dash.no_update, html.Div("Invalid ticker selected.")
    hist = fetch_historical_prices(ticker, period="5y")
    if hist.empty:
        return {}, html.Div(f"No price history found for {ticker}.")
    evs = fetch_finnhub_earnings_for_symbol(ticker)
    normalized_events = []
    for e in evs:
        ev_date = None
        if isinstance(e, dict):
            for k in ("date", "period", "startDate", "startdatetime"):
                if k in e and e[k]:
                    try:
                        ev_date = pd.to_datetime(e[k]).date()
                        break
                    except Exception:
                        pass
            ev_time = e.get("time") or e.get("hour") or e.get("epsReportTime") or e.get("startdatetimetype") or e.get("reportTime") or "TBA"
            normalized_events.append({"date": ev_date.isoformat() if ev_date else None, "time": ev_time, **e})
    history_events = normalized_events
    moves_df = compute_earnings_moves_from_hist(hist, history_events)
    if moves_df.empty:
        summary = html.Div(f"No earnings moves found for {ticker}.")
        return {}, summary
    if min_pct is not None and min_pct > 0:
        moves_df = moves_df[moves_df["%move_at_close"].abs() >= float(min_pct)]
    avg_move = moves_df["%move_at_close"].dropna()
    avg_text = f"Avg % move at close: {round(avg_move.mean(),2)}%" if not avg_move.empty else "N/A"
    spikes = moves_df[moves_df["volume_spike"] == True]
    spike_text = f"Volume spikes: {len(spikes)}"
    card = dbc.Card(dbc.CardBody([
        html.H4(f"{ticker}", className="card-title"),
        html.P(avg_text),
        html.P(spike_text),
        html.P(f"Events found: {len(moves_df)}"),
        dbc.Button("Show chart & table", id="show-chart-btn", color="primary")
    ]))
    return moves_df.to_dict("records"), card

@app.callback(
    Output("chart-container", "children"),
    Input("history-data-store", "data"),
    State("selected-ticker-store", "data"),
    prevent_initial_call=False
)
def render_chart(history_data, selected_ticker_data):
    if not history_data or not selected_ticker_data:
        return html.Div("Select a ticker and press 'Show chart & table'.")
    df = pd.DataFrame(history_data)
    ticker = selected_ticker_data.get("ticker")
    price_hist = fetch_historical_prices(ticker, period="5y")
    if price_hist.empty:
        return html.Div("No price history available for plotting.")
    price_hist_plot = price_hist.copy()
    price_hist_plot["DateStr"] = pd.to_datetime(price_hist_plot["Date"]).astype(str)
    cand = go.Candlestick(
        x=price_hist_plot["DateStr"],
        open=price_hist_plot["Open"],
        high=price_hist_plot["High"],
        low=price_hist_plot["Low"],
        close=price_hist_plot["Close"],
        name="Price"
    )
    markers = []
    for idx, row in df.iterrows():
        post_date = row.get("post_date") or row.get("postDate")
        if not post_date:
            continue
        try:
            yval = price_hist_plot.loc[price_hist_plot["Date"].astype(str) == post_date, "Close"].values
            if len(yval) == 0:
                continue
            yval = float(yval[0])
        except Exception:
            continue
        color = "green" if row.get("%move_at_close", 0) >= 0 else "red"
        text = f"{row.get('earnings_date')}<br>{row.get('%move_at_close')}% ({row.get('report_time')})"
        markers.append(go.Scatter(
            x=[post_date],
            y=[yval],
            mode="markers+text",
            marker=dict(color=color, size=10),
            text=[f"{row.get('%move_at_close')}%"],
            textposition="top center",
            hoverinfo="text",
            hovertext=text,
            showlegend=False
        ))
    layout = go.Layout(
        title=f"{ticker} price (candles) with earnings markers",
        xaxis=dict(rangeslider=dict(visible=False)),
        autosize=True,
        height=600,
    )
    fig = go.Figure(data=[cand] + markers, layout=layout)
    table = dash_table.DataTable(
        id="history-table",
        columns=[{"name": c, "id": c} for c in pd.DataFrame(history_data).columns],
        data=history_data,
        page_size=10,
        style_cell={"textAlign": "center"}
    )
    return html.Div([
        dcc.Graph(figure=fig, config={"displayModeBar": True}),
        html.H5("Historical earnings moves"),
        table
    ])

@app.callback(
    Output("download-link", "children"),
    Input("download-csv", "n_clicks"),
    State("history-data-store", "data"),
    State("selected-ticker-store", "data"),
    prevent_initial_call=True
)
def generate_csv(n_clicks, history_data, selected_ticker_data):
    if not history_data or not selected_ticker_data:
        return html.Div("No data to download.")
    df = pd.DataFrame(history_data)
    ticker = selected_ticker_data.get("ticker")
    csv_str = df.to_csv(index=False)
    href = "data:text/csv;charset=utf-8," + requests.utils.requote_uri(csv_str)
    filename = f"{ticker}_earnings_history.csv"
    return html.A("Download CSV", href=href, download=filename, className="btn btn-outline-secondary")

# ---------- RUN ----------
if __name__ == "__main__":
    app.run_server(debug=True, port=8050)