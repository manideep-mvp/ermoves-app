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

# Finnhub API Key from environment
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
if not FINNHUB_API_KEY:
    raise ValueError("Please set your Finnhub API key in the environment variable 'FINNHUB_API_KEY'")

def fetch_finnhub_earnings_for_day(day_name):
    today = dt.date.today()
    weekday_map = {
        'Monday': 0,
        'Tuesday': 1,
        'Wednesday': 2,
        'Thursday': 3,
        'Friday': 4
    }
    
    target_weekday = weekday_map.get(day_name, 0)
    days_ahead = target_weekday - today.weekday()
    if days_ahead < 0:
        days_ahead += 7
    target_date = today + dt.timedelta(days=days_ahead)
    from_date = target_date.strftime('%Y-%m-%d')
    to_date = from_date  # single day earnings calendar
    
    url = "https://finnhub.io/api/v1/calendar/earnings"
    params = {
        "from": from_date,
        "to": to_date,
        "token": FINNHUB_API_KEY
    }
    res = requests.get(url, params=params)
    res.raise_for_status()
    data = res.json()
    
    earnings = data.get("earningsCalendar") or data.get("earnings") or []
    df = pd.DataFrame(earnings)
    if df.empty:
        return df
    
    # Rename to match your existing columns
    df.rename(columns={
        "symbol": "ticker",
        "companyShortName": "company",
        "date": "earningsday",
        "hour": "earningstime"
    }, inplace=True)
    
    # Fetch last price for each ticker using yfinance (slow for many tickers)
    def get_last_price(ticker):
        try:
            hist = yf.Ticker(ticker).history(period='1d')
            if not hist.empty:
                return hist['Close'][-1]
            else:
                return None
        except:
            return None
        
    df['last price'] = df['ticker'].apply(get_last_price)
    
    keep_cols = ['ticker', 'company', 'earningstime', 'last price', 'earningsday']
    df = df[keep_cols]
    df.sort_values(by='last price', ascending=False, inplace=True)
    
    return df


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(suppress_callback_exceptions=True, external_stylesheets=external_stylesheets)
server = app.server

colors = {'background': '#FDFEFE',
          'text': '#283747 '}

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    html.Div(id='intermediate-value', style={'display': 'none'}),
    html.Div(id='intermediate-value2', style={'display': 'none'})
])

home_page = html.Div(
    style={'backgroundColor': colors['background']},
    children=[
        html.Div([
            html.H4(style={'display': 'inline-block', 'color': colors['text']}, children='Upcoming Earnings - '),
            html.Div(children='Select individual stock from the table, click "load previous Earnings data" button, and hit Earnings History button to get historical earning moves',
                     style={"margin-left": "15px", 'display': 'inline-block', 'float': 'middle', 'color': colors['text']})
        ]),

        html.Div(id='selected-ticker', style={'display': 'inline-block'}),

        dbc.Button(children='Load Previous Earnings data',
                   id='load-button',
                   outline=True, color="dark",
                   n_clicks=0,
                   style={"margin-bottom": "10px", "margin-left": "5px", 'display': 'inline-block'}),

        html.Div(style={'display': 'inline-block', 'float': 'right'},
                 children=[dcc.Link(dbc.Button(children='Earnings History',
                                              id='output-button',
                                              outline=True, color="dark"),
                                    href='/earn-hist')]),

        dcc.Tabs(id='testing_tabs', value='tab-1',
                 children=[
                     dcc.Tab(label='Monday', value='tab-1'),
                     dcc.Tab(label='Tuesday', value='tab-2'),
                     dcc.Tab(label='Wednesday', value='tab-3'),
                     dcc.Tab(label='Thursday', value='tab-4'),
                     dcc.Tab(label='Friday', value='tab-5'),
                 ],
                 colors={
                     "border": "solid orange",
                     "primary": "solid orange",
                     "background": "grey"
                 },
                 style={
                     'fontFamily': 'system-ui',
                     'borderRadius': '15px',
                     'overflow': 'hidden',
                 }
                 ),
        html.Div(id='tabs-test-content')
    ]
)

@app.callback(Output('tabs-test-content', 'children'),
              [Input('testing_tabs', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        df = fetch_finnhub_earnings_for_day('Monday')
    elif tab == 'tab-2':
        df = fetch_finnhub_earnings_for_day('Tuesday')
    elif tab == 'tab-3':
        df = fetch_finnhub_earnings_for_day('Wednesday')
    elif tab == 'tab-4':
        df = fetch_finnhub_earnings_for_day('Thursday')
    elif tab == 'tab-5':
        df = fetch_finnhub_earnings_for_day('Friday')
    else:
        df = pd.DataFrame()
    
    if df.empty:
        return html.Div("No earnings data found for this day.")
    
    return dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
        row_selectable="single",
        selected_rows=[],
        style_cell={'textAlign': 'left'},
        style_table={
            'border': '1px solid orange',
            'borderRadius': '15px',
            'overflow': 'hidden',
            'fontFamily': 'Sans-Serif',
            'fontSize': '13px',
            'overflowX': 'scroll'
        }
    )

@app.callback(Output('selected-ticker', 'children'),
              [Input('table', 'selected_rows'),
               Input('table', 'data')])
def get_selected_row(selected_rows, data):
    if not data:
        return ''
    df = pd.DataFrame(data)
    if selected_rows and len(selected_rows) > 0:
        stockstr = df.loc[selected_rows[0], 'ticker']
        return stockstr
    else:
        return ''

@app.callback(Output('intermediate-value2', 'children'),
              [Input('load-button', 'n_clicks')],
              [State('selected-ticker', 'children')])
def update_output1(n_clicks, selticker):
    if selticker and n_clicks > 0:
        return selticker

@app.callback(Output('intermediate-value', 'children'),
              [Input('load-button', 'n_clicks')],
              [State('selected-ticker', 'children')])
def update_output(n_clicks, selticker):
    if selticker and n_clicks > 0:
        stock = selticker
        stock_data = yf.Ticker(stock)
        data = stock_data.history(period="5y", actions=False)
        data = data.reset_index()

        earnings_dates = si.get_earnings_history(stock)

        date_outrange = []
        ER_dates = []

        pre_columns = ['pre' + str(x) for x in data.columns]
        post_columns = ['post' + str(x) for x in data.columns]

        earning_data_pre = pd.DataFrame()
        earning_data_post = pd.DataFrame()

        for i in range(len(earnings_dates)):
            each_date = earnings_dates[i]['startdatetime'][:10]
            if i < len(earnings_dates) - 1:
                next_date = earnings_dates[i + 1]['startdatetime'][:10]

            if dt.datetime.strptime(each_date, '%Y-%m-%d').date() < min(data['Date']) or dt.datetime.strptime(each_date,
                                                                                                         '%Y-%m-%d').date() > max(
                    data['Date']):
                date_outrange.append(each_date)
            else:
                if ((dt.datetime.strptime(each_date, '%Y-%m-%d').date() - dt.datetime.strptime(next_date, '%Y-%m-%d').date()).days) > 7:
                    ER_dates.append(each_date)

        ER_dates = list(dict.fromkeys(ER_dates))

        if earnings_dates[0]['startdatetimetype'] == 'AMC' or earnings_dates[0]['startdatetimetype'] == 'TNS' or int(
                earnings_dates[0]['startdatetime'].split('T')[1][:2]) >= 16:

            for dtes in ER_dates:
                dtindex = data[data['Date'] == dtes].index
                if dtindex.any() and dtindex[0] < len(data.index) - 1:
                    earning_data_pre = earning_data_pre.append(data.iloc[dtindex])
                    earning_data_post = earning_data_post.append(data.iloc[dtindex[0] + 1])

            earning_data_post = earning_data_post.reset_index(drop=True)
            earning_data_pre = earning_data_pre.reset_index(drop=True)
            earning_data_pre.columns = pre_columns
            earning_data_post.columns = post_columns

        elif earnings_dates[0]['startdatetimetype'] == 'BMO' or int(earnings_dates[0]['startdatetime'].split('T')[1][:2]) < 16:

            for dtes in ER_dates:
                dtindex = data[data['Date'] == dtes].index
                if dtindex.any() and dtindex[0] > 0:
                    earning_data_pre = earning_data_pre.append(data.iloc[dtindex[0] - 1])
                    earning_data_post = earning_data_post.append(data.iloc[dtindex])

            earning_data_post = earning_data_post.reset_index(drop=True)
            earning_data_pre = earning_data_pre.reset_index(drop=True)
            earning_data_pre.columns = pre_columns
            earning_data_post.columns = post_columns

        earning_data_pre['pre_average'] = (earning_data_pre.preHigh + earning_data_pre.preLow) / 2
        earning_data_post['post_average'] = (earning_data_post.postHigh + earning_data_post.postLow) / 2

        final_data = pd.concat([earning_data_pre, earning_data_post], axis=1)

        final_data['% move at open'] = ((final_data.postOpen - final_data.preClose) / final_data.preClose) * 100
        final_data['% move at close'] = ((final_data.postClose - final_data.preClose) / final_data.preClose) * 100
        final_data['% intraday move'] = ((final_data.postClose - final_data.postOpen) / final_data.preClose) * 100
        final_data['% highest move'] = ((final_data.postHigh - final_data.preClose) / final_data.preClose) * 100
        final_data['% lowest move'] = ((final_data.postLow - final_data.preClose) / final_data.preClose) * 100

        final_data['postDate'] = pd.DatetimeIndex(final_data['postDate']).strftime("%b-%d-%Y")
        final_data['preDate'] = pd.DatetimeIndex(final_data['preDate']).strftime("%b-%d-%Y")

        del final_data['preVolume']
        del final_data['postVolume']

        final_data = final_data.round(0)

        return final_data.to_json(orient='split')


earnhist_layout = html.Div(
    style={'backgroundColor': colors['background']},
    children=[
        html.Div([
            html.H4(style={'display': 'inline-block', 'color': colors['text']},
                    children='Earning Stats for previous quarters for '),
            dcc.Loading(
                id="loading-1",
                type="circle",
                children=html.H4(id='stock-ticker', style={"margin-left": "10px", 'display': 'inline-block', 'color': colors['text']}),
                color="#AA6924")
        ]),

        html.Div(id='table_stats'),

        dcc.Link('Go back to home', href='/')
    ]
)

@app.callback(Output('stock-ticker', 'children'), [Input('intermediate-value2', 'children')])
def update_str(strtkr):
    if strtkr:
        time.sleep(2.5)
        return strtkr

@app.callback(Output('table_stats', 'children'), [Input('intermediate-value', 'children')])
def update_table(finaldata):
    if finaldata:
        dff = pd.read_json(finaldata, orient='split')
        return dash_table.DataTable(
            id='table_stat',
            columns=[{"name": i, "id": i} for i in dff.columns],
            data=dff.to_dict('records'),
            style_cell={'textAlign': 'center'},
            style_table={
                'border': '1px solid orange',
                'borderRadius': '15px',
                'overflow': 'hidden',
                'fontFamily': 'Sans-Serif',
                'fontSize': '13px',
                'overflowX': 'scroll'
            }
        )
    else:
        return html.Div([html.H4(children='No Stock Selected')])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/earn-hist':
        return earnhist_layout
    else:
        return home_page


if __name__ == '__main__':
    app.run_server(debug=True)