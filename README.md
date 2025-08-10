# Earnings Moves Analyzer (Finnhub + Dash)

This repository contains a Dash web app that fetches earnings calendar data from **Finnhub** and price history from **yfinance**, computes post-earnings moves, and displays interactive candlestick charts with earnings markers.

## Quick start (local)

1. Clone the repo:
```bash
git clone <this-repo-url>
cd earnings-analyzer
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate    # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and set your Finnhub API key:
```bash
cp .env.example .env
# edit .env and add your FINNHUB_API_KEY
```

4. Run locally:
```bash
python app.py
# open http://127.0.0.1:8050 in your browser
```

## Deploying to a cloud host

This app is ready for deployment on services like **Render**, **Railway**, or **Heroku**.
Make sure to add the environment variable `FINNHUB_API_KEY` in the host settings.

**Procfile** included: `web: gunicorn app:server`

## Notes
- Finnhub has rate limits; cache and avoid hammering the API.
- yfinance may throttle for heavy usage.

---