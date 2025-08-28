
SPX Premarket Dashboard - Files included:

1) spx_premarket_dashboard.py
   - Streamlit app. Run with:
       pip install streamlit yfinance pandas numpy plotly pandas_ta requests reportlab
       streamlit run spx_premarket_dashboard.py

2) spx_premarket_dashboard_instructions.pdf
   - PDF containing step-by-step instructions and the script for easy copy/paste.

If you want the app to include SPX options open interest (OI) and implied volatility (IV),
you must provide an options API and edit the following variables inside the Python script:
   - OPTIONS_API_ENABLED = True
   - OPTIONS_API_URL = "https://YOUR_OPTION_API_ENDPOINT"
   - OPTIONS_API_KEY = "YOUR_API_KEY"
and adapt the fetch_options_data_api() function to parse your provider's JSON.

Common issues:
 - Yahoo sometimes limits 1m data or returns empty frames; use 5m interval.
 - If pandas_ta is not installed, ATR will be computed via fallback logic.
