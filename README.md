# NIFTY OI UI

Short Streamlit app for monitoring **NIFTY options open interest (OI)** around ATM (`ATM ± 100`) using Zerodha Kite data.  
It fetches live option/futures quotes, stores minute snapshots in a local DuckDB file, and shows:

- OI board for CE/PE strikes
- 1-minute and 5-minute OI deltas
- PCR and net OI bias (PE - CE)
- Basic OI trend charts from stored snapshots

## Dependencies

The project needs:

- Python 3.10+ (for `zoneinfo` and modern typing)
- `streamlit`
- `pandas`
- `numpy`
- `duckdb`
- `python-dotenv`
- `kiteconnect`

Install with:

```bash
pip install streamlit pandas numpy duckdb python-dotenv kiteconnect
```

## Required Environment Variables

Create a `.env` file with:

```env
KITE_API_KEY=your_api_key
KITE_API_SECRET=your_api_secret
KITE_ACCESS_TOKEN=your_access_token
```

## Run

```bash
source .venv/bin/activate
streamlit run app.py
```

If you need to generate a fresh access token, use:

```bash
python token_gen.py
```

## Metrics Guide

For explanations of the top dashboard metrics and how to read them, see:

- [METRICS_GUIDE.md](./METRICS_GUIDE.md)
