import os
from datetime import datetime, date, timedelta, time as dtime
import math
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from kiteconnect import KiteConnect
import duckdb
import time
import numpy as np
from zoneinfo import ZoneInfo

st.set_page_config(page_title="NIFTY OI UI - Step 5", layout="wide")
load_dotenv()

API_KEY = os.getenv("KITE_API_KEY")
ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN")

st.title("📊 NIFTY OI UI — Step 5 (Net Bias + PCR + Impulses)")

if not API_KEY or not ACCESS_TOKEN:
    st.error("Missing KITE_API_KEY or KITE_ACCESS_TOKEN in .env")
    st.stop()

kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

DB_PATH = "oi_snapshots.duckdb"

# -------------------- Utilities --------------------
@st.cache_data(ttl=3600)
def load_instruments():
    inst = kite.instruments()
    return pd.DataFrame(inst)

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)

def now_ist() -> datetime:
    return datetime.now(IST)

def is_market_open(dt_ist: datetime) -> bool:
    t = dt_ist.time()
    return (t >= MARKET_OPEN) and (t <= MARKET_CLOSE)

def round_to_50(x: float) -> int:
    return int(round(x / 50.0) * 50)

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_price(spot: float, strike: float, t_years: float, r: float, vol: float, right: str) -> float:
    if t_years <= 0.0 or vol <= 0.0 or spot <= 0.0 or strike <= 0.0:
        intrinsic = max(spot - strike, 0.0) if right == "CE" else max(strike - spot, 0.0)
        return intrinsic

    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (r + 0.5 * vol * vol) * t_years) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    df = math.exp(-r * t_years)
    if right == "CE":
        return spot * norm_cdf(d1) - strike * df * norm_cdf(d2)
    return strike * df * norm_cdf(-d2) - spot * norm_cdf(-d1)

def implied_volatility(
    option_price: float,
    spot: float,
    strike: float,
    t_years: float,
    right: str,
    r: float = 0.06,
    min_vol: float = 1e-4,
    max_vol: float = 5.0,
    max_iter: int = 60,
    tol: float = 1e-4,
) -> float | None:
    if (
        not math.isfinite(option_price)
        or not math.isfinite(spot)
        or not math.isfinite(strike)
        or not math.isfinite(t_years)
        or option_price <= 0.0
        or spot <= 0.0
        or strike <= 0.0
        or t_years <= 0.0
    ):
        return None

    intrinsic = max(spot - strike, 0.0) if right == "CE" else max(strike - spot, 0.0)
    if option_price < intrinsic:
        return None

    low = min_vol
    high = max_vol
    low_p = bs_price(spot, strike, t_years, r, low, right)
    high_p = bs_price(spot, strike, t_years, r, high, right)
    if option_price < low_p or option_price > high_p:
        return None

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        mid_p = bs_price(spot, strike, t_years, r, mid, right)
        err = mid_p - option_price
        if abs(err) <= tol:
            return mid
        if err > 0:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)

def time_to_expiry_years(expiry_iso: str, now_ts_ist: datetime) -> float:
    expiry_dt = datetime.combine(pd.to_datetime(expiry_iso).date(), MARKET_CLOSE).replace(tzinfo=IST)
    dt_secs = (expiry_dt - now_ts_ist).total_seconds()
    return max(dt_secs / (365.0 * 24.0 * 3600.0), 1e-8)

def infer_vol_state(atm_iv: float | None, iv_series: pd.Series) -> str:
    if atm_iv is None or iv_series.empty:
        return "—"
    iv_clean = pd.to_numeric(iv_series, errors="coerce").dropna()
    if len(iv_clean) < 10:
        return "NORMAL"
    p33 = float(iv_clean.quantile(0.33))
    p66 = float(iv_clean.quantile(0.66))
    if atm_iv <= p33:
        return "LOW"
    if atm_iv >= p66:
        return "HIGH"
    return "NORMAL"

def compute_adx(price_df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Expects columns: high, low, close sorted by time.
    Returns ADX series aligned to input index.
    """
    if price_df.empty:
        return pd.Series(dtype=float)

    high = pd.to_numeric(price_df["high"], errors="coerce")
    low = pd.to_numeric(price_df["low"], errors="coerce")
    close = pd.to_numeric(price_df["close"], errors="coerce")

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()

    plus_dm_s = pd.Series(plus_dm, index=price_df.index).ewm(alpha=1.0 / period, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm, index=price_df.index).ewm(alpha=1.0 / period, adjust=False).mean()

    plus_di = 100.0 * plus_dm_s / atr.replace(0, np.nan)
    minus_di = 100.0 * minus_dm_s / atr.replace(0, np.nan)
    dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0.0)
    adx = dx.ewm(alpha=1.0 / period, adjust=False).mean()
    return adx

def classify_quadrant(d_price: float, d_oi: float, eps_price: float = 0.01, eps_oi: float = 1.0) -> str:
    if not math.isfinite(d_price) or not math.isfinite(d_oi):
        return "UNKNOWN"
    if abs(d_price) <= eps_price or abs(d_oi) <= eps_oi:
        return "FLAT-MIXED"
    if d_price > 0 and d_oi > 0:
        return "LONG_BUILD"
    if d_price < 0 and d_oi > 0:
        return "SHORT_BUILD"
    if d_price > 0 and d_oi < 0:
        return "SHORT_COVER"
    return "LONG_UNWIND"

def sign_label(x: float, eps: float = 0.0) -> str:
    if not math.isfinite(x):
        return "NA"
    if x > eps:
        return "BULLISH"
    if x < -eps:
        return "BEARISH"
    return "NEUTRAL"

def mtf_alignment_label(v1: float, v5: float, v15: float) -> str:
    s1 = sign_label(v1)
    s5 = sign_label(v5)
    s15 = sign_label(v15)
    if s1 == "BULLISH" and s5 == "BULLISH" and s15 == "BULLISH":
        return "BULLISH_STRONG"
    if s1 == "BEARISH" and s5 == "BEARISH" and s15 == "BEARISH":
        return "BEARISH_STRONG"
    if s1 == "BULLISH" and s5 == "BULLISH" and s15 != "BEARISH":
        return "BULLISH_WEAK"
    if s1 == "BEARISH" and s5 == "BEARISH" and s15 != "BULLISH":
        return "BEARISH_WEAK"
    return "MIXED/NO-TRADE"

def top2_oi_share(df: pd.DataFrame, right: str) -> float:
    x = df[df["opt_right"] == right].copy()
    if x.empty:
        return float("nan")
    by_strike = x.groupby("strike")["oi"].sum().sort_values(ascending=False)
    total = float(by_strike.sum())
    if total <= 0:
        return float("nan")
    return float(by_strike.head(2).sum() / total)

def find_wall(snapshot: pd.DataFrame, right: str) -> tuple[int | None, float]:
    x = snapshot[(snapshot["opt_right"] == right)].copy()
    if x.empty:
        return None, float("nan")
    by_strike = x.groupby("strike")["d_oi_5m_num"].sum().sort_values(ascending=False)
    if by_strike.empty:
        return None, float("nan")
    strike = int(by_strike.index[0])
    return strike, float(by_strike.iloc[0])

def shift_label(curr: int | None, prev: int | None) -> str:
    if curr is None:
        return "NA"
    if prev is None:
        return "NEW"
    if curr > prev:
        return "UP"
    if curr < prev:
        return "DOWN"
    return "UNCHANGED"

def human_compact(n: float) -> str:
    if not math.isfinite(n):
        return "—"
    abs_n = abs(n)
    if abs_n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if abs_n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if abs_n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return f"{n:.0f}"

def pretty_enum(s: str) -> str:
    if not s:
        return "—"
    return s.replace("_", " ").title()

def pick_nearest_nifty_fut(inst_df: pd.DataFrame) -> tuple[int, str, str]:
    df = inst_df.copy()
    df = df[(df["exchange"] == "NFO") & (df["segment"] == "NFO-FUT") & (df["name"] == "NIFTY")]
    df["expiry"] = pd.to_datetime(df["expiry"])
    today = pd.Timestamp(date.today())
    df = df[df["expiry"] >= today].sort_values("expiry", ascending=True)
    if df.empty:
        raise RuntimeError("No upcoming NIFTY futures found.")
    row = df.iloc[0]
    return int(row["instrument_token"]), str(row["tradingsymbol"]), row["expiry"].date().isoformat()

def pick_nearest_weekly_nifty_opt_expiry(inst_df: pd.DataFrame) -> str:
    df = inst_df.copy()
    df = df[(df["exchange"] == "NFO") & (df["segment"] == "NFO-OPT") & (df["name"] == "NIFTY")]
    df["expiry"] = pd.to_datetime(df["expiry"])
    today = pd.Timestamp(date.today())
    df = df[df["expiry"] >= today].sort_values("expiry", ascending=True)
    if df.empty:
        raise RuntimeError("No upcoming NIFTY option expiries found.")
    return df.iloc[0]["expiry"].date().isoformat()

def resolve_option_instruments(inst_df: pd.DataFrame, expiry_iso: str, strikes: list[int]) -> pd.DataFrame:
    expiry_dt = pd.to_datetime(expiry_iso).date()
    df = inst_df.copy()
    df = df[(df["exchange"] == "NFO") & (df["segment"] == "NFO-OPT") & (df["name"] == "NIFTY")]
    df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
    df = df[df["expiry"] == expiry_dt]
    if df.empty:
        raise RuntimeError(f"No NIFTY options found for expiry {expiry_iso}.")
    df["strike"] = df["strike"].astype(float).astype(int)
    df = df[df["strike"].isin(strikes)]
    if df.empty:
        raise RuntimeError(f"No option instruments for strikes {strikes} on expiry {expiry_iso}.")
    out = df[["tradingsymbol", "instrument_token", "strike", "instrument_type", "expiry"]].copy()
    out = out.sort_values(["strike", "instrument_type"], ascending=[True, True])
    return out

def init_db():
    con = duckdb.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS oi_snapshots (
            ts TIMESTAMP,
            trade_date DATE,
            fut_symbol VARCHAR,
            fut_ltp DOUBLE,
            opt_expiry DATE,
            strike INTEGER,
            opt_right VARCHAR,
            tradingsymbol VARCHAR,
            instrument_token BIGINT,
            oi BIGINT,
            ltp DOUBLE
        )
    """)
    cols = {row[1] for row in con.execute("PRAGMA table_info('oi_snapshots')").fetchall()}
    if "volume" not in cols:
        con.execute("ALTER TABLE oi_snapshots ADD COLUMN volume BIGINT")
    con.close()

def insert_snapshot(rows: pd.DataFrame):
    con = duckdb.connect(DB_PATH)
    con.register("df_rows", rows)
    con.execute("INSERT INTO oi_snapshots SELECT * FROM df_rows")
    con.close()

def load_day(trade_date_iso: str, opt_expiry_iso: str) -> pd.DataFrame:
    con = duckdb.connect(DB_PATH)
    q = """
        SELECT *
        FROM oi_snapshots
        WHERE trade_date = ? AND opt_expiry = ?
        ORDER BY ts ASC
    """
    df = con.execute(q, [trade_date_iso, opt_expiry_iso]).fetchdf()
    con.close()
    return df

def load_yday_close(trade_date_iso: str, opt_expiry_iso: str) -> pd.DataFrame:
    """
    Yesterday close baseline: take the last snapshot after 15:25 if available,
    else take the last snapshot of yesterday.
    """
    yday = (pd.to_datetime(trade_date_iso).date() - timedelta(days=1)).isoformat()
    con = duckdb.connect(DB_PATH)
    q = """
        SELECT *
        FROM oi_snapshots
        WHERE trade_date = ? AND opt_expiry = ?
        ORDER BY ts ASC
    """
    df = con.execute(q, [yday, opt_expiry_iso]).fetchdf()
    con.close()
    if df.empty:
        return df

    df = df.sort_values("ts")
    # Prefer last snapshot after 15:25
    cutoff = datetime.combine(pd.to_datetime(yday).date(), dtime(15, 25))
    after = df[df["ts"] >= cutoff]
    if not after.empty:
        last = after.groupby("tradingsymbol").tail(1)
    else:
        last = df.groupby("tradingsymbol").tail(1)

    return last[["tradingsymbol", "oi"]].rename(columns={"oi": "oi_yday_close"})

def compute_deltas(hist: pd.DataFrame) -> pd.DataFrame:
    if hist.empty:
        return hist
    hist = hist.sort_values(["tradingsymbol", "ts"]).copy()
    hist["oi_prev_1"] = hist.groupby("tradingsymbol")["oi"].shift(1)
    hist["oi_prev_5"] = hist.groupby("tradingsymbol")["oi"].shift(5)
    hist["d_oi_1m"] = hist["oi"] - hist["oi_prev_1"]
    hist["d_oi_5m"] = hist["oi"] - hist["oi_prev_5"]
    hist["ltp_prev_1"] = hist.groupby("tradingsymbol")["ltp"].shift(1)
    hist["ltp_prev_5"] = hist.groupby("tradingsymbol")["ltp"].shift(5)
    hist["d_ltp_1m"] = hist["ltp"] - hist["ltp_prev_1"]
    hist["d_ltp_5m"] = hist["ltp"] - hist["ltp_prev_5"]
    return hist

def should_insert_snapshot(now_ts: datetime) -> bool:
    minute_key = now_ts.strftime("%Y-%m-%d %H:%M")
    last_key = st.session_state.get("last_insert_minute_key")
    if last_key == minute_key:
        return False
    st.session_state["last_insert_minute_key"] = minute_key
    return True

def load_latest_snapshot(trade_date_iso: str, opt_expiry_iso: str) -> pd.DataFrame:
    con = duckdb.connect(DB_PATH)
    q = """
        SELECT *
        FROM oi_snapshots
        WHERE trade_date = ? AND opt_expiry = ?
        QUALIFY ROW_NUMBER() OVER (PARTITION BY tradingsymbol ORDER BY ts DESC) = 1
    """
    df = con.execute(q, [trade_date_iso, opt_expiry_iso]).fetchdf()
    con.close()
    return df

def add_impulse_tags(latest: pd.DataFrame, hist: pd.DataFrame, lookback_minutes: int = 30) -> pd.DataFrame:
    """
    Adds an impulse tag if abs(Δ1m) > mean + 3*std over last N minutes for that symbol.
    """
    if latest.empty or hist.empty:
        latest["impulse"] = ""
        return latest

    cutoff = latest["ts"].max() - timedelta(minutes=lookback_minutes)
    recent = hist[hist["ts"] >= cutoff].copy()

    stats = recent.groupby("tradingsymbol")["d_oi_1m"].agg(["mean", "std"]).reset_index()
    merged = latest.merge(stats, on="tradingsymbol", how="left")

    # Handle std=0 / NaN
    merged["std"] = merged["std"].fillna(0.0)
    merged["mean"] = merged["mean"].fillna(0.0)

    thr = merged["mean"].abs() + 3.0 * merged["std"]
    merged["impulse"] = np.where(merged["d_oi_1m"].abs() > thr, "IMPULSE", "")
    return merged

def delete_outside_session(trade_date_iso: str, opt_expiry_iso: str):
    con = duckdb.connect(DB_PATH)
    con.execute("""
        DELETE FROM oi_snapshots
        WHERE trade_date = ? AND opt_expiry = ?
          AND (CAST(ts AS TIME) < ? OR CAST(ts AS TIME) > ?)
    """, [
        trade_date_iso,
        pd.to_datetime(opt_expiry_iso).date(),
        MARKET_OPEN.strftime("%H:%M:%S"),
        MARKET_CLOSE.strftime("%H:%M:%S"),
    ])
    con.close()

def latest_available_trade_date(opt_expiry_iso: str) -> str | None:
    con = duckdb.connect(DB_PATH)
    q = """
        SELECT MAX(trade_date) AS d
        FROM oi_snapshots
        WHERE opt_expiry = ?
    """
    d = con.execute(q, [opt_expiry_iso]).fetchone()[0]
    con.close()
    return None if d is None else str(d)

# -------------------- UI Controls --------------------
st.sidebar.header("Controls")
auto_refresh = st.sidebar.toggle("Auto refresh every 60s", value=True)
refresh_now = st.sidebar.button("Refresh now")
recenter_atm = st.sidebar.button("Recenter ATM now")
st.sidebar.caption("Stores snapshots locally in oi_snapshots.duckdb")

init_db()

try:
    now_ts = now_ist()
    trade_date = now_ts.date().isoformat()
    market_live = True #is_market_open(now_ts)

    inst_df = load_instruments()
    opt_expiry = pick_nearest_weekly_nifty_opt_expiry(inst_df)
    opt_expiry_date = pd.to_datetime(opt_expiry).date()

    if market_live:
        # ---------------- LIVE MODE ----------------
        fut_token, fut_symbol, fut_expiry = pick_nearest_nifty_fut(inst_df)
        fut_ltp = float(kite.ltp([f"NFO:{fut_symbol}"])[f"NFO:{fut_symbol}"]["last_price"])

        spot_symbol = "NSE:NIFTY 50"
        spot_ltp = float(kite.ltp([spot_symbol])[spot_symbol]["last_price"])
        
        # ATM (freeze outside market hours happens naturally because this block runs only when live)
        atm_candidate = round_to_50(spot_ltp)

        if "stable_atm" not in st.session_state:
            st.session_state["stable_atm"] = atm_candidate
            st.session_state["atm_candidate_prev"] = atm_candidate
            st.session_state["atm_candidate_streak"] = 1

        # manual recenter
        if recenter_atm:
            st.session_state["stable_atm"] = atm_candidate
            st.session_state["atm_candidate_prev"] = atm_candidate
            st.session_state["atm_candidate_streak"] = 1

        # stability update (2 consecutive refreshes)
        prev = st.session_state["atm_candidate_prev"]
        if atm_candidate == prev:
            st.session_state["atm_candidate_streak"] += 1
        else:
            st.session_state["atm_candidate_prev"] = atm_candidate
            st.session_state["atm_candidate_streak"] = 1

        if st.session_state["atm_candidate_streak"] >= 2:
            st.session_state["stable_atm"] = st.session_state["atm_candidate_prev"]

        stable_atm = int(st.session_state["stable_atm"])
        strikes = [stable_atm - 100, stable_atm - 50, stable_atm, stable_atm + 50, stable_atm + 100]

        opt_df = resolve_option_instruments(inst_df, opt_expiry, strikes)

        # Fetch quotes (OI + LTP) + insert snapshot once/min
        symbols = [f"NFO:{s}" for s in opt_df["tradingsymbol"].tolist()]
        quotes = kite.quote(symbols)
        now_ts_naive = now_ts.replace(tzinfo=None)
        
        snap_rows = []
        for _, r in opt_df.iterrows():
            sym = f"NFO:{r['tradingsymbol']}"
            q = quotes.get(sym, {})
            oi = q.get("oi", None)
            last_price = q.get("last_price", None)
            volume = q.get("volume", None)
            snap_rows.append({
                "ts": now_ts_naive,
                "trade_date": trade_date,
                "fut_symbol": fut_symbol,
                "fut_ltp": fut_ltp,
                "opt_expiry": opt_expiry_date,
                "strike": int(r["strike"]),
                "opt_right": str(r["instrument_type"]),
                "tradingsymbol": str(r["tradingsymbol"]),
                "instrument_token": int(r["instrument_token"]),
                "oi": None if oi is None else int(oi),
                "ltp": None if last_price is None else float(last_price),
                "volume": None if volume is None else int(volume),
            })

        snap_df = pd.DataFrame(snap_rows)
        if should_insert_snapshot(now_ts):
            insert_snapshot(snap_df)

    else:
        # ---------------- READ-ONLY MODE ----------------
        st.warning("Market closed (9:15–15:30 IST). Read-only mode: showing last stored snapshot.")

        latest_snap = load_latest_snapshot(trade_date, opt_expiry)

        if latest_snap.empty:
            fallback_date = latest_available_trade_date(opt_expiry)
            if fallback_date is None:
                st.error("No stored snapshots found in DB at all. Run during market hours to collect data.")
                st.stop()

            st.info(f"No snapshots for today. Showing last available day: {fallback_date}")
            trade_date = fallback_date
            latest_snap = load_latest_snapshot(trade_date, opt_expiry)

            if latest_snap.empty:
                st.error("DB has data but could not load latest snapshot. (Unexpected) ")
                st.stop()

        # Use DB values (do NOT fetch live)
        fut_symbol = str(latest_snap["fut_symbol"].iloc[0])
        fut_ltp = float(latest_snap["fut_ltp"].iloc[0])
        spot_symbol = "NSE:NIFTY 50"
        spot_ltp = fut_ltp

        strikes = sorted(latest_snap["strike"].unique().tolist())
        if len(strikes) >= 5:
            stable_atm = int(strikes[len(strikes)//2])
        else:
            stable_atm = int(strikes[0])
        atm_candidate = stable_atm  # frozen

        # Build opt_df from stored snapshot rows
        opt_df = latest_snap[["tradingsymbol", "instrument_token", "strike", "opt_right"]].copy()
        opt_df.rename(columns={"opt_right": "instrument_type"}, inplace=True)

    # Load today history + deltas
    hist = load_day(trade_date, opt_expiry)
    hist = compute_deltas(hist)
    # Yesterday close baseline (safe: always produce columns)
    ybase = load_yday_close(trade_date, opt_expiry)

    if not ybase.empty:
        hist = hist.merge(ybase, on="tradingsymbol", how="left")
    else:
        if "oi_yday_close" not in hist.columns:
            hist["oi_yday_close"] = pd.NA

    # Always compute d_oi_vs_yday (even if NaN)
    hist["d_oi_vs_yday"] = (
        pd.to_numeric(hist["oi"], errors="coerce")
        - pd.to_numeric(hist["oi_yday_close"], errors="coerce")
    )
    # ---- FILTER DISPLAY TO MARKET SESSION ONLY (IST) ----
    # --- Force ts to IST and filter to market session (09:15–15:30 IST) ---
    hist["ts"] = pd.to_datetime(hist["ts"], errors="coerce")

    # If DuckDB returned tz-naive timestamps, treat them as IST (since we store IST in app)
    if getattr(hist["ts"].dt, "tz", None) is None:
        hist["ts_ist"] = hist["ts"].dt.tz_localize(IST)
    else:
        hist["ts_ist"] = hist["ts"].dt.tz_convert(IST)

    session_start = datetime.combine(pd.to_datetime(trade_date).date(), MARKET_OPEN).replace(tzinfo=IST)
    session_end   = datetime.combine(pd.to_datetime(trade_date).date(), MARKET_CLOSE).replace(tzinfo=IST)

    hist = hist[(hist["ts_ist"] >= session_start) & (hist["ts_ist"] <= session_end)].copy()

    # Use ts_ist for charts / sorting, but keep original ts for DB integrity
    hist = hist.sort_values("ts_ist")

    latest = hist.sort_values("ts_ist").groupby("tradingsymbol").tail(1)
    latest = add_impulse_tags(latest, hist, lookback_minutes=30)

    # Build strike board
    # Build strike board (safe)
    ce = (
        latest[latest["opt_right"] == "CE"]
        .reindex(columns=["strike", "oi", "d_oi_1m", "d_oi_5m", "d_oi_vs_yday", "impulse"])
        .rename(
            columns={
                "oi": "CE_OI",
                "d_oi_1m": "CE_Δ1m",
                "d_oi_5m": "CE_Δ5m",
                "d_oi_vs_yday": "CE_ΔvsYday",
                "impulse": "CE_tag",
            }
        )
    )

    pe = (
        latest[latest["opt_right"] == "PE"]
        .reindex(columns=["strike", "oi", "d_oi_1m", "d_oi_5m", "d_oi_vs_yday", "impulse"])
        .rename(
            columns={
                "oi": "PE_OI",
                "d_oi_1m": "PE_Δ1m",
                "d_oi_5m": "PE_Δ5m",
                "d_oi_vs_yday": "PE_ΔvsYday",
                "impulse": "PE_tag",
            }
        )
    )

    board = pd.merge(ce, pe, on="strike", how="outer").sort_values("strike")

    # Summary metrics
    sum_ce_oi = float(pd.to_numeric(board["CE_OI"], errors="coerce").fillna(0).sum())
    sum_pe_oi = float(pd.to_numeric(board["PE_OI"], errors="coerce").fillna(0).sum())
    sum_ce_d5 = float(pd.to_numeric(board["CE_Δ5m"], errors="coerce").fillna(0).sum())
    sum_pe_d5 = float(pd.to_numeric(board["PE_Δ5m"], errors="coerce").fillna(0).sum())

    pcr = (sum_pe_oi / sum_ce_oi) if sum_ce_oi > 0 else float("nan")
    net_d5 = sum_pe_d5 - sum_ce_d5

    # Shared numeric fields
    hist["d_oi_1m_num"] = pd.to_numeric(hist["d_oi_1m"], errors="coerce").fillna(0.0)
    hist["d_oi_5m_num"] = pd.to_numeric(hist["d_oi_5m"], errors="coerce").fillna(0.0)
    hist["oi_num"] = pd.to_numeric(hist["oi"], errors="coerce").fillna(0.0)
    hist["fut_ltp_num"] = pd.to_numeric(hist["fut_ltp"], errors="coerce")
    hist["volume_num"] = pd.to_numeric(hist.get("volume", pd.Series(index=hist.index, dtype=float)), errors="coerce").fillna(0.0)
    hist["signed_d_oi_1m"] = np.where(hist["opt_right"] == "PE", 1.0, -1.0) * hist["d_oi_1m_num"]

    minute = (
        hist.groupby("ts_ist")
        .agg(
            fut_ltp=("fut_ltp_num", "mean"),
            total_oi=("oi_num", "sum"),
            net_oi_1m=("signed_d_oi_1m", "sum"),
            opt_volume=("volume_num", "sum"),
        )
        .sort_index()
    )
    minute["d_price_1m"] = minute["fut_ltp"].diff(1)
    minute["d_price_5m"] = minute["fut_ltp"].diff(5)
    minute["d_total_oi_1m"] = minute["total_oi"].diff(1)
    minute["d_total_oi_5m"] = minute["total_oi"].diff(5)
    minute["net_oi_5m"] = minute["net_oi_1m"].rolling(5).sum()
    minute["net_oi_15m"] = minute["net_oi_1m"].rolling(15).sum()
    minute["vwap_proxy"] = (
        (minute["fut_ltp"] * minute["opt_volume"]).cumsum()
        / minute["opt_volume"].replace(0.0, np.nan).cumsum()
    )
    minute["vwap_proxy"] = minute["vwap_proxy"].fillna(minute["fut_ltp"].expanding().mean())

    # ---------------- Volatility context (Point 3) ----------------
    t_exp = time_to_expiry_years(opt_expiry, now_ts)
    atm_rows = latest[latest["strike"] == stable_atm].copy()
    atm_ce = atm_rows[atm_rows["opt_right"] == "CE"]
    atm_pe = atm_rows[atm_rows["opt_right"] == "PE"]

    ce_iv = None
    pe_iv = None
    if not atm_ce.empty:
        ce_ltp = float(pd.to_numeric(atm_ce["ltp"], errors="coerce").iloc[0])
        ce_iv = implied_volatility(ce_ltp, float(spot_ltp), float(stable_atm), t_exp, "CE")
    if not atm_pe.empty:
        pe_ltp = float(pd.to_numeric(atm_pe["ltp"], errors="coerce").iloc[0])
        pe_iv = implied_volatility(pe_ltp, float(spot_ltp), float(stable_atm), t_exp, "PE")

    iv_vals = [x for x in [ce_iv, pe_iv] if x is not None]
    atm_iv = (sum(iv_vals) / len(iv_vals)) if iv_vals else None

    # Build intraday ATM IV series for a simple LOW/NORMAL/HIGH label.
    hist_atm = hist[hist["strike"] == stable_atm].copy()
    if not hist_atm.empty:
        hist_atm["ltp_num"] = pd.to_numeric(hist_atm["ltp"], errors="coerce")
        hist_atm["fut_ltp_num"] = pd.to_numeric(hist_atm["fut_ltp"], errors="coerce")
        t_exp_hist = time_to_expiry_years(opt_expiry, now_ts)

        def _iv_row(row):
            px = row["ltp_num"]
            s0 = row["fut_ltp_num"]
            if pd.isna(px) or pd.isna(s0):
                return np.nan
            iv = implied_volatility(float(px), float(s0), float(stable_atm), t_exp_hist, str(row["opt_right"]))
            return np.nan if iv is None else iv

        hist_atm["iv"] = hist_atm.apply(_iv_row, axis=1)
        iv_series = hist_atm.groupby("ts_ist")["iv"].mean()
    else:
        iv_series = pd.Series(dtype=float)
    vol_state = infer_vol_state(atm_iv, iv_series)

    # ---------------- Point 1: Regime filter ----------------
    p5 = minute["fut_ltp"].resample("5min").ohlc().dropna()
    adx_5m = compute_adx(p5.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close"}), period=14)
    adx_last = float(adx_5m.dropna().iloc[-1]) if not adx_5m.dropna().empty else float("nan")
    p15 = minute["fut_ltp"].resample("15min").ohlc().dropna()
    adx_15m = compute_adx(p15.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close"}), period=14)
    adx15_last = float(adx_15m.dropna().iloc[-1]) if not adx_15m.dropna().empty else float("nan")
    latest_price = float(minute["fut_ltp"].iloc[-1]) if not minute.empty else float("nan")
    latest_vwap = float(minute["vwap_proxy"].iloc[-1]) if not minute.empty else float("nan")
    or_window_end = datetime.combine(pd.to_datetime(trade_date).date(), dtime(9, 30)).replace(tzinfo=IST)
    or_slice = minute.loc[minute.index <= or_window_end]
    if not or_slice.empty:
        or_high = float(or_slice["fut_ltp"].max())
        or_low = float(or_slice["fut_ltp"].min())
    else:
        or_high = float("nan")
        or_low = float("nan")

    adx_thr = 20.0
    above_or = math.isfinite(or_high) and latest_price > or_high
    below_or = math.isfinite(or_low) and latest_price < or_low
    if (
        math.isfinite(latest_price)
        and math.isfinite(latest_vwap)
        and math.isfinite(adx_last)
        and adx_last > adx_thr
        and latest_price > latest_vwap
        and above_or
    ):
        regime = "TREND_UP"
    elif (
        math.isfinite(latest_price)
        and math.isfinite(latest_vwap)
        and math.isfinite(adx_last)
        and adx_last > adx_thr
        and latest_price < latest_vwap
        and below_or
    ):
        regime = "TREND_DOWN"
    else:
        regime = "CHOP"

    # ---------------- Point 2: Price + OI quadrant ----------------
    dpx1 = float(minute["d_price_1m"].iloc[-1]) if not minute.empty else float("nan")
    doi1 = float(minute["d_total_oi_1m"].iloc[-1]) if not minute.empty else float("nan")
    dpx5 = float(minute["d_price_5m"].iloc[-1]) if not minute.empty else float("nan")
    doi5 = float(minute["d_total_oi_5m"].iloc[-1]) if not minute.empty else float("nan")
    quad_1m = classify_quadrant(dpx1, doi1)
    quad_5m = classify_quadrant(dpx5, doi5)

    # ---------------- Point 4: Flow quality ----------------
    latest_view = hist.sort_values("ts_ist").groupby("tradingsymbol").tail(1).copy()
    prev5_cutoff = latest_view["ts_ist"].max() - timedelta(minutes=5) if not latest_view.empty else None
    if prev5_cutoff is not None:
        prev5_view = (
            hist[hist["ts_ist"] <= prev5_cutoff]
            .sort_values("ts_ist")
            .groupby("tradingsymbol")
            .tail(1)
            .copy()
        )
    else:
        prev5_view = pd.DataFrame()

    latest_view["volume_num"] = pd.to_numeric(latest_view.get("volume", 0), errors="coerce").fillna(0.0)
    ce_vol = float(latest_view[latest_view["opt_right"] == "CE"]["volume_num"].sum())
    pe_vol = float(latest_view[latest_view["opt_right"] == "PE"]["volume_num"].sum())
    pcvr = (pe_vol / ce_vol) if ce_vol > 0 else float("nan")

    ce_top2_now = top2_oi_share(latest_view, "CE")
    pe_top2_now = top2_oi_share(latest_view, "PE")
    ce_top2_prev = top2_oi_share(prev5_view, "CE")
    pe_top2_prev = top2_oi_share(prev5_view, "PE")
    ce_conc_shift = ce_top2_now - ce_top2_prev if (ce_top2_now == ce_top2_now and ce_top2_prev == ce_top2_prev) else float("nan")
    pe_conc_shift = pe_top2_now - pe_top2_prev if (pe_top2_now == pe_top2_now and pe_top2_prev == pe_top2_prev) else float("nan")

    latest_view["d_oi_5m_num"] = pd.to_numeric(latest_view["d_oi_5m"], errors="coerce").fillna(0.0)
    latest_view["ltp_num"] = pd.to_numeric(latest_view["ltp"], errors="coerce")
    latest_view["ltp_prev_5_num"] = pd.to_numeric(latest_view["ltp_prev_5"], errors="coerce")
    writing = latest_view[(latest_view["d_oi_5m_num"] > 0) & (latest_view["ltp_num"] <= latest_view["ltp_prev_5_num"])].copy()
    ce_wall, ce_wall_oi = find_wall(writing, "CE")
    pe_wall, pe_wall_oi = find_wall(writing, "PE")

    if not prev5_view.empty:
        prev5_walls = prev5_view.copy()
        if "d_oi_5m" in prev5_walls.columns:
            prev5_walls["d_oi_5m_num"] = pd.to_numeric(prev5_walls["d_oi_5m"], errors="coerce").fillna(0.0)
        else:
            prev5_walls["d_oi_5m_num"] = 0.0
        prev_ce_wall, _ = find_wall(prev5_walls, "CE")
        prev_pe_wall, _ = find_wall(prev5_walls, "PE")
    else:
        prev_ce_wall, prev_pe_wall = None, None
    ce_wall_shift = shift_label(ce_wall, prev_ce_wall)
    pe_wall_shift = shift_label(pe_wall, prev_pe_wall)

    # ---------------- Point 5: MTF net-bias confirmation ----------------
    net1 = float(minute["net_oi_1m"].iloc[-1]) if not minute.empty else float("nan")
    net5 = float(minute["net_oi_5m"].iloc[-1]) if not minute.empty else float("nan")
    net15 = float(minute["net_oi_15m"].iloc[-1]) if not minute.empty else float("nan")
    mtf_alignment = mtf_alignment_label(net1, net5, net15)

    spot_name = "NIFTY 50" if str(spot_symbol).startswith("NSE:NIFTY") else str(spot_symbol)
    quadrant_disp = f"{pretty_enum(quad_1m)} / {pretty_enum(quad_5m)}"
    mtf_disp = pretty_enum(mtf_alignment)
    vol_disp = pretty_enum(vol_state)

    r1c1, r1c2, r1c3 = st.columns(3)
    r1c1.metric("Spot", f"{spot_ltp:.2f}")
    r1c2.metric("ATM", str(stable_atm))
    r1c3.metric("PCR", f"{pcr:.3f}" if pcr == pcr else "—")

    r2c1, r2c2, r2c3 = st.columns(3)
    r2c1.metric("Net OI 5m", human_compact(net_d5))
    r2c2.metric("ATM IV", f"{atm_iv * 100:.2f}%" if atm_iv is not None else "—")
    r2c3.metric("Vol State", vol_disp)

    r3c1, r3c2, r3c3 = st.columns(3)
    r3c1.metric("Regime", pretty_enum(regime))
    r3c2.metric("Quadrant (1m/5m)", quadrant_disp)
    r3c3.metric("MTF Align", mtf_disp)

    r4c1, r4c2, r4c3 = st.columns(3)
    r4c1.metric("PCVR", f"{pcvr:.3f}" if pcvr == pcvr else "—")
    adx_5m_txt = f"{adx_last:.1f}" if adx_last == adx_last else "—"
    adx_15m_txt = f"{adx15_last:.1f}" if adx15_last == adx15_last else "—"
    r4c2.metric("ADX 5m/15m", f"{adx_5m_txt} / {adx_15m_txt}")
    r4c3.metric("Symbol", spot_name)

    st.subheader("Live OI Board (Stable ATM ±100)")
    st.dataframe(board, use_container_width=True, hide_index=True)

    tab1, tab2, tab3 = st.tabs(["Charts", "Raw (latest)", "Debug"])

    with tab1:
        st.caption("Charts are from stored snapshots today.")
        ce_ts = hist[hist["opt_right"] == "CE"].copy()
        pe_ts = hist[hist["opt_right"] == "PE"].copy()

        if not ce_ts.empty:
            ce_piv = ce_ts.pivot_table(index="ts_ist", columns="strike", values="oi", aggfunc="last").sort_index()
            st.write("CE OI trend")
            st.line_chart(ce_piv)

        if not pe_ts.empty:
            pe_piv = pe_ts.pivot_table(index="ts_ist", columns="strike", values="oi", aggfunc="last").sort_index()
            st.write("PE OI trend")
            st.line_chart(pe_piv)

        # Net bias line (PEΔ1m - CEΔ1m aggregated)
        net_line = minute["net_oi_1m"].sort_index()
        st.write("Net ΔOI 1m (PE−CE) trend")
        st.line_chart(net_line)

        if not iv_series.empty:
            st.write("ATM IV trend")
            st.line_chart(iv_series)

    with tab2:
        st.dataframe(latest.sort_values(["strike", "opt_right"]), use_container_width=True, hide_index=True)

    with tab3:
        st.write("ATM candidate:", atm_candidate)
        st.write("ATM streak:", st.session_state.get("atm_candidate_streak"))
        st.write("Last inserted minute key:", st.session_state.get("last_insert_minute_key"))
        st.write("Rows today (for expiry):", len(hist))
        st.write("ATM CE IV:", f"{ce_iv * 100:.2f}%" if ce_iv is not None else "—")
        st.write("ATM PE IV:", f"{pe_iv * 100:.2f}%" if pe_iv is not None else "—")
        st.write("ATM IV:", f"{atm_iv * 100:.2f}%" if atm_iv is not None else "—")
        st.write("Vol regime:", vol_state)
        st.write("OR high / low:", f"{or_high:.2f}" if or_high == or_high else "—", "/", f"{or_low:.2f}" if or_low == or_low else "—")
        st.write("VWAP proxy:", f"{latest_vwap:.2f}" if latest_vwap == latest_vwap else "—")
        st.write("Quadrants 1m/5m:", quad_1m, "/", quad_5m)
        st.write("PCVR:", f"{pcvr:.3f}" if pcvr == pcvr else "—")
        st.write("CE top2 OI share shift:", f"{ce_conc_shift:+.3f}" if ce_conc_shift == ce_conc_shift else "—")
        st.write("PE top2 OI share shift:", f"{pe_conc_shift:+.3f}" if pe_conc_shift == pe_conc_shift else "—")
        st.write("CE wall (ΔOI5m):", ce_wall if ce_wall is not None else "—", f"{ce_wall_oi:,.0f}" if ce_wall_oi == ce_wall_oi else "—", ce_wall_shift)
        st.write("PE wall (ΔOI5m):", pe_wall if pe_wall is not None else "—", f"{pe_wall_oi:,.0f}" if pe_wall_oi == pe_wall_oi else "—", pe_wall_shift)
        st.write("MTF net bias 1m/5m/15m:", f"{net1:,.0f}" if net1 == net1 else "—", "/", f"{net5:,.0f}" if net5 == net5 else "—", "/", f"{net15:,.0f}" if net15 == net15 else "—")
        st.write("MTF alignment:", mtf_alignment)
        # if st.button("Clean today after-hours rows"):
        #     delete_outside_session(trade_date, opt_expiry)
        #     st.success("Deleted rows outside 09:15–15:30 for today.")
        #     st.rerun()

    data_asof = hist["ts_ist"].max() if (not hist.empty) else now_ts
    st.caption(f"Data as of: {data_asof} | Market live: {market_live} | DB: {DB_PATH}")

    # Proper 60s cadence
    if auto_refresh and (not refresh_now) and market_live:
        sec = now_ts.second
        time.sleep(max(1, 60 - sec))
        st.rerun()

except Exception as e:
    st.error("Error in Step 5.")
    st.exception(e)
