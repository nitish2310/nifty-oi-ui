import os
from datetime import datetime, date, timedelta, time as dtime
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

    top1, top2, top3, top4, top5, top6 = st.columns(6)
    top1.metric("Nearest NIFTY SPOT", spot_symbol)
    top2.metric("SPOT LTP", f"{spot_ltp:.2f}")
    top3.metric("Stable ATM", str(stable_atm))
    top4.metric("PCR (tracked)", f"{pcr:.3f}" if pcr == pcr else "—")
    top5.metric("Net ΔOI 5m (PE−CE)", f"{net_d5:,.0f}")

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
        tmp = hist.copy()
        tmp["d_oi_1m_num"] = pd.to_numeric(tmp["d_oi_1m"], errors="coerce").fillna(0)
        tmp["sign"] = np.where(tmp["opt_right"] == "PE", 1, -1)
        tmp["net_component"] = tmp["sign"] * tmp["d_oi_1m_num"]
        net_line = tmp.groupby("ts_ist")["net_component"].sum().sort_index()
        st.write("Net ΔOI 1m (PE−CE) trend")
        st.line_chart(net_line)

    with tab2:
        st.dataframe(latest.sort_values(["strike", "opt_right"]), use_container_width=True, hide_index=True)

    with tab3:
        st.write("ATM candidate:", atm_candidate)
        st.write("ATM streak:", st.session_state.get("atm_candidate_streak"))
        st.write("Last inserted minute key:", st.session_state.get("last_insert_minute_key"))
        st.write("Rows today (for expiry):", len(hist))
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
