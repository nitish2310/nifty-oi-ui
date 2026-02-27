import os
import time
from datetime import datetime, timedelta, time as dtime

import numpy as np
import pandas as pd
import streamlit as st
import math
from dotenv import load_dotenv
from kiteconnect import KiteConnect

from nifty_oi.analytics.market_analytics import MarketAnalytics
from nifty_oi.config import DB_PATH, IST, MARKET_CLOSE, MARKET_OPEN, SPOT_SYMBOL
from nifty_oi.market_clock import MarketClock
from nifty_oi.services.kite_data_service import KiteDataService
from nifty_oi.storage.snapshot_repository import SnapshotRepository

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

analytics = MarketAnalytics()
data_service = KiteDataService(kite)
snapshot_repo = SnapshotRepository(DB_PATH, MARKET_OPEN, MARKET_CLOSE)


@st.cache_data(ttl=3600)
def load_instruments():
    return data_service.load_instruments()


now_ist = MarketClock.now_ist
is_market_open = MarketClock.is_market_open
round_to_50 = analytics.round_to_50
time_to_expiry_years = analytics.time_to_expiry_years
infer_vol_state = analytics.infer_vol_state
compute_adx = analytics.compute_adx
classify_quadrant = analytics.classify_quadrant
mtf_alignment_label = analytics.mtf_alignment_label
top2_oi_share = analytics.top2_oi_share
find_wall = analytics.find_wall
shift_label = analytics.shift_label
human_compact = analytics.human_compact
pretty_enum = analytics.pretty_enum
implied_volatility = analytics.implied_volatility
compute_deltas = analytics.compute_deltas
add_impulse_tags = analytics.add_impulse_tags
pick_nearest_nifty_fut = data_service.pick_nearest_nifty_fut
pick_nearest_weekly_nifty_opt_expiry = data_service.pick_nearest_weekly_nifty_opt_expiry
resolve_option_instruments = data_service.resolve_option_instruments
init_db = snapshot_repo.init_db
insert_snapshot = snapshot_repo.insert_snapshot
load_day = snapshot_repo.load_day
load_yday_close = snapshot_repo.load_yday_close
load_latest_snapshot = snapshot_repo.load_latest_snapshot
delete_outside_session = snapshot_repo.delete_outside_session
latest_available_trade_date = snapshot_repo.latest_available_trade_date
list_opt_expiries = snapshot_repo.list_opt_expiries
list_trade_dates = snapshot_repo.list_trade_dates
load_filtered = snapshot_repo.load_filtered


def should_insert_snapshot(now_ts: datetime) -> bool:
    minute_key = now_ts.strftime("%Y-%m-%d %H:%M")
    last_key = st.session_state.get("last_insert_minute_key")
    if last_key == minute_key:
        return False
    st.session_state["last_insert_minute_key"] = minute_key
    return True

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
    market_live = is_market_open(now_ts)

    inst_df = load_instruments()
    opt_expiry = pick_nearest_weekly_nifty_opt_expiry(inst_df)
    opt_expiry_date = pd.to_datetime(opt_expiry).date()

    if market_live:
        # ---------------- LIVE MODE ----------------
        fut_token, fut_symbol, fut_expiry = pick_nearest_nifty_fut(inst_df)
        fut_ltp = float(data_service.ltp([f"NFO:{fut_symbol}"])[f"NFO:{fut_symbol}"]["last_price"])

        spot_symbol = SPOT_SYMBOL
        spot_ltp = float(data_service.ltp([spot_symbol])[spot_symbol]["last_price"])
        
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
        quotes = data_service.quote(symbols)
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
        spot_symbol = SPOT_SYMBOL
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

    tab1, tab2, tab3, tab4 = st.tabs(["Charts", "Raw (latest)", "Debug", "Data Explorer"])

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

    with tab4:
        st.caption("Explore historical snapshots stored in DuckDB.")
        expiry_options = list_opt_expiries()
        if not expiry_options:
            st.info("No data found in DB yet.")
        else:
            default_expiry_idx = expiry_options.index(opt_expiry) if opt_expiry in expiry_options else 0
            exp_sel = st.selectbox("Option expiry", expiry_options, index=default_expiry_idx, key="explorer_expiry")

            date_options = list_trade_dates(exp_sel)
            if not date_options:
                st.info("No trade dates found for selected expiry.")
            else:
                default_date_idx = date_options.index(trade_date) if trade_date in date_options else 0
                date_sel = st.selectbox("Trade date", date_options, index=default_date_idx, key="explorer_date")

                day_df = load_day(date_sel, exp_sel)
                strike_options = sorted(pd.to_numeric(day_df["strike"], errors="coerce").dropna().astype(int).unique().tolist()) if not day_df.empty else []

                c1, c2, c3 = st.columns([2, 2, 1])
                with c1:
                    rights_sel = st.multiselect("Option right", ["CE", "PE"], default=["CE", "PE"], key="explorer_rights")
                with c2:
                    strike_sel = st.multiselect("Strikes", strike_options, default=strike_options[:5], key="explorer_strikes")
                with c3:
                    limit_sel = st.number_input("Rows", min_value=100, max_value=50000, value=5000, step=100, key="explorer_limit")
                c4, c5 = st.columns([1, 1])
                with c4:
                    include_after_hours = st.toggle("Include after-hours", value=False, key="explorer_after_hours")
                with c5:
                    interval_sel = st.selectbox("View interval", ["1m", "5m", "15m"], index=0, key="explorer_interval")

                qdf = load_filtered(
                    trade_date_iso=date_sel,
                    opt_expiry_iso=exp_sel,
                    rights=rights_sel,
                    strikes=strike_sel,
                    limit=int(limit_sel),
                )

                if qdf.empty:
                    st.warning("No rows for selected filters.")
                else:
                    qdf = qdf.copy()
                    qdf["ts"] = pd.to_datetime(qdf["ts"], errors="coerce")
                    if not include_after_hours:
                        qdf = qdf[
                            (qdf["ts"].dt.time >= MARKET_OPEN)
                            & (qdf["ts"].dt.time <= MARKET_CLOSE)
                        ].copy()

                    freq_map = {"1m": "1min", "5m": "5min", "15m": "15min"}
                    if interval_sel in freq_map:
                        qdf["bucket_ts"] = qdf["ts"].dt.floor(freq_map[interval_sel])
                        qdf = (
                            qdf.sort_values("ts")
                            .groupby(
                                [
                                    "bucket_ts",
                                    "trade_date",
                                    "opt_expiry",
                                    "strike",
                                    "opt_right",
                                    "tradingsymbol",
                                ],
                                as_index=False,
                            )
                            .agg(
                                oi=("oi", "last"),
                                ltp=("ltp", "last"),
                                volume=("volume", "sum"),
                                fut_ltp=("fut_ltp", "last"),
                            )
                            .rename(columns={"bucket_ts": "ts"})
                            .sort_values("ts", ascending=False)
                        )

                    if qdf.empty:
                        st.warning("No rows after session/interval filters.")
                    else:
                        s1, s2, s3, s4 = st.columns(4)
                        s1.metric("Rows", f"{len(qdf):,}")
                        s2.metric("Min TS", str(pd.to_datetime(qdf["ts"]).min()))
                        s3.metric("Max TS", str(pd.to_datetime(qdf["ts"]).max()))
                        s4.metric("Distinct Symbols", f"{qdf['tradingsymbol'].nunique():,}")

                        st.dataframe(qdf, use_container_width=True, hide_index=True)
                        st.download_button(
                            label="Download CSV",
                            data=qdf.to_csv(index=False).encode("utf-8"),
                            file_name=f"oi_snapshots_{date_sel}_{exp_sel}.csv",
                            mime="text/csv",
                        )

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
