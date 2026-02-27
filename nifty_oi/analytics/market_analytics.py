import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from nifty_oi.config import IST, MARKET_CLOSE


class MarketAnalytics:
    @staticmethod
    def round_to_50(x: float) -> int:
        return int(round(x / 50.0) * 50)

    @staticmethod
    def norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @classmethod
    def bs_price(cls, spot: float, strike: float, t_years: float, r: float, vol: float, right: str) -> float:
        if t_years <= 0.0 or vol <= 0.0 or spot <= 0.0 or strike <= 0.0:
            intrinsic = max(spot - strike, 0.0) if right == "CE" else max(strike - spot, 0.0)
            return intrinsic

        sqrt_t = math.sqrt(t_years)
        d1 = (math.log(spot / strike) + (r + 0.5 * vol * vol) * t_years) / (vol * sqrt_t)
        d2 = d1 - vol * sqrt_t
        df = math.exp(-r * t_years)
        if right == "CE":
            return spot * cls.norm_cdf(d1) - strike * df * cls.norm_cdf(d2)
        return strike * df * cls.norm_cdf(-d2) - spot * cls.norm_cdf(-d1)

    @classmethod
    def implied_volatility(
        cls,
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
        low_p = cls.bs_price(spot, strike, t_years, r, low, right)
        high_p = cls.bs_price(spot, strike, t_years, r, high, right)
        if option_price < low_p or option_price > high_p:
            return None

        for _ in range(max_iter):
            mid = 0.5 * (low + high)
            mid_p = cls.bs_price(spot, strike, t_years, r, mid, right)
            err = mid_p - option_price
            if abs(err) <= tol:
                return mid
            if err > 0:
                high = mid
            else:
                low = mid
        return 0.5 * (low + high)

    @staticmethod
    def time_to_expiry_years(expiry_iso: str, now_ts_ist: datetime) -> float:
        expiry_dt = datetime.combine(pd.to_datetime(expiry_iso).date(), MARKET_CLOSE).replace(tzinfo=IST)
        dt_secs = (expiry_dt - now_ts_ist).total_seconds()
        return max(dt_secs / (365.0 * 24.0 * 3600.0), 1e-8)

    @staticmethod
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

    @staticmethod
    def compute_adx(price_df: pd.DataFrame, period: int = 14) -> pd.Series:
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
        return dx.ewm(alpha=1.0 / period, adjust=False).mean()

    @staticmethod
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

    @staticmethod
    def sign_label(x: float, eps: float = 0.0) -> str:
        if not math.isfinite(x):
            return "NA"
        if x > eps:
            return "BULLISH"
        if x < -eps:
            return "BEARISH"
        return "NEUTRAL"

    @classmethod
    def mtf_alignment_label(cls, v1: float, v5: float, v15: float) -> str:
        s1 = cls.sign_label(v1)
        s5 = cls.sign_label(v5)
        s15 = cls.sign_label(v15)
        if s1 == "BULLISH" and s5 == "BULLISH" and s15 == "BULLISH":
            return "BULLISH_STRONG"
        if s1 == "BEARISH" and s5 == "BEARISH" and s15 == "BEARISH":
            return "BEARISH_STRONG"
        if s1 == "BULLISH" and s5 == "BULLISH" and s15 != "BEARISH":
            return "BULLISH_WEAK"
        if s1 == "BEARISH" and s5 == "BEARISH" and s15 != "BULLISH":
            return "BEARISH_WEAK"
        return "MIXED/NO-TRADE"

    @staticmethod
    def top2_oi_share(df: pd.DataFrame, right: str) -> float:
        x = df[df["opt_right"] == right].copy()
        if x.empty:
            return float("nan")
        by_strike = x.groupby("strike")["oi"].sum().sort_values(ascending=False)
        total = float(by_strike.sum())
        if total <= 0:
            return float("nan")
        return float(by_strike.head(2).sum() / total)

    @staticmethod
    def find_wall(snapshot: pd.DataFrame, right: str) -> tuple[int | None, float]:
        x = snapshot[snapshot["opt_right"] == right].copy()
        if x.empty:
            return None, float("nan")
        by_strike = x.groupby("strike")["d_oi_5m_num"].sum().sort_values(ascending=False)
        if by_strike.empty:
            return None, float("nan")
        strike = int(by_strike.index[0])
        return strike, float(by_strike.iloc[0])

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def pretty_enum(s: str) -> str:
        if not s:
            return "—"
        return s.replace("_", " ").replace("-", " ").title()

    @staticmethod
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

    @staticmethod
    def add_impulse_tags(latest: pd.DataFrame, hist: pd.DataFrame, lookback_minutes: int = 30) -> pd.DataFrame:
        if latest.empty or hist.empty:
            latest["impulse"] = ""
            return latest

        cutoff = latest["ts"].max() - timedelta(minutes=lookback_minutes)
        recent = hist[hist["ts"] >= cutoff].copy()
        stats = recent.groupby("tradingsymbol")["d_oi_1m"].agg(["mean", "std"]).reset_index()
        merged = latest.merge(stats, on="tradingsymbol", how="left")
        merged["std"] = merged["std"].fillna(0.0)
        merged["mean"] = merged["mean"].fillna(0.0)
        thr = merged["mean"].abs() + 3.0 * merged["std"]
        merged["impulse"] = np.where(merged["d_oi_1m"].abs() > thr, "IMPULSE", "")
        return merged

