from datetime import date

import pandas as pd


class KiteDataService:
    def __init__(self, kite_client):
        self.kite = kite_client

    def load_instruments(self) -> pd.DataFrame:
        return pd.DataFrame(self.kite.instruments())

    def pick_nearest_nifty_fut(self, inst_df: pd.DataFrame) -> tuple[int, str, str]:
        df = inst_df.copy()
        df = df[(df["exchange"] == "NFO") & (df["segment"] == "NFO-FUT") & (df["name"] == "NIFTY")]
        df["expiry"] = pd.to_datetime(df["expiry"])
        today = pd.Timestamp(date.today())
        df = df[df["expiry"] >= today].sort_values("expiry", ascending=True)
        if df.empty:
            raise RuntimeError("No upcoming NIFTY futures found.")
        row = df.iloc[0]
        return int(row["instrument_token"]), str(row["tradingsymbol"]), row["expiry"].date().isoformat()

    def pick_nearest_weekly_nifty_opt_expiry(self, inst_df: pd.DataFrame) -> str:
        df = inst_df.copy()
        df = df[(df["exchange"] == "NFO") & (df["segment"] == "NFO-OPT") & (df["name"] == "NIFTY")]
        df["expiry"] = pd.to_datetime(df["expiry"])
        today = pd.Timestamp(date.today())
        df = df[df["expiry"] >= today].sort_values("expiry", ascending=True)
        if df.empty:
            raise RuntimeError("No upcoming NIFTY option expiries found.")
        return df.iloc[0]["expiry"].date().isoformat()

    def resolve_option_instruments(self, inst_df: pd.DataFrame, expiry_iso: str, strikes: list[int]) -> pd.DataFrame:
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
        return out.sort_values(["strike", "instrument_type"], ascending=[True, True])

    def ltp(self, symbols: list[str]) -> dict:
        return self.kite.ltp(symbols)

    def quote(self, symbols: list[str]) -> dict:
        return self.kite.quote(symbols)

