from datetime import datetime, time as dtime, timedelta

import duckdb
import pandas as pd


class SnapshotRepository:
    def __init__(self, db_path: str, market_open: dtime, market_close: dtime):
        self.db_path = db_path
        self.market_open = market_open
        self.market_close = market_close

    def init_db(self) -> None:
        con = duckdb.connect(self.db_path)
        con.execute(
            """
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
            """
        )
        cols = {row[1] for row in con.execute("PRAGMA table_info('oi_snapshots')").fetchall()}
        if "volume" not in cols:
            con.execute("ALTER TABLE oi_snapshots ADD COLUMN volume BIGINT")
        con.close()

    def insert_snapshot(self, rows: pd.DataFrame) -> None:
        con = duckdb.connect(self.db_path)
        con.register("df_rows", rows)
        con.execute("INSERT INTO oi_snapshots SELECT * FROM df_rows")
        con.close()

    def load_day(self, trade_date_iso: str, opt_expiry_iso: str) -> pd.DataFrame:
        con = duckdb.connect(self.db_path)
        q = """
            SELECT *
            FROM oi_snapshots
            WHERE trade_date = ? AND opt_expiry = ?
            ORDER BY ts ASC
        """
        df = con.execute(q, [trade_date_iso, opt_expiry_iso]).fetchdf()
        con.close()
        return df

    def load_yday_close(self, trade_date_iso: str, opt_expiry_iso: str) -> pd.DataFrame:
        yday = (pd.to_datetime(trade_date_iso).date() - timedelta(days=1)).isoformat()
        con = duckdb.connect(self.db_path)
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
        cutoff = datetime.combine(pd.to_datetime(yday).date(), dtime(15, 25))
        after = df[df["ts"] >= cutoff]
        if not after.empty:
            last = after.groupby("tradingsymbol").tail(1)
        else:
            last = df.groupby("tradingsymbol").tail(1)
        return last[["tradingsymbol", "oi"]].rename(columns={"oi": "oi_yday_close"})

    def load_latest_snapshot(self, trade_date_iso: str, opt_expiry_iso: str) -> pd.DataFrame:
        con = duckdb.connect(self.db_path)
        q = """
            SELECT *
            FROM oi_snapshots
            WHERE trade_date = ? AND opt_expiry = ?
            QUALIFY ROW_NUMBER() OVER (PARTITION BY tradingsymbol ORDER BY ts DESC) = 1
        """
        df = con.execute(q, [trade_date_iso, opt_expiry_iso]).fetchdf()
        con.close()
        return df

    def delete_outside_session(self, trade_date_iso: str, opt_expiry_iso: str) -> None:
        con = duckdb.connect(self.db_path)
        con.execute(
            """
            DELETE FROM oi_snapshots
            WHERE trade_date = ? AND opt_expiry = ?
              AND (CAST(ts AS TIME) < ? OR CAST(ts AS TIME) > ?)
            """,
            [
                trade_date_iso,
                pd.to_datetime(opt_expiry_iso).date(),
                self.market_open.strftime("%H:%M:%S"),
                self.market_close.strftime("%H:%M:%S"),
            ],
        )
        con.close()

    def latest_available_trade_date(self, opt_expiry_iso: str) -> str | None:
        con = duckdb.connect(self.db_path)
        q = """
            SELECT MAX(trade_date) AS d
            FROM oi_snapshots
            WHERE opt_expiry = ?
        """
        d = con.execute(q, [opt_expiry_iso]).fetchone()[0]
        con.close()
        return None if d is None else str(d)

    def list_opt_expiries(self) -> list[str]:
        con = duckdb.connect(self.db_path)
        q = """
            SELECT DISTINCT CAST(opt_expiry AS VARCHAR) AS expiry
            FROM oi_snapshots
            ORDER BY expiry DESC
        """
        rows = con.execute(q).fetchall()
        con.close()
        return [r[0] for r in rows if r[0] is not None]

    def list_trade_dates(self, opt_expiry_iso: str | None = None) -> list[str]:
        con = duckdb.connect(self.db_path)
        if opt_expiry_iso:
            q = """
                SELECT DISTINCT CAST(trade_date AS VARCHAR) AS d
                FROM oi_snapshots
                WHERE opt_expiry = ?
                ORDER BY d DESC
            """
            rows = con.execute(q, [opt_expiry_iso]).fetchall()
        else:
            q = """
                SELECT DISTINCT CAST(trade_date AS VARCHAR) AS d
                FROM oi_snapshots
                ORDER BY d DESC
            """
            rows = con.execute(q).fetchall()
        con.close()
        return [r[0] for r in rows if r[0] is not None]

    def load_filtered(
        self,
        trade_date_iso: str,
        opt_expiry_iso: str,
        rights: list[str] | None = None,
        strikes: list[int] | None = None,
        limit: int = 5000,
    ) -> pd.DataFrame:
        con = duckdb.connect(self.db_path)
        rights = rights or []
        strikes = strikes or []
        q = """
            SELECT ts, trade_date, opt_expiry, strike, opt_right, tradingsymbol, oi, ltp, volume, fut_ltp
            FROM oi_snapshots
            WHERE trade_date = ? AND opt_expiry = ?
        """
        params: list = [trade_date_iso, opt_expiry_iso]
        if rights:
            q += " AND opt_right IN ({})".format(",".join(["?"] * len(rights)))
            params.extend(rights)
        if strikes:
            q += " AND strike IN ({})".format(",".join(["?"] * len(strikes)))
            params.extend(strikes)
        q += " ORDER BY ts DESC LIMIT ?"
        params.append(int(limit))
        df = con.execute(q, params).fetchdf()
        con.close()
        return df
