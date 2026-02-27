from datetime import time as dtime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)
SPOT_SYMBOL = "NSE:NIFTY 50"
DB_PATH = "oi_snapshots.duckdb"

