from datetime import datetime

from nifty_oi.config import IST, MARKET_CLOSE, MARKET_OPEN


class MarketClock:
    @staticmethod
    def now_ist() -> datetime:
        return datetime.now(IST)

    @staticmethod
    def is_market_open(dt_ist: datetime) -> bool:
        t = dt_ist.time()
        return (t >= MARKET_OPEN) and (t <= MARKET_CLOSE)

