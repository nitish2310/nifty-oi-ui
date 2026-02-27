# Top Metrics Guide

This file explains the top metrics shown in the Streamlit dashboard and how to read them quickly.

## Metric Meanings

### Spot
- Current NIFTY spot price.
- Gives immediate market level context.

### ATM
- Current stable at-the-money strike used for option tracking.
- OI and IV stats are centered around this strike window.

### PCR
- Put OI / Call OI for tracked strikes.
- `> 1` means relatively higher put OI, `< 1` means relatively higher call OI.
- Use with trend context; not a standalone directional signal.

### Net OI 5m
- 5-minute aggregate of: `PE OI change - CE OI change`.
- Positive: put-side OI build dominates (often bullish/defensive context).
- Negative: call-side OI build dominates (often bearish/overhead supply context).

### ATM IV
- Implied volatility estimated from ATM CE/PE prices using Black-Scholes inversion.
- Higher IV implies higher expected move and richer premium.

### Vol State
- Relative IV regime from today’s ATM IV distribution: `Low`, `Normal`, `High`.
- `High`: faster moves and possible whipsaws.
- `Low`: breakouts may need stronger confirmation.

### Regime
- Market structure filter: `Trend Up`, `Trend Down`, `Chop`.
- Built from price vs VWAP proxy, opening range break, and ADX.
- `Chop` generally means lower directional signal quality.

### Quadrant (1m/5m)
- Price-OI behavior class on 1-minute and 5-minute windows:
  - `Long Build`: price up + OI up
  - `Short Build`: price down + OI up
  - `Short Cover`: price up + OI down
  - `Long Unwind`: price down + OI down
  - `Flat/Mixed`: weak or unclear move
- 1m is fast pulse; 5m is stronger confirmation.

### MTF Align
- Multi-timeframe net OI bias agreement (1m, 5m, 15m):
  - `Bullish/Bearish Strong`: all timeframes aligned
  - `Bullish/Bearish Weak`: 1m+5m aligned, 15m not opposing
  - `Mixed/No-Trade`: timeframe conflict

### PCVR
- Put volume / Call volume ratio across tracked strikes.
- Higher PCVR means relatively heavier put-side flow.
- Best used as context, not standalone.

### ADX 5m
- Trend strength (not direction) on 5-minute price bars.
- Higher ADX usually means stronger trend structure.
- Lower ADX usually means range/chop conditions.

### Symbol
- Underlying symbol shown for context.

## Quick Reading Order

1. `Regime` and `ADX 5m` (trend vs chop quality)
2. `MTF Align` (timeframe agreement check)
3. `Quadrant (1m/5m)` (current build/unwind behavior)
4. `Net OI 5m` with `PCR/PCVR` (flow confirmation)
5. `ATM IV` and `Vol State` (volatility/risk environment)
