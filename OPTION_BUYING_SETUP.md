# Option Buying Setup

Practical checklist for using the dashboard metrics to evaluate CE/PE option buying setups.

## CE Buy (Bullish) Ideal Setup

1. `Regime`: `Trend Up` (avoid `Chop`)
2. `MTF Align`: `Bullish Strong` (or at least `Bullish Weak`)
3. `Quadrant (1m/5m)`: prefer `Long Build` or `Short Cover` on both, or 1m supportive with 5m not opposite
4. `Net OI 5m`: positive and improving
5. `ADX 5m`: rising and typically > 20
6. `Vol State`: `Low` or `Normal` generally cleaner for expansion entries; in `High`, use tighter stops
7. `PCVR/PCR`: use as supporting context, not as primary trigger

## PE Buy (Bearish) Ideal Setup

1. `Regime`: `Trend Down`
2. `MTF Align`: `Bearish Strong` (or `Bearish Weak`)
3. `Quadrant (1m/5m)`: prefer `Short Build` or `Long Unwind`
4. `Net OI 5m`: negative and expanding in magnitude
5. `ADX 5m`: rising and typically > 20
6. `Vol State`: `Low` or `Normal` generally cleaner; in `High`, reduce size and use quicker exits
7. `PCVR/PCR`: supporting context only

## Avoid / Skip Conditions

1. `Regime = Chop`
2. `MTF Align = Mixed/No-Trade`
3. 1m and 5m quadrants strongly opposite for multiple bars
4. Large IV spike at entry (unless it is a clean momentum breakout with strict risk control)

## Simple Execution Rule

1. Enter only when the first four conditions align.
2. Exit if `MTF Align` flips to mixed/opposite.
3. Exit if 5m quadrant turns opposite with weakening ADX.
