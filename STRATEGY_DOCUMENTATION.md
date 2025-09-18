# Equity Backtesting System: Strategy Documentation

## 1. System Overview

This document details the logic and workflow of the integrated equity backtesting system. The system is designed to backtest a specific intraday momentum breakout strategy using historical data from the Fyers API.

The core workflow can be broken down into three main phases:
1.  **Pre-Market Filtering:** Selects a small watchlist of stocks that exhibit strong bullish characteristics based on their previous day's performance.
2.  **Intraday Breakout Scanning:** Monitors the watchlist for specific breakout patterns on a 5-minute timeframe.
3.  **Trade Execution & Management:** Manages the entry, exit, and risk for trades based on 1-minute candle data.

---

## 2. Strategy Logic in Detail

### Phase 1: Pre-Market Filtering

Before the market opens, the system filters the entire list of symbols to find the most promising candidates. A stock must pass **ALL** of the following criteria to be placed on the day's watchlist. If it fails even one, it is rejected.

*   **Price Range:** The stock's last closing price must be within the `MIN_PRICE` and `MAX_PRICE` defined in `config.py`.
*   **Bullish Candle:** The previous day must have been an "up" day (the closing price was higher than the opening price).
*   **Higher High:** The highest price reached yesterday must be greater than the highest price reached the day before.
*   **Strength:** The closing price from yesterday must be higher than the opening price from two days ago.
*   **EMA Trend:** The 15-day Exponential Moving Average (EMA) must be in an uptrend (the EMA value from yesterday is higher than the value from two days ago).
*   **EMA Support:** The lowest price yesterday must have remained above the 15-day EMA, indicating the trend line acted as a support level.

### Phase 2: Intraday Breakout Scanning (5-Minute Chart)

Once the market opens, the system watches the filtered stocks for two types of breakouts on the 5-minute chart.

1.  **Initial Breakout (Opening Candle):**
    *   This scan happens only on the very first 5-minute candle of the day (e.g., 09:15 - 09:20).
    *   It looks for a breakout above the **Previous Day's High (PDH)**.
    *   A key component is the **Rejection** calculation. This measures the amount of selling pressure on the breakout candle. A low rejection percentage indicates a strong, decisive breakout with buyers in control. The trade is only considered if `Rejection <= INITIAL_REJECTION_THRESHOLD`.

2.  **Stage Breakout (Later Candles):**
    *   This scan happens on any subsequent 5-minute candle.
    *   It also looks for a breakout where the candle closes above the Previous Day's High (PDH).

If either breakout is detected, a potential trade setup is created and passed to the next phase.

### Phase 3: Trade Execution & Management (1-Minute Chart)

When a potential trade is identified, the system switches to the 1-minute chart to time the entry and manage the trade.

*   **Entry Condition:** An entry is triggered only when **two conditions are met simultaneously**:
    1.  **Price Breakout:** The price on the 1-minute chart must touch or exceed the entry point defined by the high of the 5-minute breakout candle.
    2.  **Spurt Signal:** A proprietary "Spurt Signal" must be active, confirming a surge in volume and price momentum. This acts as a confirmation to avoid false breakouts.

*   **Exit Conditions:** Once a trade is active, it can be closed in one of three ways:
    1.  **Target Hit:** The price reaches the pre-defined target price.
    2.  **Stop-Loss Hit:** The price falls to the pre-defined stop-loss price.
    3.  **End-of-Day (EOD) Square-Off:** If the trade is still open at `DAY_END_SQUARE_OFF` time, it is automatically closed at the market price.

---

## 3. System Flow Diagram

This diagram illustrates the decision-making process of the system from start to finish.

```mermaid
graph TD
    A[Start] --> B{Load Symbols & Config};
    B --> C{For Each Symbol};
    C --> D[Run Pre-Market Filter];
    D --> E{Pass All Filters?};
    E -- No --> C;
    E -- Yes --> F[Add to Watchlist];
    C -- All Symbols Processed --> G{Start Market Simulation (Minute by Minute)};
    G --> H{Is it a 5-min candle closing time?};
    H -- Yes --> I[Scan 5-min Candle for Breakout Signal];
    I --> J{Breakout Found?};
    J -- Yes --> K[Create Potential Trade Setup];
    J -- No --> L;
    H -- No --> L{Process 1-min Candle};
    K --> L;
    L --> M{Any Pending Trades?};
    M -- Yes --> N[Check 1-min Candle for Entry Conditions];
    N --> O{Price Breakout AND Spurt Signal?};
    O -- Yes --> P[Enter Trade (Status: Active)];
    O -- No --> Q;
    M -- No --> Q;
    P --> Q;
    Q{Any Active Trades?} --> R[Check 1-min Candle for Exit Conditions];
    R --> S{Target or Stop-Loss Hit?};
    S -- Yes --> T[Close Trade & Log P/L];
    S -- No --> U{End of Day?};
    U -- Yes --> T;
    U -- No --> V{Continue to Next Minute};
    T --> V;
    Q -- No --> V;
    V --> G;
    G -- Market Closed --> W[Generate Final Report];
    W --> X[End];
```

---

## 4. Key Configuration Parameters (`config.py`)

The following parameters in the `config.py` file can be tuned to adjust the strategy's behavior:

*   `MIN_PRICE` / `MAX_PRICE`: The price range for stocks to consider.
*   `EMA_LENGTH`: The lookback period for the Exponential Moving Average.
*   `INITIAL_REJECTION_THRESHOLD`: The maximum allowed "rejection" or selling pressure on an initial breakout candle.
*   `INITIAL_TARGET_PERCENT` / `INITIAL_SL_PERCENT`: Risk/Reward settings for initial breakouts.
*   `STAGE_TARGET_PERCENT` / `STAGE_SL_PERCENT`: Risk/Reward settings for stage breakouts.
*   **Spurt Signal Tuning:**
    *   `VOLUME_SPURT_FACTOR`: Lower for more sensitivity to volume spikes.
    *   `PRICE_SPURT_FACTOR`: Lower for more sensitivity to price moves.
    *   `VOLUME_WINDOW` / `ATR_WINDOW`: Shorter for faster-reacting indicators.
