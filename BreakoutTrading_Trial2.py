#!/usr/bin/env python3
"""
ENHANCED STOCK FILTER WITH BACKTESTING FRAMEWORK
===============================================
Filter equities and backtest the strategy performance
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Try to import Fyers SDK
try:
    from fyers_apiv3 import fyersModel
except ImportError:
    try:
        from fyers_api import fyersModel
    except ImportError:
        print(f"{Fore.RED}Error: No Fyers SDK found. Please install: pip install fyers-apiv3")
        sys.exit(1)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    # File Paths
    SYMBOL_CSV: str = "/Users/Scripts/Equity/EQ_ST_EMA/Fyers/Nifty_SmallCap.csv"
    CONFIG_PY_PATH: str = "/Users/Scripts/Equity/EQ_ST_EMA/Fyers"
    ACCESS_TOKEN_FILE: str = "fyers_access_token.json"

    # Filter Criteria - Stage 1
    MIN_PRICE: float = 23.0
    MAX_PRICE: float = 2300.0

    # EMA Parameters
    EMA_LENGTH: int = 15

    # Backtesting Parameters
    BACKTEST_DAYS: int = 5  # Days to backtest
    POSITION_SIZE: float = 10000  # Position size in INR
    STOP_LOSS_PCT: float = 2.0  # Stop loss percentage
    TARGET_PCT: float = 5.0  # Target percentage
    HOLDING_DAYS: int = 1  # Maximum holding days


config = Config()

# Import API credentials
sys.path.append(config.CONFIG_PY_PATH)
try:
    import Config as APIConfig
except Exception as e:
    print(f"{Fore.RED}Could not import Config.py: {e}")
    sys.exit(1)


# ============================================================================
# AUTHENTICATION
# ============================================================================

class FyersAuth:
    def __init__(self):
        self.access_token = None
        self.token_file = config.ACCESS_TOKEN_FILE

    def load_saved_token(self):
        """Load saved access token if valid"""
        try:
            if os.path.exists(self.token_file):
                with open(self.token_file, 'r') as f:
                    data = json.load(f)
                token = data.get('access_token')
                expiry = data.get('expiry', 0)
                # Check if token expires in next 10 minutes
                if token and time.time() + 600 < expiry:
                    return token
        except Exception:
            pass
        return None

    def authenticate(self):
        """Authenticate with Fyers API"""
        saved_token = self.load_saved_token()
        if saved_token:
            self.access_token = saved_token
            return saved_token

        print(f"{Fore.YELLOW}Please authenticate with Fyers API manually and update the token file.")
        return None


# ============================================================================
# DATA FETCHER
# ============================================================================

class FyersData:
    def __init__(self, access_token):
        self.client = fyersModel.FyersModel(client_id=APIConfig.client_id, token=access_token)

    def format_symbol(self, symbol):
        """Format symbol for Fyers API"""
        if not symbol.startswith('NSE:') or not symbol.endswith('-EQ'):
            # Handle special characters like &
            formatted_symbol = symbol.replace('&', '%26')
            return f"NSE:{formatted_symbol}-EQ"
        return symbol

    def get_daily_data(self, symbol, days_back=90):
        """Get daily OHLCV data for a symbol with sufficient history for backtesting"""
        formatted_symbol = self.format_symbol(symbol)

        # Calculate date range - need more days for backtesting
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        params = {
            "symbol": formatted_symbol,
            "resolution": "D",  # Daily data
            "date_format": "1",
            "range_from": start_date,
            "range_to": end_date,
            "cont_flag": "1"
        }

        try:
            response = self.client.history(params)
            if response.get('code') != 200:
                return None, f"API Error: {response.get('message', 'Unknown')}"

            candles = response.get("candles", [])
            if not candles or len(candles) < config.EMA_LENGTH + 10:
                return None, f"Insufficient data: got {len(candles)} candles"

            # Convert to DataFrame
            data = []
            for candle in candles:
                timestamp = int(candle[0]) // 1000 if candle[0] > 1e12 else int(candle[0])
                data.append({
                    "date": datetime.fromtimestamp(timestamp).date(),
                    "open": float(candle[1]),
                    "high": float(candle[2]),
                    "low": float(candle[3]),
                    "close": float(candle[4]),
                    "volume": float(candle[5]) if len(candle) > 5 else 0.0
                })

            df = pd.DataFrame(data).sort_values('date').reset_index(drop=True)

            # Calculate EMA
            df = self.calculate_ema(df)

            return df, None

        except Exception as e:
            return None, str(e)

    def calculate_ema(self, df):
        """Calculate Exponential Moving Average"""
        if df.empty or len(df) < config.EMA_LENGTH:
            return df

        # Calculate EMA using pandas ewm function
        df[f'ema_{config.EMA_LENGTH}'] = df['close'].ewm(span=config.EMA_LENGTH, adjust=False).mean()

        return df


# ============================================================================
# ENHANCED STOCK FILTER ENGINE
# ============================================================================

class EnhancedStockFilterEngine:
    def __init__(self, fyers_data):
        self.fyers_data = fyers_data

    def filter_stock(self, symbol, date_idx=None):
        """Filter individual stock based on enhanced Stage 1 criteria"""
        try:
            # Get daily data with sufficient history
            daily_df, error = self.fyers_data.get_daily_data(symbol)
            if error or daily_df is None or len(daily_df) < config.EMA_LENGTH + 3:
                return None, f"Data error: {error or 'Insufficient data'}"

            # For backtesting, use specific date index, otherwise use latest
            if date_idx is None:
                date_idx = len(daily_df) - 1

            if date_idx < 2:
                return None, "Need at least 3 days of data"

            # Get the required days of data
            latest_candle = daily_df.iloc[date_idx]  # Current day
            previous_day = daily_df.iloc[date_idx - 1]  # 1 day ago
            two_days_ago = daily_df.iloc[date_idx - 2]  # 2 days ago

            # Extract required values
            current_price = latest_candle['close']
            pdh = previous_day['high']  # Previous Day High
            pdl = previous_day['low']  # Previous Day Low
            two_dh = two_days_ago['high']  # 2 Days High
            two_dl = two_days_ago['low']  # 2 Days Low
            two_days_close = two_days_ago['close']  # 2 days ago close
            one_day_low = previous_day['low']  # 1 day ago low

            # Get EMA values
            ema_col = f'ema_{config.EMA_LENGTH}'
            if ema_col not in daily_df.columns:
                return None, "EMA not calculated"

            ema_2days = two_days_ago[ema_col]  # EMA of 2 days ago
            ema_1day = previous_day[ema_col]  # EMA of 1 day ago

            # STAGE 1 FILTER CONDITIONS
            filter_reasons = []

            # 1. Price range filter (Rs 23 - 2300)
            if not (config.MIN_PRICE <= current_price <= config.MAX_PRICE):
                filter_reasons.append(
                    f"Price {current_price:.2f} outside range [{config.MIN_PRICE}-{config.MAX_PRICE}]")

            # 2. PDH > 2DH condition
            if not (pdh > two_dh):
                filter_reasons.append(f"PDH {pdh:.2f} not > 2DH {two_dh:.2f}")

            # 3. EMA slope condition: EMA of 1 day ago > EMA of 2 days ago
            if not (ema_1day > ema_2days):
                filter_reasons.append(f"EMA slope negative: EMA1D {ema_1day:.2f} not > EMA2D {ema_2days:.2f}")

            # 4. 2 days ago close > EMA (2 days ago)
            if not (two_days_close > ema_2days):
                filter_reasons.append(f"2D close {two_days_close:.2f} not > EMA2D {ema_2days:.2f}")

            # 5. 1 day ago low > EMA (2 days ago) - bullish sign
            if not (one_day_low > ema_2days):
                filter_reasons.append(f"1D low {one_day_low:.2f} not > EMA2D {ema_2days:.2f}")

            # If any condition fails, return rejection
            if filter_reasons:
                return None, "; ".join(filter_reasons[:2])  # Show first 2 reasons

            # If all conditions pass, return the filtered data
            filtered_data = {
                'symbol': symbol,
                'date': daily_df['date'].iloc[date_idx],
                'current_price': round(current_price, 2),
                'pdh': round(pdh, 2),
                'pdl': round(pdl, 2),
                '2dh': round(two_dh, 2),
                '2dl': round(two_dl, 2),
                'pdh_vs_2dh': round(pdh - two_dh, 2),
                '2d_close': round(two_days_close, 2),
                '1d_low': round(one_day_low, 2),
                f'ema_{config.EMA_LENGTH}_2d': round(ema_2days, 2),
                f'ema_{config.EMA_LENGTH}_1d': round(ema_1day, 2),
                'ema_slope': round(ema_1day - ema_2days, 2),
                'bullish_score': round(
                    (pdh - two_dh) + (ema_1day - ema_2days) +
                    (two_days_close - ema_2days) + (one_day_low - ema_2days), 2
                )
            }

            return filtered_data, daily_df

        except Exception as e:
            return None, f"Exception: {str(e)}"


# ============================================================================
# BACKTESTING FRAMEWORK
# ============================================================================

class BacktestFramework:
    def __init__(self, filter_engine, fyers_data):
        self.filter_engine = filter_engine
        self.fyers_data = fyers_data
        self.trades = []

    def simulate_trade(self, symbol, entry_date_idx, daily_df):
        """Simulate a single trade"""
        try:
            entry_price = daily_df.iloc[entry_date_idx]['close']
            entry_date = daily_df.iloc[entry_date_idx]['date']

            # Calculate position size
            shares = int(config.POSITION_SIZE / entry_price)
            if shares == 0:
                return None

            # Calculate stop loss and target
            stop_loss = entry_price * (1 - config.STOP_LOSS_PCT / 100)
            target = entry_price * (1 + config.TARGET_PCT / 100)

            # Simulate holding period
            max_exit_idx = min(entry_date_idx + config.HOLDING_DAYS, len(daily_df) - 1)

            for i in range(entry_date_idx + 1, max_exit_idx + 1):
                current_day = daily_df.iloc[i]

                # Check stop loss
                if current_day['low'] <= stop_loss:
                    exit_price = stop_loss
                    exit_date = current_day['date']
                    exit_reason = 'Stop Loss'
                    break

                # Check target
                if current_day['high'] >= target:
                    exit_price = target
                    exit_date = current_day['date']
                    exit_reason = 'Target Hit'
                    break

                # If last day, exit at close
                if i == max_exit_idx:
                    exit_price = current_day['close']
                    exit_date = current_day['date']
                    exit_reason = 'Time Exit'
                    break
            else:
                # If no exit condition met, exit at last available price
                exit_price = daily_df.iloc[max_exit_idx]['close']
                exit_date = daily_df.iloc[max_exit_idx]['date']
                exit_reason = 'Time Exit'

            # Calculate P&L
            pnl_per_share = exit_price - entry_price
            total_pnl = pnl_per_share * shares
            pnl_percent = (pnl_per_share / entry_price) * 100

            trade = {
                'symbol': symbol,
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': round(entry_price, 2),
                'exit_price': round(exit_price, 2),
                'shares': shares,
                'position_value': round(entry_price * shares, 2),
                'pnl': round(total_pnl, 2),
                'pnl_percent': round(pnl_percent, 2),
                'exit_reason': exit_reason,
                'holding_days': (exit_date - entry_date).days
            }

            return trade

        except Exception as e:
            print(f"Error simulating trade for {symbol}: {e}")
            return None

    def run_backtest(self, symbols, start_date=None, end_date=None):
        """Run backtest on filtered symbols"""
        print(f"\n{Fore.CYAN}RUNNING BACKTEST...{Style.RESET_ALL}")

        if start_date is None:
            start_date = datetime.now() - timedelta(days=config.BACKTEST_DAYS)
        if end_date is None:
            end_date = datetime.now()

        all_trades = []

        for symbol in symbols[:10]:  # Limit to 10 symbols for demo
            print(f"Backtesting {symbol}...", end=" ")

            # Get data for this symbol
            daily_df, error = self.fyers_data.get_daily_data(symbol, 100)
            if error or daily_df is None:
                print(f"{Fore.RED}Failed{Style.RESET_ALL}")
                continue

            # Convert dates to datetime for comparison
            daily_df['datetime'] = pd.to_datetime(daily_df['date'])

            # Find qualifying dates in backtest period
            trades_for_symbol = []

            for idx in range(config.EMA_LENGTH + 2, len(daily_df) - config.HOLDING_DAYS):
                current_date = daily_df.iloc[idx]['datetime']

                if start_date <= current_date.to_pydatetime() <= end_date:
                    # Check if stock qualifies on this date
                    stock_data, _ = self.filter_engine.filter_stock(symbol, idx)

                    if stock_data:
                        # Simulate trade
                        trade = self.simulate_trade(symbol, idx, daily_df)
                        if trade:
                            trades_for_symbol.append(trade)

            all_trades.extend(trades_for_symbol)
            print(f"{Fore.GREEN}{len(trades_for_symbol)} trades{Style.RESET_ALL}")
            time.sleep(0.1)  # Rate limiting

        return all_trades

    def analyze_backtest_results(self, trades):
        """Analyze backtest results"""
        if not trades:
            print(f"{Fore.RED}No trades to analyze{Style.RESET_ALL}")
            return

        df_trades = pd.DataFrame(trades)

        # Calculate statistics
        total_trades = len(trades)
        winning_trades = len(df_trades[df_trades['pnl'] > 0])
        losing_trades = len(df_trades[df_trades['pnl'] < 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

        total_pnl = df_trades['pnl'].sum()
        avg_pnl = df_trades['pnl'].mean()
        max_profit = df_trades['pnl'].max()
        max_loss = df_trades['pnl'].min()

        avg_holding_days = df_trades['holding_days'].mean()

        # Display results
        print(f"\n{Fore.CYAN}{'=' * 80}")
        print(f"{'BACKTEST RESULTS':^80}")
        print(f"{'=' * 80}{Style.RESET_ALL}")

        print(f"\n{Fore.WHITE}TRADE STATISTICS:")
        print(f"• Total Trades: {total_trades}")
        print(f"• Winning Trades: {Fore.GREEN}{winning_trades}{Style.RESET_ALL}")
        print(f"• Losing Trades: {Fore.RED}{losing_trades}{Style.RESET_ALL}")
        print(f"• Win Rate: {win_rate:.1f}%")
        print(f"• Average Holding Days: {avg_holding_days:.1f}")

        print(f"\n{Fore.WHITE}P&L ANALYSIS:")
        print(f"• Total P&L: {Fore.GREEN if total_pnl > 0 else Fore.RED}₹{total_pnl:,.2f}{Style.RESET_ALL}")
        print(f"• Average P&L per Trade: {Fore.GREEN if avg_pnl > 0 else Fore.RED}₹{avg_pnl:,.2f}{Style.RESET_ALL}")
        print(f"• Maximum Profit: {Fore.GREEN}₹{max_profit:,.2f}{Style.RESET_ALL}")
        print(f"• Maximum Loss: {Fore.RED}₹{max_loss:,.2f}{Style.RESET_ALL}")

        # Show sample trades
        print(f"\n{Fore.YELLOW}SAMPLE TRADES (Top 5 Profitable):")
        top_trades = df_trades.nlargest(5, 'pnl')
        for _, trade in top_trades.iterrows():
            print(f"• {trade['symbol']} | Entry: ₹{trade['entry_price']} | "
                  f"Exit: ₹{trade['exit_price']} | P&L: ₹{trade['pnl']} | "
                  f"Reason: {trade['exit_reason']}")

        return df_trades


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_symbols():
    """Load symbols from CSV file"""
    if not os.path.exists(config.SYMBOL_CSV):
        raise FileNotFoundError(f"Symbol file not found: {config.SYMBOL_CSV}")

    df = pd.read_csv(config.SYMBOL_CSV)
    symbols = df["Symbol"].dropna().astype(str).str.strip().tolist()

    # Clean symbols
    clean_symbols = []
    for symbol in symbols:
        if symbol and symbol != 'nan':
            clean_symbol = symbol.replace('NSE:', '').replace('-EQ', '').strip()
            if clean_symbol:
                clean_symbols.append(clean_symbol)

    return clean_symbols


def run_current_filter():
    """Run current day filtering"""
    print(f"\n{Fore.CYAN}RUNNING CURRENT DAY FILTERING...{Style.RESET_ALL}")

    # Get current day qualified stocks
    symbols = load_symbols()

    # Initialize components
    auth = FyersAuth()
    access_token = auth.authenticate()
    if not access_token:
        return []

    fyers_data = FyersData(access_token)
    filter_engine = EnhancedStockFilterEngine(fyers_data)

    filtered_stocks = []
    for symbol in symbols[:20]:  # Limit for demo
        print(f"Filtering {symbol}...", end=" ")
        stock_data, _ = filter_engine.filter_stock(symbol)
        if stock_data:
            filtered_stocks.append(stock_data)
            print(f"{Fore.GREEN}✓{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}✗{Style.RESET_ALL}")
        time.sleep(0.05)

    return filtered_stocks


# ============================================================================
# MAIN EXECUTION FUNCTIONS
# ============================================================================

def main():
    """Main function with menu-driven execution"""
    print(f"\n{Fore.CYAN}ENHANCED STOCK FILTER WITH BACKTESTING")
    print(f"======================================={Style.RESET_ALL}")

    while True:
        print(f"\n{Fore.WHITE}Choose an option:")
        print("1. Run Current Day Filter")
        print("2. Run Backtest")
        print("3. Run Both (Filter + Backtest)")
        print("4. Exit")

        choice = input(f"\n{Fore.YELLOW}Enter your choice (1-4): {Style.RESET_ALL}").strip()

        if choice == '1':
            run_filter_only()
        elif choice == '2':
            run_backtest_only()
        elif choice == '3':
            run_filter_and_backtest()
        elif choice == '4':
            print(f"{Fore.CYAN}Goodbye!{Style.RESET_ALL}")
            break
        else:
            print(f"{Fore.RED}Invalid choice. Please try again.{Style.RESET_ALL}")


def run_filter_only():
    """Run only the current day filtering"""
    try:
        print(f"\n{Fore.YELLOW}Step 1: Authenticating...{Style.RESET_ALL}")
        auth = FyersAuth()
        access_token = auth.authenticate()
        if not access_token:
            return

        print(f"\n{Fore.YELLOW}Step 2: Loading symbols...{Style.RESET_ALL}")
        symbols = load_symbols()

        print(f"\n{Fore.YELLOW}Step 3: Filtering stocks...{Style.RESET_ALL}")
        fyers_data = FyersData(access_token)
        filter_engine = EnhancedStockFilterEngine(fyers_data)

        filtered_stocks = []
        for symbol in symbols:
            stock_data, _ = filter_engine.filter_stock(symbol)
            if stock_data:
                filtered_stocks.append(stock_data)

        # Display results
        if filtered_stocks:
            df = pd.DataFrame(filtered_stocks)
            print(f"\n{Fore.GREEN}Found {len(filtered_stocks)} qualified stocks!{Style.RESET_ALL}")
            print(df[['symbol', 'current_price', 'bullish_score']].head(10))
        else:
            print(f"{Fore.RED}No stocks passed the filter{Style.RESET_ALL}")

    except Exception as e:
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")


def run_backtest_only():
    """Run only the backtesting"""
    try:
        print(f"\n{Fore.YELLOW}Step 1: Authenticating...{Style.RESET_ALL}")
        auth = FyersAuth()
        access_token = auth.authenticate()
        if not access_token:
            return

        print(f"\n{Fore.YELLOW}Step 2: Loading symbols...{Style.RESET_ALL}")
        symbols = load_symbols()

        print(f"\n{Fore.YELLOW}Step 3: Initializing backtest...{Style.RESET_ALL}")
        fyers_data = FyersData(access_token)
        filter_engine = EnhancedStockFilterEngine(fyers_data)
        backtest = BacktestFramework(filter_engine, fyers_data)

        # Run backtest
        trades = backtest.run_backtest(symbols)

        # Analyze results
        backtest.analyze_backtest_results(trades)

    except Exception as e:
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")


def run_filter_and_backtest():
    """Run both filtering and backtesting"""
    try:
        # First run current day filter
        print(f"\n{Fore.MAGENTA}PHASE 1: CURRENT DAY FILTERING{Style.RESET_ALL}")
        run_filter_only()

        # Then run backtest
        print(f"\n{Fore.MAGENTA}PHASE 2: BACKTESTING{Style.RESET_ALL}")
        run_backtest_only()

    except Exception as e:
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")


if __name__ == "__main__":
    main()