#!/usr/bin/env python3
"""
INTEGRATED STOCK FILTER AND REAL-TIME MONITOR
=============================================
Stage 1: Enhanced Stock Filter
Stage 2: Real-time Breakout Monitoring
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, List, Dict
from colorama import Fore, Style, init
import threading
from concurrent.futures import ThreadPoolExecutor
import queue

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
    EMA_LENGTH: int = 15

    # Stage 2 - Real-time Monitoring
    MARKET_OPEN_TIME: str = "09:15"
    MARKET_CLOSE_TIME: str = "15:30"
    CANDLE_INTERVAL: int = 5  # minutes
    REJECTION_THRESHOLD: float = 0.33  # 33% rejection rule
    TICK_REFRESH_INTERVAL: float = 1.0  # seconds for tick monitoring


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
            formatted_symbol = symbol.replace('&', '%26')
            return f"NSE:{formatted_symbol}-EQ"
        return symbol

    def get_daily_data(self, symbol, days_back=30):
        """Get daily OHLCV data for Stage 1 filtering"""
        formatted_symbol = self.format_symbol(symbol)
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        params = {
            "symbol": formatted_symbol,
            "resolution": "D",
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
            if not candles or len(candles) < config.EMA_LENGTH + 3:
                return None, f"Insufficient data: got {len(candles)} candles"

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
            df = self.calculate_ema(df)
            return df, None

        except Exception as e:
            return None, str(e)

    def get_intraday_data(self, symbol, resolution="5"):
        """Get intraday data for Stage 2 monitoring"""
        formatted_symbol = self.format_symbol(symbol)
        today = datetime.now().strftime("%Y-%m-%d")

        params = {
            "symbol": formatted_symbol,
            "resolution": resolution,
            "date_format": "1",
            "range_from": today,
            "range_to": today,
            "cont_flag": "1"
        }

        try:
            response = self.client.history(params)
            if response.get('code') != 200:
                return None, f"API Error: {response.get('message', 'Unknown')}"

            candles = response.get("candles", [])
            if not candles:
                return None, "No intraday data available"

            data = []
            for candle in candles:
                timestamp = int(candle[0]) // 1000 if candle[0] > 1e12 else int(candle[0])
                dt = datetime.fromtimestamp(timestamp)
                data.append({
                    "datetime": dt,
                    "time": dt.strftime("%H:%M"),
                    "open": float(candle[1]),
                    "high": float(candle[2]),
                    "low": float(candle[3]),
                    "close": float(candle[4]),
                    "volume": float(candle[5]) if len(candle) > 5 else 0.0
                })

            return pd.DataFrame(data), None

        except Exception as e:
            return None, str(e)

    def get_current_price(self, symbol):
        """Get current tick price for real-time monitoring"""
        formatted_symbol = self.format_symbol(symbol)

        try:
            response = self.client.quotes({"symbols": formatted_symbol})
            if response.get('code') != 200:
                return None, f"Quote API Error: {response.get('message', 'Unknown')}"

            quotes = response.get('d', [])
            if quotes:
                quote = quotes[0]
                return float(quote.get('v', {}).get('lp', 0)), None

            return None, "No quote data available"

        except Exception as e:
            return None, str(e)

    def calculate_ema(self, df):
        """Calculate Exponential Moving Average"""
        if df.empty or len(df) < config.EMA_LENGTH:
            return df
        df[f'ema_{config.EMA_LENGTH}'] = df['close'].ewm(span=config.EMA_LENGTH, adjust=False).mean()
        return df


# ============================================================================
# STAGE 1 - ENHANCED STOCK FILTER ENGINE
# ============================================================================

class EnhancedStockFilterEngine:
    def __init__(self, fyers_data):
        self.fyers_data = fyers_data

    def filter_stock(self, symbol):
        """Filter individual stock based on enhanced Stage 1 criteria"""
        try:
            daily_df, error = self.fyers_data.get_daily_data(symbol)
            if error or daily_df is None or len(daily_df) < config.EMA_LENGTH + 3:
                return None, f"Data error: {error or 'Insufficient data'}"

            if len(daily_df) < 3:
                return None, "Need at least 3 days of data"

            latest_candle = daily_df.iloc[-1]
            previous_day = daily_df.iloc[-2]
            two_days_ago = daily_df.iloc[-3]

            current_price = latest_candle['close']
            pdh = previous_day['high']
            pdl = previous_day['low']
            two_dh = two_days_ago['high']
            two_dl = two_days_ago['low']
            two_days_close = two_days_ago['close']
            one_day_low = previous_day['low']

            ema_col = f'ema_{config.EMA_LENGTH}'
            if ema_col not in daily_df.columns:
                return None, "EMA not calculated"

            ema_2days = two_days_ago[ema_col]
            ema_1day = previous_day[ema_col]

            filter_reasons = []

            # Stage 1 Filter Conditions
            if not (config.MIN_PRICE <= current_price <= config.MAX_PRICE):
                filter_reasons.append(f"Price {current_price:.2f} outside range")

            if not (pdh > two_dh):
                filter_reasons.append(f"PDH {pdh:.2f} not > 2DH {two_dh:.2f}")

            if not (ema_1day > ema_2days):
                filter_reasons.append(f"EMA slope negative")

            if not (two_days_close > ema_2days):
                filter_reasons.append(f"2D close not > EMA2D")

            if not (one_day_low > ema_2days):
                filter_reasons.append(f"1D low not > EMA2D")

            if filter_reasons:
                return None, "; ".join(filter_reasons[:2])

            filtered_data = {
                'symbol': symbol,
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
                'bullish_score': round((pdh - two_dh) + (ema_1day - ema_2days) +
                                       (two_days_close - ema_2days) + (one_day_low - ema_2days), 2),
                'date': daily_df['date'].iloc[-1].strftime('%Y-%m-%d')
            }

            return filtered_data, None

        except Exception as e:
            return None, f"Exception: {str(e)}"

    def process_symbols(self, symbols):
        """Process all symbols and return filtered results"""
        print(f"\n{Fore.BLUE}Processing {len(symbols)} symbols for Stage 1 filtering...{Style.RESET_ALL}")

        filtered_stocks = []
        failed_stocks = []

        for i, symbol in enumerate(symbols, 1):
            print(f"{Fore.WHITE}Progress: {i:>3}/{len(symbols)} - Processing {symbol:<15}{Style.RESET_ALL}", end="")

            stock_data, error = self.filter_stock(symbol)

            if stock_data:
                filtered_stocks.append(stock_data)
                print(f" {Fore.GREEN}✓ PASSED{Style.RESET_ALL}")
            else:
                failed_stocks.append({'symbol': symbol, 'reason': error})
                print(f" {Fore.RED}✗ FAILED{Style.RESET_ALL}")

            time.sleep(0.05)

        return filtered_stocks, failed_stocks


# ============================================================================
# STAGE 2 - REAL-TIME BREAKOUT MONITOR
# ============================================================================

class RealTimeBreakoutMonitor:
    def __init__(self, fyers_data, filtered_stocks_df):
        self.fyers_data = fyers_data
        self.filtered_stocks_df = filtered_stocks_df
        self.initial_breakers = pd.DataFrame()
        self.stage_breakouts = pd.DataFrame()
        self.monitoring_active = False
        self.buy_signals = []

        # Thread-safe queues for data sharing
        self.signal_queue = queue.Queue()

    def is_market_open(self):
        """Check if market is currently open"""
        now = datetime.now()
        current_time = now.strftime("%H:%M")

        # Convert to comparable format
        market_open = datetime.strptime(config.MARKET_OPEN_TIME, "%H:%M").time()
        market_close = datetime.strptime(config.MARKET_CLOSE_TIME, "%H:%M").time()
        current_time_obj = datetime.strptime(current_time, "%H:%M").time()

        # Check if it's a weekday (Monday=0, Sunday=6)
        if now.weekday() >= 5:  # Saturday or Sunday
            return False

        return market_open <= current_time_obj <= market_close

    def wait_for_market_open(self):
        """Wait until market opens"""
        while not self.is_market_open():
            now = datetime.now()
            print(
                f"\r{Fore.YELLOW}Waiting for market to open... Current time: {now.strftime('%H:%M:%S')}{Style.RESET_ALL}",
                end="")
            time.sleep(30)  # Check every 30 seconds
        print(f"\n{Fore.GREEN}Market is now open! Starting real-time monitoring...{Style.RESET_ALL}")

    def get_first_candle_data(self, symbol):
        """Get the first 5-minute candle (09:15-09:20) data"""
        try:
            intraday_df, error = self.fyers_data.get_intraday_data(symbol, "5")
            if error or intraday_df is None or intraday_df.empty:
                return None, f"Intraday data error: {error}"

            # Find the first candle (09:15)
            first_candle = intraday_df[intraday_df['time'] == '09:15']
            if first_candle.empty:
                return None, "First candle (09:15) not found"

            return first_candle.iloc[0], None

        except Exception as e:
            return None, str(e)

    def analyze_initial_breakout(self, symbol, pdh):
        """Analyze if stock qualifies for Initial Breakout monitoring"""
        try:
            first_candle, error = self.get_first_candle_data(symbol)
            if error or first_candle is None:
                return None, error

            candle_open = first_candle['open']
            candle_high = first_candle['high']
            candle_low = first_candle['low']
            candle_close = first_candle['close']

            # Check if open is below PDH (filter condition)
            if candle_open >= pdh:
                return None, f"Open {candle_open:.2f} >= PDH {pdh:.2f}, filtered out"

            # Check if candle is bullish (close > open)
            if candle_close <= candle_open:
                return "stage_breakout", f"Bearish candle, moved to Stage Breakouts"

            # Calculate rejection percentage
            candle_range = candle_high - candle_low
            rejection = candle_high - candle_close
            rejection_percentage = (rejection / candle_range) if candle_range > 0 else 0

            # Check 33% rejection rule
            if rejection_percentage > config.REJECTION_THRESHOLD:
                return None, f"Rejection {rejection_percentage:.1%} > 33%, filtered out"

            # Qualified for Initial Breakout
            initial_data = {
                'symbol': symbol,
                'pdh': pdh,
                'candle_open': candle_open,
                'candle_high': candle_high,
                'candle_low': candle_low,
                'candle_close': candle_close,
                'initial_entry_point': candle_high,
                'rejection_pct': rejection_percentage,
                'status': 'monitoring',
                'timestamp': datetime.now().strftime('%H:%M:%S')
            }

            return "initial_breaker", initial_data

        except Exception as e:
            return None, f"Exception: {str(e)}"

    def monitor_tick_data(self, symbol, entry_point):
        """Monitor tick data for buy signal generation"""
        try:
            current_price, error = self.fyers_data.get_current_price(symbol)
            if error or current_price is None:
                return False, error

            if current_price > entry_point:
                buy_signal = {
                    'symbol': symbol,
                    'signal': 'BUY',
                    'entry_point': entry_point,
                    'trigger_price': current_price,
                    'timestamp': datetime.now().strftime('%H:%M:%S')
                }
                self.buy_signals.append(buy_signal)
                return True, buy_signal

            return False, None

        except Exception as e:
            return False, str(e)

    def process_initial_breakouts(self):
        """Process all filtered stocks for Initial Breakout analysis"""
        if self.filtered_stocks_df.empty:
            print(f"{Fore.RED}No stocks from Stage 1 to monitor{Style.RESET_ALL}")
            return

        print(f"\n{Fore.BLUE}Analyzing {len(self.filtered_stocks_df)} stocks for Initial Breakouts...{Style.RESET_ALL}")

        initial_breakers_list = []
        stage_breakouts_list = []

        for _, stock in self.filtered_stocks_df.iterrows():
            symbol = stock['symbol']
            pdh = stock['pdh']

            print(f"{Fore.WHITE}Analyzing {symbol} (PDH: {pdh:.2f})...{Style.RESET_ALL}", end="")

            result, data = self.analyze_initial_breakout(symbol, pdh)

            if result == "initial_breaker":
                initial_breakers_list.append(data)
                print(f" {Fore.GREEN}✓ INITIAL BREAKER{Style.RESET_ALL}")
            elif result == "stage_breakout":
                stage_breakouts_list.append({
                    'symbol': symbol,
                    'pdh': pdh,
                    'status': 'stage_monitoring',
                    'timestamp': datetime.now().strftime('%H:%M:%S')
                })
                print(f" {Fore.YELLOW}→ STAGE BREAKOUT{Style.RESET_ALL}")
            else:
                print(f" {Fore.RED}✗ FILTERED: {data}{Style.RESET_ALL}")

        # Create DataFrames
        if initial_breakers_list:
            self.initial_breakers = pd.DataFrame(initial_breakers_list)

        if stage_breakouts_list:
            self.stage_breakouts = pd.DataFrame(stage_breakouts_list)

    def display_breakout_results(self):
        """Display Initial Breakout analysis results"""
        print(f"\n{Fore.CYAN}{'=' * 100}")
        print(f"{'STAGE 2 - INITIAL BREAKOUT ANALYSIS RESULTS':^100}")
        print(f"{'=' * 100}{Style.RESET_ALL}")

        print(f"\n{Fore.GREEN}INITIAL BREAKERS (Ready for Tick Monitoring):")
        if not self.initial_breakers.empty:
            print(f"{'Symbol':<12} {'PDH':<8} {'Open':<8} {'Close':<8} {'EntryPoint':<10} {'Rejection':<10} {'Status'}")
            print("-" * 80)
            for _, row in self.initial_breakers.iterrows():
                print(f"{row['symbol']:<12} {row['pdh']:<8.2f} {row['candle_open']:<8.2f} "
                      f"{row['candle_close']:<8.2f} {row['initial_entry_point']:<10.2f} "
                      f"{row['rejection_pct']:<10.1%} {row['status']}")
        else:
            print(f"{Fore.YELLOW}No stocks qualified for Initial Breakout monitoring{Style.RESET_ALL}")

        print(f"\n{Fore.YELLOW}STAGE BREAKOUTS (For Future Monitoring):")
        if not self.stage_breakouts.empty:
            print(f"{'Symbol':<12} {'PDH':<8} {'Status':<15} {'Timestamp'}")
            print("-" * 50)
            for _, row in self.stage_breakouts.iterrows():
                print(f"{row['symbol']:<12} {row['pdh']:<8.2f} {row['status']:<15} {row['timestamp']}")
        else:
            print(f"{Fore.YELLOW}No stocks moved to Stage Breakouts{Style.RESET_ALL}")

    def start_real_time_monitoring(self):
        """Start real-time tick monitoring for Initial Breakers"""
        if self.initial_breakers.empty:
            print(f"\n{Fore.YELLOW}No stocks to monitor for tick data{Style.RESET_ALL}")
            return

        print(f"\n{Fore.CYAN}Starting Real-Time Tick Monitoring...{Style.RESET_ALL}")
        print(f"Monitoring {len(self.initial_breakers)} stocks for BUY signals")
        print(f"Press Ctrl+C to stop monitoring\n")

        self.monitoring_active = True

        try:
            while self.monitoring_active and self.is_market_open():
                for _, stock in self.initial_breakers.iterrows():
                    if stock['status'] != 'monitoring':
                        continue

                    symbol = stock['symbol']
                    entry_point = stock['initial_entry_point']

                    triggered, result = self.monitor_tick_data(symbol, entry_point)

                    if triggered:
                        print(f"\n{Fore.GREEN}🚀 BUY SIGNAL GENERATED!")
                        print(f"Symbol: {result['symbol']}")
                        print(f"Entry Point: {result['entry_point']:.2f}")
                        print(f"Trigger Price: {result['trigger_price']:.2f}")
                        print(f"Time: {result['timestamp']}")
                        print(f"{'=' * 50}{Style.RESET_ALL}\n")

                        # Update status to avoid duplicate signals
                        self.initial_breakers.loc[
                            self.initial_breakers['symbol'] == symbol, 'status'
                        ] = 'triggered'

                # Check if all stocks have triggered
                active_stocks = len(self.initial_breakers[self.initial_breakers['status'] == 'monitoring'])
                if active_stocks == 0:
                    print(f"{Fore.GREEN}All stocks have generated signals or market closed{Style.RESET_ALL}")
                    break

                # Display current monitoring status
                current_time = datetime.now().strftime('%H:%M:%S')
                print(f"\r{Fore.WHITE}[{current_time}] Monitoring {active_stocks} stocks...{Style.RESET_ALL}", end="")

                time.sleep(config.TICK_REFRESH_INTERVAL)

        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Monitoring stopped by user{Style.RESET_ALL}")
        finally:
            self.monitoring_active = False

    def display_final_summary(self):
        """Display final summary of the day's monitoring"""
        print(f"\n{Fore.CYAN}{'=' * 100}")
        print(f"{'FINAL SUMMARY - END OF MONITORING':^100}")
        print(f"{'=' * 100}{Style.RESET_ALL}")

        total_monitored = len(self.initial_breakers) if not self.initial_breakers.empty else 0
        total_signals = len(self.buy_signals)

        print(f"\n{Fore.WHITE}Statistics:")
        print(f"• Total Stocks Monitored: {total_monitored}")
        print(f"• BUY Signals Generated: {Fore.GREEN}{total_signals}{Style.RESET_ALL}")
        print(f"• Stage Breakouts: {len(self.stage_breakouts) if not self.stage_breakouts.empty else 0}")

        if self.buy_signals:
            print(f"\n{Fore.GREEN}BUY SIGNALS SUMMARY:")
            print(f"{'Time':<10} {'Symbol':<12} {'Entry':<8} {'Trigger':<8} {'Signal'}")
            print("-" * 60)
            for signal in self.buy_signals:
                print(f"{signal['timestamp']:<10} {signal['symbol']:<12} "
                      f"{signal['entry_point']:<8.2f} {signal['trigger_price']:<8.2f} "
                      f"{signal['signal']}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_symbols():
    """Load symbols from CSV file"""
    if not os.path.exists(config.SYMBOL_CSV):
        raise FileNotFoundError(f"Symbol file not found: {config.SYMBOL_CSV}")

    df = pd.read_csv(config.SYMBOL_CSV)
    symbols = df["Symbol"].dropna().astype(str).str.strip().tolist()

    clean_symbols = []
    for symbol in symbols:
        if symbol and symbol != 'nan':
            clean_symbol = symbol.replace('NSE:', '').replace('-EQ', '').strip()
            if clean_symbol:
                clean_symbols.append(clean_symbol)

    return clean_symbols


def create_filtered_dataframe(filtered_stocks):
    """Create DataFrame from Stage 1 filtered stocks"""
    if not filtered_stocks:
        return pd.DataFrame()

    df = pd.DataFrame(filtered_stocks)
    df = df.sort_values('bullish_score', ascending=False).reset_index(drop=True)
    return df


def display_stage1_results(filtered_df, failed_stocks, total_symbols):
    """Display Stage 1 filtering results"""
    print(f"\n{Fore.CYAN}{'=' * 100}")
    print(f"{'STAGE 1 - ENHANCED STOCK FILTERING RESULTS':^100}")
    print(f"{'=' * 100}{Style.RESET_ALL}")

    print(f"\n{Fore.WHITE}Filter Criteria:")
    print(f"• Price Range: ₹{config.MIN_PRICE} - ₹{config.MAX_PRICE}")
    print(f"• PDH > 2DH + EMA Uptrend + Bullish Price Action")
    print(f"• Total Symbols Processed: {total_symbols}")
    print(f"• Filtered Stocks: {Fore.GREEN}{len(filtered_df)}{Style.RESET_ALL}")
    print(f"• Failed Stocks: {Fore.RED}{len(failed_stocks)}{Style.RESET_ALL}")

    if not filtered_df.empty:
        print(f"\n{Fore.GREEN}STAGE 1 FILTERED STOCKS (Top 10):")
        print(f"{'Rank':<4} {'Symbol':<12} {'Price':<8} {'PDH':<7} {'Bullish Score':<12}")
        print("-" * 60)

        for idx, (_, row) in enumerate(filtered_df.head(10).iterrows(), 1):
            print(f"{idx:<4} {row['symbol']:<12} {row['current_price']:<8} "
                  f"{row['pdh']:<7} {row['bullish_score']:<12}")

    success_rate = (len(filtered_df) / total_symbols) * 100 if total_symbols > 0 else 0
    print(f"\n{Fore.MAGENTA}Stage 1 Success Rate: {success_rate:.1f}%{Style.RESET_ALL}")


# ============================================================================
# MAIN INTEGRATED FUNCTION
# ============================================================================

def main():
    """Main integrated function for Stage 1 + Stage 2"""
    print(f"\n{Fore.CYAN}INTEGRATED STOCK FILTER AND REAL-TIME MONITOR")
    print(f"Stage 1: Enhanced Stock Filtering")
    print(f"Stage 2: Real-time Breakout Monitoring")
    print(f"=========================================={Style.RESET_ALL}")

    try:
        # ==================== STAGE 1: STOCK FILTERING ====================

        print(f"\n{Fore.YELLOW}STAGE 1: Starting Enhanced Stock Filtering...{Style.RESET_ALL}")

        # Authentication
        auth = FyersAuth()
        access_token = auth.authenticate()
        if not access_token:
            print(f"{Fore.RED}Authentication failed{Style.RESET_ALL}")
            return None

        # Load symbols
        symbols = load_symbols()
        print(f"{Fore.GREEN}Loaded {len(symbols)} symbols{Style.RESET_ALL}")

        # Initialize components
        fyers_data = FyersData(access_token)
        filter_engine = EnhancedStockFilterEngine(fyers_data)

        # Process symbols
        filtered_stocks, failed_stocks = filter_engine.process_symbols(symbols)

        # Create DataFrame
        stage1_filtered_df = create_filtered_dataframe(filtered_stocks)

        # Display Stage 1 results
        display_stage1_results(stage1_filtered_df, failed_stocks, len(symbols))

        if stage1_filtered_df.empty:
            print(f"\n{Fore.RED}No stocks passed Stage 1 filtering. Exiting...{Style.RESET_ALL}")
            return None

        # ==================== STAGE 2: REAL-TIME MONITORING ====================

        print(f"\n{Fore.YELLOW}STAGE 2: Starting Real-time Breakout Monitoring...{Style.RESET_ALL}")

        # Initialize breakout monitor
        breakout_monitor = RealTimeBreakoutMonitor(fyers_data, stage1_filtered_df)

        # Wait for market to open if needed
        if not breakout_monitor.is_market_open():
            print(f"{Fore.YELLOW}Market is currently closed. Waiting for market to open...{Style.RESET_ALL}")
            breakout_monitor.wait_for_market_open()

        # Wait for first candle (09:15-09:20) to complete
        print(f"\n{Fore.BLUE}Waiting for first 5-minute candle (09:15-09:20) to complete...{Style.RESET_ALL}")

        # Wait until 09:20 to analyze first candle
        while True:
            current_time = datetime.now().strftime("%H:%M")
            if current_time >= "09:20":
                break
            print(f"\r{Fore.WHITE}Current time: {current_time}, waiting for 09:20...{Style.RESET_ALL}", end="")
            time.sleep(30)

        print(f"\n{Fore.GREEN}First candle completed! Analyzing breakouts...{Style.RESET_ALL}")

        # Process Initial Breakouts
        breakout_monitor.process_initial_breakouts()

        # Display breakout results
        breakout_monitor.display_breakout_results()

        # Start real-time tick monitoring
        if not breakout_monitor.initial_breakers.empty:
            breakout_monitor.start_real_time_monitoring()

        # Display final summary
        breakout_monitor.display_final_summary()

        print(f"\n{Fore.GREEN}✓ Integrated monitoring completed successfully!{Style.RESET_ALL}")

        return {
            'stage1_filtered': stage1_filtered_df,
            'initial_breakers': breakout_monitor.initial_breakers,
            'stage_breakouts': breakout_monitor.stage_breakouts,
            'buy_signals': breakout_monitor.buy_signals
        }

    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Monitoring stopped by user{Style.RESET_ALL}")
        return None
    except Exception as e:
        print(f"\n{Fore.RED}Error: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    print(f"{Fore.CYAN}╔══════════════════════════════════════════════════════════════════════════════════════╗")
    print(f"║                    INTEGRATED STOCK FILTER & REAL-TIME MONITOR                      ║")
    print(f"║                                                                                      ║")
    print(f"║  Stage 1: Enhanced Stock Filtering (PDH > 2DH + EMA + Bullish Conditions)          ║")
    print(f"║  Stage 2: Real-time Breakout Monitoring (Initial & Stage Breakouts)                ║")
    print(f"║                                                                                      ║")
    print(f"║  Market Hours: 09:15 - 15:30 IST                                                   ║")
    print(f"║  Monitoring: Continuous 5-minute candle analysis + Tick-by-tick BUY signals        ║")
    print(f"╚══════════════════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}")

    # Execute the integrated system
    results = main()

    if results:
        stage1_count = len(results['stage1_filtered'])
        initial_breakers_count = len(results['initial_breakers']) if not results['initial_breakers'].empty else 0
        stage_breakouts_count = len(results['stage_breakouts']) if not results['stage_breakouts'].empty else 0
        buy_signals_count = len(results['buy_signals'])

        print(f"\n{Fore.CYAN}╔══════════════════════════════════════════════════════════════════════════════════════╗")
        print(f"║                               FINAL EXECUTION SUMMARY                               ║")
        print(f"║                                                                                      ║")
        print(f"║  Stage 1 Filtered Stocks: {stage1_count:<58} ║")
        print(f"║  Initial Breakers: {initial_breakers_count:<67} ║")
        print(f"║  Stage Breakouts: {stage_breakouts_count:<68} ║")
        print(f"║  BUY Signals Generated: {buy_signals_count:<60} ║")
        print(f"║                                                                                      ║")
        print(f"║  System Status: {'COMPLETED SUCCESSFULLY':<61} ║")
        print(
            f"╚══════════════════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}")

        # DataFrames are available for further analysis:
        # - results['stage1_filtered']: Stage 1 filtered stocks
        # - results['initial_breakers']: Initial breakout candidates
        # - results['stage_breakouts']: Stage breakout candidates
        # - results['buy_signals']: Generated BUY signals
    else:
        print(f"\n{Fore.RED}╔══════════════════════════════════════════════════════════════════════════════════════╗")
        print(f"║                                 EXECUTION FAILED                                    ║")
        print(f"║                      Please check the error messages above                          ║")
        print(
            f"╚══════════════════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}")