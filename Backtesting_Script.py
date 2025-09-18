#!/usr/bin/env python3
"""
INTEGRATED STOCK FILTER AND HISTORICAL BACKTEST SYSTEM
======================================================
Stage 1: Enhanced Stock Filter
Stage 2: Historical Backtest on 1-minute data
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

    # Stage 2 - Historical Backtest
    MARKET_OPEN_TIME: str = "09:15"
    MARKET_CLOSE_TIME: str = "15:30"
    FIRST_CANDLE_END: str = "09:20"  # First 5-minute candle ends at 09:20
    CANDLE_INTERVAL: int = 5  # minutes for initial analysis
    REJECTION_THRESHOLD: float = 0.33  # 33% rejection rule

    # Backtest Settings
    BACKTEST_DAYS: int = 1  # Number of past trading days to backtest
    MINUTE_RESOLUTION: str = "1"  # 1-minute data for backtesting
    STOP_LOSS_PERCENT: float = 0.6  # 2% stop loss
    TARGET_PERCENT: float = 1.5  # 5% target


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

    def get_historical_minute_data(self, symbol, date):
        """Get historical 1-minute data for a specific date"""
        formatted_symbol = self.format_symbol(symbol)
        date_str = date.strftime("%Y-%m-%d")

        params = {
            "symbol": formatted_symbol,
            "resolution": config.MINUTE_RESOLUTION,
            "date_format": "1",
            "range_from": date_str,
            "range_to": date_str,
            "cont_flag": "1"
        }

        try:
            response = self.client.history(params)
            if response.get('code') != 200:
                return None, f"API Error: {response.get('message', 'Unknown')}"

            candles = response.get("candles", [])
            if not candles:
                return None, "No minute data available"

            data = []
            for candle in candles:
                timestamp = int(candle[0]) // 1000 if candle[0] > 1e12 else int(candle[0])
                dt = datetime.fromtimestamp(timestamp)

                # Filter only market hours
                time_str = dt.strftime("%H:%M")
                if "09:15" <= time_str <= "15:30":
                    data.append({
                        "datetime": dt,
                        "time": time_str,
                        "open": float(candle[1]),
                        "high": float(candle[2]),
                        "low": float(candle[3]),
                        "close": float(candle[4]),
                        "volume": float(candle[5]) if len(candle) > 5 else 0.0
                    })

            return pd.DataFrame(data).sort_values('datetime'), None

        except Exception as e:
            return None, str(e)

    def get_first_candle_data(self, symbol, date):
        """Get the first 5-minute candle (09:15-09:20) from minute data"""
        try:
            minute_df, error = self.get_historical_minute_data(symbol, date)
            if error or minute_df is None or minute_df.empty:
                return None, f"Minute data error: {error}"

            # Filter first 5 minutes (09:15 to 09:19)
            first_5_minutes = minute_df[
                (minute_df['time'] >= '09:15') & (minute_df['time'] <= '09:19')
                ]

            if first_5_minutes.empty:
                return None, "No data for first 5 minutes"

            # Create 5-minute candle from 1-minute data
            candle_open = first_5_minutes.iloc[0]['open']
            candle_high = first_5_minutes['high'].max()
            candle_low = first_5_minutes['low'].min()
            candle_close = first_5_minutes.iloc[-1]['close']
            candle_volume = first_5_minutes['volume'].sum()

            first_candle = {
                'open': candle_open,
                'high': candle_high,
                'low': candle_low,
                'close': candle_close,
                'volume': candle_volume,
                'datetime': first_5_minutes.iloc[0]['datetime']
            }

            return first_candle, minute_df

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
# STAGE 2 - HISTORICAL BACKTEST ENGINE
# ============================================================================

class HistoricalBacktestEngine:
    def __init__(self, fyers_data, filtered_stocks_df):
        self.fyers_data = fyers_data
        self.filtered_stocks_df = filtered_stocks_df
        self.backtest_results = []
        self.trade_log = []

    def get_trading_days(self, days_back):
        """Get list of trading days (excluding weekends)"""
        trading_days = []
        current_date = datetime.now().date()

        days_found = 0
        check_date = current_date

        while days_found < days_back:
            # Skip weekends (Saturday=5, Sunday=6)
            if check_date.weekday() < 5:
                trading_days.append(check_date)
                days_found += 1
            check_date -= timedelta(days=1)

        return list(reversed(trading_days))

    def analyze_initial_breakout_historical(self, symbol, pdh, date):
        """Analyze initial breakout for a specific historical date"""
        try:
            first_candle, minute_df = self.fyers_data.get_first_candle_data(symbol, date)
            if first_candle is None:
                return None, f"No first candle data for {date}"

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

            # Qualified for Initial Breakout - return for backtesting
            initial_data = {
                'symbol': symbol,
                'date': date,
                'pdh': pdh,
                'candle_open': candle_open,
                'candle_high': candle_high,
                'candle_low': candle_low,
                'candle_close': candle_close,
                'initial_entry_point': candle_high,
                'rejection_pct': rejection_percentage,
                'minute_data': minute_df
            }

            return "initial_breaker", initial_data

        except Exception as e:
            return None, f"Exception: {str(e)}"

    def backtest_trade(self, trade_data):
        """Backtest a single trade using 1-minute historical data"""
        try:
            symbol = trade_data['symbol']
            entry_point = trade_data['initial_entry_point']
            minute_df = trade_data['minute_data']
            date = trade_data['date']

            if minute_df is None or minute_df.empty:
                return None

            # Calculate stop loss and target
            stop_loss = entry_point * (1 - config.STOP_LOSS_PERCENT / 100)
            target = entry_point * (1 + config.TARGET_PERCENT / 100)

            # Filter data after 09:20 (after first candle completion)
            post_entry_data = minute_df[minute_df['time'] > '09:20'].copy()

            if post_entry_data.empty:
                return {
                    'symbol': symbol,
                    'date': date,
                    'entry_point': entry_point,
                    'entry_time': '09:20',
                    'exit_point': None,
                    'exit_time': None,
                    'result': 'NO_DATA',
                    'pnl_percent': 0,
                    'pnl_points': 0,
                    'stop_loss': stop_loss,
                    'target': target,
                    'max_profit': 0,
                    'max_loss': 0,
                    'trade_duration_minutes': 0
                }

            # Track trade progression
            entry_triggered = False
            entry_time = None
            exit_point = None
            exit_time = None
            result = 'NO_SIGNAL'
            max_profit_pct = 0
            max_loss_pct = 0

            for idx, row in post_entry_data.iterrows():
                current_price = row['high']  # Use high to check if entry point is breached
                current_time = row['time']

                # Check if entry point is triggered
                if not entry_triggered and current_price >= entry_point:
                    entry_triggered = True
                    entry_time = current_time
                    continue

                # If entry is triggered, monitor for exit conditions
                if entry_triggered:
                    high_price = row['high']
                    low_price = row['low']
                    close_price = row['close']

                    # Calculate current P&L
                    current_profit_pct = ((close_price - entry_point) / entry_point) * 100
                    max_profit_pct = max(max_profit_pct, ((high_price - entry_point) / entry_point) * 100)
                    max_loss_pct = min(max_loss_pct, ((low_price - entry_point) / entry_point) * 100)

                    # Check for stop loss hit
                    if low_price <= stop_loss:
                        exit_point = stop_loss
                        exit_time = current_time
                        result = 'STOP_LOSS'
                        break

                    # Check for target hit
                    if high_price >= target:
                        exit_point = target
                        exit_time = current_time
                        result = 'TARGET'
                        break

            # If no exit condition met, close at market close
            if entry_triggered and result not in ['STOP_LOSS', 'TARGET']:
                last_row = post_entry_data.iloc[-1]
                exit_point = last_row['close']
                exit_time = last_row['time']
                result = 'MARKET_CLOSE'

            # Calculate trade metrics
            if entry_triggered and exit_point:
                pnl_points = exit_point - entry_point
                pnl_percent = (pnl_points / entry_point) * 100

                # Calculate trade duration
                if entry_time and exit_time:
                    entry_dt = datetime.strptime(f"{date} {entry_time}", "%Y-%m-%d %H:%M")
                    exit_dt = datetime.strptime(f"{date} {exit_time}", "%Y-%m-%d %H:%M")
                    trade_duration = (exit_dt - entry_dt).total_seconds() / 60
                else:
                    trade_duration = 0
            else:
                pnl_points = 0
                pnl_percent = 0
                trade_duration = 0

            return {
                'symbol': symbol,
                'date': date,
                'entry_point': round(entry_point, 2),
                'entry_time': entry_time or 'NOT_TRIGGERED',
                'exit_point': round(exit_point, 2) if exit_point else None,
                'exit_time': exit_time,
                'result': result,
                'pnl_percent': round(pnl_percent, 2),
                'pnl_points': round(pnl_points, 2),
                'stop_loss': round(stop_loss, 2),
                'target': round(target, 2),
                'max_profit': round(max_profit_pct, 2),
                'max_loss': round(max_loss_pct, 2),
                'trade_duration_minutes': int(trade_duration)
            }

        except Exception as e:
            return {
                'symbol': trade_data['symbol'],
                'date': trade_data['date'],
                'result': 'ERROR',
                'error': str(e)
            }

    def run_backtest(self):
        """Run complete backtest on filtered stocks"""
        if self.filtered_stocks_df.empty:
            print(f"{Fore.RED}No stocks from Stage 1 to backtest{Style.RESET_ALL}")
            return

        trading_days = self.get_trading_days(config.BACKTEST_DAYS)
        total_stocks = len(self.filtered_stocks_df)
        total_combinations = total_stocks * len(trading_days)

        print(f"\n{Fore.BLUE}{'=' * 80}")
        print(f"STAGE 2 - HISTORICAL BACKTESTING")
        print(f"{'=' * 80}")
        print(f"Stocks to backtest: {total_stocks}")
        print(f"Trading days: {len(trading_days)} ({trading_days[0]} to {trading_days[-1]})")
        print(f"Total combinations: {total_combinations}")
        print(f"{'=' * 80}{Style.RESET_ALL}")

        processed = 0
        for _, stock in self.filtered_stocks_df.iterrows():
            symbol = stock['symbol']
            pdh = stock['pdh']

            print(f"\n{Fore.CYAN}Backtesting {symbol} (PDH: {pdh:.2f})...{Style.RESET_ALL}")

            stock_results = []
            for date in trading_days:
                processed += 1
                print(f"  {Fore.WHITE}[{processed:>3}/{total_combinations}] {date} - {symbol}{Style.RESET_ALL}", end="")

                # Analyze initial breakout for this date
                result, data = self.analyze_initial_breakout_historical(symbol, pdh, date)

                if result == "initial_breaker":
                    # Run backtest
                    trade_result = self.backtest_trade(data)
                    if trade_result:
                        stock_results.append(trade_result)
                        self.trade_log.append(trade_result)

                        if trade_result['result'] == 'TARGET':
                            print(f" {Fore.GREEN}✓ TARGET ({trade_result['pnl_percent']:+.1f}%){Style.RESET_ALL}")
                        elif trade_result['result'] == 'STOP_LOSS':
                            print(f" {Fore.RED}✗ STOP ({trade_result['pnl_percent']:+.1f}%){Style.RESET_ALL}")
                        elif trade_result['result'] == 'MARKET_CLOSE':
                            print(f" {Fore.YELLOW}→ CLOSE ({trade_result['pnl_percent']:+.1f}%){Style.RESET_ALL}")
                        else:
                            print(f" {Fore.MAGENTA}○ {trade_result['result']}{Style.RESET_ALL}")
                    else:
                        print(f" {Fore.RED}✗ BACKTEST_ERROR{Style.RESET_ALL}")
                else:
                    print(f" {Fore.YELLOW}○ {data if data else 'FILTERED'}{Style.RESET_ALL}")

                time.sleep(0.02)  # Small delay to prevent API throttling

            # Store results for this stock
            if stock_results:
                self.backtest_results.append({
                    'symbol': symbol,
                    'trades': stock_results,
                    'total_trades': len(stock_results),
                    'winning_trades': len([t for t in stock_results if t.get('pnl_percent', 0) > 0]),
                    'avg_pnl': np.mean([t.get('pnl_percent', 0) for t in stock_results])
                })

    def display_backtest_results(self):
        """Display comprehensive backtest results"""
        if not self.trade_log:
            print(f"\n{Fore.RED}No trades executed in backtest{Style.RESET_ALL}")
            return

        print(f"\n{Fore.CYAN}{'=' * 100}")
        print(f"{'BACKTEST RESULTS SUMMARY':^100}")
        print(f"{'=' * 100}{Style.RESET_ALL}")

        # Overall Statistics
        total_trades = len(self.trade_log)
        winning_trades = len([t for t in self.trade_log if t.get('pnl_percent', 0) > 0])
        losing_trades = len([t for t in self.trade_log if t.get('pnl_percent', 0) < 0])
        breakeven_trades = len([t for t in self.trade_log if t.get('pnl_percent', 0) == 0])

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        total_pnl = sum([t.get('pnl_percent', 0) for t in self.trade_log])
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0

        target_hits = len([t for t in self.trade_log if t.get('result') == 'TARGET'])
        stop_hits = len([t for t in self.trade_log if t.get('result') == 'STOP_LOSS'])
        market_close = len([t for t in self.trade_log if t.get('result') == 'MARKET_CLOSE'])

        print(f"\n{Fore.WHITE}OVERALL PERFORMANCE:")
        print(f"{'=' * 50}")
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {Fore.GREEN}{winning_trades}{Style.RESET_ALL} ({win_rate:.1f}%)")
        print(f"Losing Trades: {Fore.RED}{losing_trades}{Style.RESET_ALL}")
        print(f"Breakeven Trades: {Fore.YELLOW}{breakeven_trades}{Style.RESET_ALL}")
        print(f"")
        print(f"Target Hits: {Fore.GREEN}{target_hits}{Style.RESET_ALL}")
        print(f"Stop Loss Hits: {Fore.RED}{stop_hits}{Style.RESET_ALL}")
        print(f"Market Close: {Fore.YELLOW}{market_close}{Style.RESET_ALL}")
        print(f"")
        print(f"Total P&L: {Fore.GREEN if total_pnl > 0 else Fore.RED}{total_pnl:+.2f}%{Style.RESET_ALL}")
        print(f"Average P&L per Trade: {Fore.GREEN if avg_pnl > 0 else Fore.RED}{avg_pnl:+.2f}%{Style.RESET_ALL}")

        # Best and Worst Trades
        if self.trade_log:
            pnl_trades = [t for t in self.trade_log if 'pnl_percent' in t]
            if pnl_trades:
                best_trade = max(pnl_trades, key=lambda x: x['pnl_percent'])
                worst_trade = min(pnl_trades, key=lambda x: x['pnl_percent'])

                print(f"\n{Fore.GREEN}BEST TRADE:")
                print(
                    f"{best_trade['symbol']} on {best_trade['date']} - {best_trade['pnl_percent']:+.2f}% ({best_trade['result']})")

                print(f"\n{Fore.RED}WORST TRADE:")
                print(
                    f"{worst_trade['symbol']} on {worst_trade['date']} - {worst_trade['pnl_percent']:+.2f}% ({worst_trade['result']})")

        # Stock-wise Performance
        print(f"\n{Fore.YELLOW}STOCK-WISE PERFORMANCE:")
        print(f"{'=' * 80}")
        print(f"{'Symbol':<12} {'Trades':<7} {'Win%':<6} {'Avg P&L':<8} {'Total P&L':<10} {'Best':<7} {'Worst'}")
        print("-" * 80)

        for result in self.backtest_results:
            symbol = result['symbol']
            trades = result['trades']
            total_trades = len(trades)

            if total_trades > 0:
                winning = len([t for t in trades if t.get('pnl_percent', 0) > 0])
                win_rate = (winning / total_trades * 100)
                avg_pnl = np.mean([t.get('pnl_percent', 0) for t in trades])
                total_pnl = sum([t.get('pnl_percent', 0) for t in trades])
                best_trade = max(trades, key=lambda x: x.get('pnl_percent', 0))
                worst_trade = min(trades, key=lambda x: x.get('pnl_percent', 0))

                color = Fore.GREEN if avg_pnl > 0 else Fore.RED
                print(f"{symbol:<12} {total_trades:<7} {win_rate:<6.1f} {color}{avg_pnl:<8.2f}{Style.RESET_ALL} "
                      f"{color}{total_pnl:<10.2f}{Style.RESET_ALL} {best_trade['pnl_percent']:<7.2f} {worst_trade['pnl_percent']}")

        # Recent Trades Detail
        print(f"\n{Fore.MAGENTA}RECENT TRADES DETAIL (Last 10):")
        print(f"{'=' * 100}")
        print(f"{'Date':<12} {'Symbol':<12} {'Entry':<8} {'Exit':<8} {'Result':<12} {'P&L%':<8} {'Duration'}")
        print("-" * 100)

        recent_trades = sorted(self.trade_log, key=lambda x: x['date'], reverse=True)[:10]
        for trade in recent_trades:
            result_color = Fore.GREEN if trade.get('result') == 'TARGET' else \
                Fore.RED if trade.get('result') == 'STOP_LOSS' else Fore.YELLOW
            pnl_color = Fore.GREEN if trade.get('pnl_percent', 0) > 0 else Fore.RED

            print(f"{trade['date']:<12} {trade['symbol']:<12} {trade['entry_point']:<8} "
                  f"{trade.get('exit_point', 'N/A'):<8} {result_color}{trade['result']:<12}{Style.RESET_ALL} "
                  f"{pnl_color}{trade.get('pnl_percent', 0):<8.2f}{Style.RESET_ALL} "
                  f"{trade.get('trade_duration_minutes', 0)} min")

        # Save results to CSV
        self.save_results_to_csv()

    def save_results_to_csv(self):
        """Save backtest results to CSV file"""
        try:
            if self.trade_log:
                df = pd.DataFrame(self.trade_log)
                filename = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df.to_csv(filename, index=False)
                print(f"\n{Fore.GREEN}Results saved to: {filename}{Style.RESET_ALL}")
        except Exception as e:
            print(f"\n{Fore.RED}Error saving results: {e}{Style.RESET_ALL}")


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
    """Main integrated function for Stage 1 + Stage 2 Historical Backtest"""
    print(f"\n{Fore.CYAN}INTEGRATED STOCK FILTER AND HISTORICAL BACKTEST SYSTEM")
    print(f"Stage 1: Enhanced Stock Filtering")
    print(f"Stage 2: Historical Backtesting on 1-minute data")
    print(f"========================================================{Style.RESET_ALL}")

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

        # ==================== STAGE 2: HISTORICAL BACKTESTING ====================

        print(f"\n{Fore.YELLOW}STAGE 2: Starting Historical Backtesting...{Style.RESET_ALL}")

        # Initialize backtest engine
        backtest_engine = HistoricalBacktestEngine(fyers_data, stage1_filtered_df)

        # Run backtest
        print(f"\n{Fore.BLUE}Backtest Configuration:")
        print(f"• Days to backtest: {config.BACKTEST_DAYS}")
        print(f"• Resolution: {config.MINUTE_RESOLUTION}-minute data")
        print(f"• Stop Loss: {config.STOP_LOSS_PERCENT}%")
        print(f"• Target: {config.TARGET_PERCENT}%")
        print(f"• Rejection Threshold: {config.REJECTION_THRESHOLD * 100}%")

        backtest_engine.run_backtest()

        # Display results
        backtest_engine.display_backtest_results()

        print(f"\n{Fore.GREEN}✓ Historical backtesting completed successfully!{Style.RESET_ALL}")

        return {
            'stage1_filtered': stage1_filtered_df,
            'backtest_results': backtest_engine.backtest_results,
            'trade_log': backtest_engine.trade_log,
            'total_trades': len(backtest_engine.trade_log),
            'winning_trades': len([t for t in backtest_engine.trade_log if t.get('pnl_percent', 0) > 0]),
            'total_pnl': sum([t.get('pnl_percent', 0) for t in backtest_engine.trade_log])
        }

    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Process interrupted by user{Style.RESET_ALL}")
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
    print(f"║                    INTEGRATED STOCK FILTER & HISTORICAL BACKTEST                    ║")
    print(f"║                                                                                      ║")
    print(f"║  Stage 1: Enhanced Stock Filtering (PDH > 2DH + EMA + Bullish Conditions)          ║")
    print(f"║  Stage 2: Historical Backtesting on 1-minute data ({config.BACKTEST_DAYS} trading days)                ║")
    print(f"║                                                                                      ║")
    print(f"║  Backtest Settings:                                                                  ║")
    print(
        f"║  • Resolution: {config.MINUTE_RESOLUTION}-minute historical data                                             ║")
    print(
        f"║  • Stop Loss: {config.STOP_LOSS_PERCENT}% | Target: {config.TARGET_PERCENT}%                                              ║")
    print(
        f"║  • Rejection Rule: {config.REJECTION_THRESHOLD * 100}% maximum rejection in first candle                      ║")
    print(f"╚══════════════════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}")

    # Execute the integrated system
    results = main()

    if results:
        stage1_count = len(results['stage1_filtered'])
        total_trades = results['total_trades']
        winning_trades = results['winning_trades']
        total_pnl = results['total_pnl']
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        print(f"\n{Fore.CYAN}╔══════════════════════════════════════════════════════════════════════════════════════╗")
        print(f"║                            FINAL BACKTEST SUMMARY                                   ║")
        print(f"║                                                                                      ║")
        print(f"║  Stage 1 Filtered Stocks: {stage1_count:<58} ║")
        print(f"║  Total Trades Executed: {total_trades:<60} ║")
        print(f"║  Winning Trades: {winning_trades:<65} ║")
        print(f"║  Win Rate: {win_rate:<70.1f}% ║")
        print(f"║  Total P&L: {total_pnl:<69.2f}% ║")
        print(f"║                                                                                      ║")
        print(f"║  System Status: {'BACKTEST COMPLETED SUCCESSFULLY':<56} ║")
        print(f"║  Results saved to CSV file for detailed analysis                                    ║")
        print(
            f"╚══════════════════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}")

        # Print key insights
        if total_trades > 0:
            avg_pnl = total_pnl / total_trades
            print(f"\n{Fore.YELLOW}KEY INSIGHTS:")
            print(f"• Average P&L per trade: {avg_pnl:+.2f}%")
            print(f"• Total backtest period: {config.BACKTEST_DAYS} trading days")
            print(f"• Risk-Reward ratio: 1:{config.TARGET_PERCENT / config.STOP_LOSS_PERCENT:.1f}")

            if win_rate > 50:
                print(f"• {Fore.GREEN}Positive win rate indicates potentially profitable strategy{Style.RESET_ALL}")
            else:
                print(f"• {Fore.RED}Win rate below 50% - strategy may need refinement{Style.RESET_ALL}")

        # Available data for further analysis:
        # - results['stage1_filtered']: Stage 1 filtered stocks DataFrame
        # - results['backtest_results']: Stock-wise backtest results
        # - results['trade_log']: Individual trade details
    else:
        print(f"\n{Fore.RED}╔══════════════════════════════════════════════════════════════════════════════════════╗")
        print(f"║                               BACKTEST FAILED                                       ║")
        print(f"║                      Please check the error messages above                          ║")
        print(
            f"╚══════════════════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}")
