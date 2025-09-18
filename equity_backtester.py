#!/usr/bin/env python3
"""
INTEGRATED EQUITY BACKTESTING SYSTEM
====================================
Enhanced version with professional output and detailed reporting
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from colorama import Fore, Style, init
from tabulate import tabulate
import warnings
import logging
import re

warnings.filterwarnings('ignore')

# Import settings from the config file
import config as cfg

# Custom Formatter to strip ANSI escape codes for clean file logging
class StripAnsiFormatter(logging.Formatter):
    """A custom formatter to strip ANSI escape codes from log messages."""
    def format(self, record):
        message = super().format(record)
        # Regex to remove ANSI escape codes for color, etc.
        return re.sub(r'\x1b\[([0-9]{1,2}(;[0-9]{1,2})?)?[m|K]', '', message)

# Initialize colorama
init(autoreset=True)

# Try to import Fyers SDK
try:
    from fyers_apiv3 import fyersModel
except ImportError:
    print(f"{Fore.RED}Error: No Fyers SDK found. Please install: pip install fyers-apiv3")
    sys.exit(1)


# ============================================================================
# ENHANCED LOGGING & REPORTING UTILITIES
# ============================================================================
class ReportFormatter:
    """Professional report formatting utilities"""

    @staticmethod
    def print_header(title: str, subtitle: str = ""):
        """Print a professional header"""
        width = 80
        logging.info(f"\n{Fore.CYAN}{'═' * width}")
        logging.info(f"  {title.upper()}")
        if subtitle:
            logging.info(f"  {subtitle}")
        logging.info(f"{'═' * width}{Style.RESET_ALL}")

    @staticmethod
    def print_section(title: str):
        """Print a section header"""
        logging.info(f"\n{Fore.BLUE}▶ {title}{Style.RESET_ALL}")
        logging.info(f"{Fore.BLUE}{'─' * (len(title) + 3)}{Style.RESET_ALL}")

    @staticmethod
    def print_status(message: str, status: str = "INFO"):
        """Print formatted status message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        colors = {
            "INFO": Fore.BLUE,
            "SUCCESS": Fore.GREEN,
            "WARNING": Fore.YELLOW,
            "ERROR": Fore.MAGENTA, # Changed from RED
            "TRADE": Fore.CYAN      # Changed from MAGENTA
        }
        color = colors.get(status, Fore.WHITE)
        logging.info(f"{Fore.WHITE}[{timestamp}] {color}[{status}]{Style.RESET_ALL} {message}")

    @staticmethod
    def format_currency(amount: float, decimals: int = 2) -> str:
        """Format currency with proper symbols"""
        return f"₹{amount:,.{decimals}f}"

    @staticmethod
    def format_percentage(pct: float, decimals: int = 2) -> str:
        """Format percentage with color coding"""
        color = Fore.GREEN if pct >= 0 else Fore.MAGENTA
        return f"{color}{pct:+.{decimals}f}%{Style.RESET_ALL}"


class PerformanceTracker:
    """Track system performance and statistics"""

    def __init__(self):
        self.start_time = time.time()
        self.api_calls = 0
        self.processed_symbols = 0
        self.filtered_symbols = 0

    def log_api_call(self):
        self.api_calls += 1

    def log_symbol_processed(self):
        self.processed_symbols += 1

    def log_symbol_filtered(self):
        self.filtered_symbols += 1

    def get_runtime(self) -> str:
        runtime = time.time() - self.start_time
        return f"{runtime:.2f}s"

    def get_stats(self) -> Dict:
        return {
            "Runtime": self.get_runtime(),
            "API Calls": self.api_calls,
            "Symbols Processed": self.processed_symbols,
            "Symbols Filtered": self.filtered_symbols,
            "Filter Rate": f"{(self.filtered_symbols / max(self.processed_symbols, 1) * 100):.1f}%"
        }


# Global performance tracker
perf_tracker = PerformanceTracker()


# ============================================================================
# AUTHENTICATION (Enhanced)
# ============================================================================
class FyersAuth:
    def __init__(self):
        self.access_token = None
        self.token_file = cfg.ACCESS_TOKEN_FILE

    def load_saved_token(self):
        """Load saved access token from file if it is valid."""
        try:
            if os.path.exists(self.token_file):
                with open(self.token_file, 'r') as f:
                    data = json.load(f)
                token = data.get('access_token')
                expiry = data.get('expiry', 0)
                if token and time.time() + 600 < expiry:
                    remaining_hours = (expiry - time.time()) / 3600
                    ReportFormatter.print_status(f"Using saved token (expires in {remaining_hours:.1f} hours)",
                                                 "SUCCESS")
                    return token
                else:
                    ReportFormatter.print_status("Saved token has expired", "WARNING")
        except Exception as e:
            ReportFormatter.print_status(f"Error loading saved token: {e}", "ERROR")
        return None

    def save_token(self, token):
        """Save the access token and its expiry time to a file."""
        try:
            expiry_time = int(time.time()) + (23 * 60 * 60)
            data = {'access_token': token, 'expiry': expiry_time, 'created_at': int(time.time())}
            with open(self.token_file, 'w') as f:
                json.dump(data, f, indent=2)
            ReportFormatter.print_status(f"Access token saved to {self.token_file}", "SUCCESS")
        except Exception as e:
            ReportFormatter.print_status(f"Failed to save access token: {e}", "ERROR")

    def authenticate(self):
        """Orchestrates the Fyers authentication process."""
        ReportFormatter.print_section("Authentication")

        saved_token = self.load_saved_token()
        if saved_token:
            self.access_token = saved_token
            return self.access_token

        ReportFormatter.print_status("Interactive authentication required", "INFO")
        try:
            session = fyersModel.SessionModel(client_id=cfg.CLIENT_ID, secret_key=cfg.SECRET_KEY,
                                              redirect_uri=cfg.REDIRECT_URI, response_type=cfg.RESPONSE_TYPE,
                                              grant_type=cfg.GRANT_TYPE)
            auth_url = session.generate_authcode()

            # Mask the client_id in the URL before printing/logging
            masked_auth_url = re.sub(r'client_id=([^&]+)', 'client_id=********', auth_url)

            # For interactive parts, we print to console and also log to file
            print(f"\n{Fore.CYAN}Steps to authenticate:")
            logging.info("\nSteps to authenticate:")
            print(f"1. Copy and open this URL in your browser:")
            logging.info("1. Copy and open this URL in your browser:")
            print(f"   {masked_auth_url}")
            logging.info(f"   {masked_auth_url}")
            print(f"2. Complete Fyers login and authorization")
            logging.info("2. Complete Fyers login and authorization")
            print(f"3. Copy the 'auth_code' from the redirect URL{Style.RESET_ALL}")
            logging.info("3. Copy the 'auth_code' from the redirect URL")

            auth_code = input(f"\n{Fore.YELLOW}Enter auth_code: {Style.RESET_ALL}").strip()

            if not auth_code:
                ReportFormatter.print_status("No authorization code provided", "ERROR")
                return None

            session.set_token(auth_code)
            response = session.generate_token()

            if response and response.get("access_token"):
                access_token = response["access_token"]
                ReportFormatter.print_status("Authentication successful", "SUCCESS")
                self.save_token(access_token)
                self.access_token = access_token
                return self.access_token
            else:
                ReportFormatter.print_status(f"Authentication failed: {response.get('message', 'Unknown error')}",
                                             "ERROR")
                return None
        except Exception as e:
            ReportFormatter.print_status(f"Authentication error: {e}", "ERROR")
            return None


# ============================================================================
# ENHANCED DATA FETCHER
# ============================================================================
class FyersData:
    def __init__(self, access_token):
        self.client = fyersModel.FyersModel(client_id=cfg.CLIENT_ID, token=access_token)

    def format_symbol(self, symbol):
        """Format symbol for Fyers API"""
        if not symbol.startswith('NSE:') or not symbol.endswith('-EQ'):
            return f"NSE:{symbol.replace('&', '%26')}-EQ"
        return symbol

    def get_daily_data(self, symbol, days_back=30):
        """Get daily OHLCV data with enhanced error handling."""
        perf_tracker.log_api_call()
        params = {"symbol": self.format_symbol(symbol), "resolution": "D", "date_format": "1",
                  "range_from": (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d"),
                  "range_to": datetime.now().strftime("%Y-%m-%d"), "cont_flag": "1"}
        try:
            response = self.client.history(params)
            if response.get('code') != 200:
                return None, f"API Error: {response.get('message', 'Unknown')}"

            candles = response.get("candles", [])
            if not candles or len(candles) < cfg.EMA_LENGTH + 3:
                return None, f"Insufficient data: {len(candles)} candles (need {cfg.EMA_LENGTH + 3})"

            data = [{'date': datetime.fromtimestamp(c[0]).date(), 'open': float(c[1]), 'high': float(c[2]),
                     'low': float(c[3]), 'close': float(c[4]), 'volume': float(c[5])} for c in candles]
            df = pd.DataFrame(data).sort_values('date').reset_index(drop=True)
            df[f'ema_{cfg.EMA_LENGTH}'] = df['close'].ewm(span=cfg.EMA_LENGTH, adjust=False).mean()
            return df, None
        except Exception as e:
            return None, str(e)

    def get_intraday_data_for_day(self, symbol, date, resolution):
        """Get historical intraday data for a specific date and resolution."""
        perf_tracker.log_api_call()
        params = {"symbol": self.format_symbol(symbol), "resolution": resolution, "date_format": "1",
                  "range_from": date.strftime("%Y-%m-%d"), "range_to": date.strftime("%Y-%m-%d"), "cont_flag": "1"}
        try:
            response = self.client.history(params)
            if response.get('code') != 200:
                return None, f"API Error: {response.get('message', 'Unknown')}"

            candles = response.get("candles", [])
            if not candles:
                return None, "No intraday data available"

            data = []
            for c in candles:
                dt = datetime.fromtimestamp(c[0])
                if cfg.MARKET_OPEN_TIME <= dt.strftime("%H:%M") <= cfg.DAY_END_SQUARE_OFF:
                    data.append({'datetime': dt, 'time': dt.strftime("%H:%M"), 'open': float(c[1]),
                                 'high': float(c[2]), 'low': float(c[3]), 'close': float(c[4]), 'volume': float(c[5])})
            return pd.DataFrame(data).sort_values('datetime'), None
        except Exception as e:
            return None, str(e)

    def get_bulk_intraday_data(self, symbols, date, resolution):
        """Fetch intraday data for multiple symbols with progress tracking."""
        all_data = {}
        total_symbols = len(symbols)
        successful_fetches = 0
        failed_fetches = 0

        ReportFormatter.print_status(
            f"Fetching {resolution}-min data for {total_symbols} symbols on {date.strftime('%Y-%m-%d')}", "INFO")

        for i, symbol in enumerate(symbols, 1):
            # Progress indicator - console only
            progress_pct = (i / total_symbols) * 100
            print(
                f"\r{Fore.WHITE}Progress: [{progress_pct:5.1f}%] {i:>3}/{total_symbols} - {symbol:<15}{Style.RESET_ALL}",
                end="")

            df, error = self.get_intraday_data_for_day(symbol, date, resolution)
            if error:
                failed_fetches += 1
                all_data[symbol] = pd.DataFrame()
            else:
                successful_fetches += 1
                all_data[symbol] = df
            time.sleep(0.2)  # Rate limit

        print()  # New line after progress
        ReportFormatter.print_status(f"Data fetch complete: {successful_fetches} successful, {failed_fetches} failed",
                                     "SUCCESS")
        return all_data


# ============================================================================
# ENHANCED PRE-FILTER ENGINE
# ============================================================================
class PreFilterEngine:
    def __init__(self, fyers_data):
        self.fyers_data = fyers_data
        self.filter_stats = {
            'total_processed': 0,
            'passed_filter': 0,
            'rejection_reasons': {}
        }

    def filter_stock(self, symbol):
        """Enhanced filtering with detailed rejection tracking."""
        self.filter_stats['total_processed'] += 1
        perf_tracker.log_symbol_processed()

        try:
            daily_df, error = self.fyers_data.get_daily_data(symbol, days_back=45)
            if error or daily_df is None or len(daily_df) < 3:
                self._track_rejection("Insufficient Data", error or "Not enough historical data")
                return None, error or "Insufficient data"

            latest, previous, two_days_ago = daily_df.iloc[-1], daily_df.iloc[-2], daily_df.iloc[-3]
            ema_col = f'ema_{cfg.EMA_LENGTH}'

            if ema_col not in daily_df.columns:
                self._track_rejection("Technical Error", "EMA calculation failed")
                return None, "EMA not calculated"

            ema_1day, ema_2days = previous[ema_col], two_days_ago[ema_col]

            # Enhanced filtering with specific criteria tracking
            filter_checks = [
                ("Price Range", cfg.MIN_PRICE <= latest['close'] <= cfg.MAX_PRICE,
                 f"Price {latest['close']:.2f} not in range [{cfg.MIN_PRICE}-{cfg.MAX_PRICE}]"),
                ("EMA Trend", ema_1day > ema_2days, f"EMA declining: {ema_1day:.2f} <= {ema_2days:.2f}"),
                ("Higher High", previous['high'] > two_days_ago['high'],
                 f"PDH {previous['high']:.2f} <= 2DH {two_days_ago['high']:.2f}"),
                ("Bullish Candle", previous['close'] > previous['open'],
                 f"Bearish PD candle: C{previous['close']:.2f} <= O{previous['open']:.2f}"),
                ("Strength", previous['close'] > two_days_ago['open'],
                 f"PD Close {previous['close']:.2f} <= 2D Open {two_days_ago['open']:.2f}"),
                ("EMA Support", previous['low'] > ema_1day, f"PD Low {previous['low']:.2f} <= EMA {ema_1day:.2f}")
            ]

            failed_checks = [(name, reason) for name, check, reason in filter_checks if not check]

            if failed_checks:
                for name, reason in failed_checks:
                    self._track_rejection(name, reason)
                return None, "; ".join([reason for _, reason in failed_checks])

            self.filter_stats['passed_filter'] += 1
            perf_tracker.log_symbol_filtered()

            return {
                'symbol': symbol,
                'pdh': round(previous['high'], 2),
                'price': round(latest['close'], 2),
                'ema': round(ema_1day, 2),
                'volume': int(previous['volume'])
            }, None

        except Exception as e:
            self._track_rejection("System Error", str(e))
            return None, f"Exception: {e}"

    def _track_rejection(self, category, reason):
        """Track rejection reasons for analysis."""
        if category not in self.filter_stats['rejection_reasons']:
            self.filter_stats['rejection_reasons'][category] = []
        self.filter_stats['rejection_reasons'][category].append(reason)

    def process_symbols(self, symbols: List[str]) -> Tuple[List[Dict], List[Dict]]:
        """Process symbols with enhanced progress tracking and reporting."""
        ReportFormatter.print_section("Pre-Market Filtering")
        ReportFormatter.print_status(
            f"Analyzing {len(symbols)} symbols against {len([c for c, _, _ in [('Price Range', True, ''), ('EMA Trend', True, ''), ('Higher High', True, ''), ('Bullish Candle', True, ''), ('Strength', True, ''), ('EMA Support', True, '')]])} criteria",
            "INFO")

        filtered_stocks, failed_stocks = [], []
        start_time = time.time()

        for i, symbol in enumerate(symbols, 1):
            progress_pct = (i / len(symbols)) * 100
            elapsed = time.time() - start_time
            eta = (elapsed / i) * (len(symbols) - i) if i > 0 else 0

            # Progress indicator - console only
            print(f"\r{Fore.WHITE}[{progress_pct:5.1f}%] Processing {symbol:<15} | ETA: {eta:4.0f}s{Style.RESET_ALL}",
                  end="")

            stock_data, error = self.filter_stock(symbol)
            if stock_data:
                filtered_stocks.append(stock_data)
            else:
                failed_stocks.append({'symbol': symbol, 'reason': error})
            time.sleep(0.1)

        print()  # New line after progress

        # Print filtering summary
        self._print_filter_summary(filtered_stocks)
        return filtered_stocks, failed_stocks

    def _print_filter_summary(self, filtered_stocks):
        """Print detailed filtering summary."""
        total = self.filter_stats['total_processed']
        passed = self.filter_stats['passed_filter']
        filter_rate = (passed / total * 100) if total > 0 else 0

        ReportFormatter.print_status(f"Pre-filtering complete: {passed}/{total} stocks passed ({filter_rate:.1f}%)",
                                     "SUCCESS")

        # Show top rejection reasons
        if self.filter_stats['rejection_reasons']:
            logging.info(f"\n{Fore.BLUE}Top Rejection Categories:{Style.RESET_ALL}")
            rejection_counts = {cat: len(reasons) for cat, reasons in self.filter_stats['rejection_reasons'].items()}
            for category, count in sorted(rejection_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                logging.info(f"  • {category:<15}: {count:>3} stocks")

        # Show qualified stocks summary
        if filtered_stocks:
            logging.info(f"\n{Fore.GREEN}Qualified Stocks for Intraday Analysis:{Style.RESET_ALL}")
            table_data = []
            for stock in filtered_stocks[:10]:  # Show top 10
                table_data.append([
                    stock['symbol'],
                    ReportFormatter.format_currency(stock['price']),
                    ReportFormatter.format_currency(stock['pdh']),
                    ReportFormatter.format_currency(stock['ema']),
                    f"{stock['volume']:,}"
                ])

            headers = ["Symbol", "LTP", "PDH", f"EMA-{cfg.EMA_LENGTH}", "Volume"]
            logging.info(tabulate(table_data, headers=headers, tablefmt="rounded_grid"))

            if len(filtered_stocks) > 10:
                logging.info(f"  ... and {len(filtered_stocks) - 10} more stocks")


# ============================================================================
# ENHANCED BREAKOUT SCANNER & ORDER MANAGER
# ============================================================================
class BreakoutScanner:
    def __init__(self):
        self.identified_stocks = set()
        self.scan_stats = {
            'initial_scans': 0,
            'stage_scans': 0,
            'breakouts_found': 0
        }

    def scan(self, stock_data, five_min_candle, pdh):
        """Enhanced scanning with detailed logging."""
        symbol = stock_data['symbol']
        if symbol in self.identified_stocks:
            return None

        candle_time = five_min_candle['time']
        candle_open, candle_high, candle_close = five_min_candle['open'], five_min_candle['high'], five_min_candle[
            'close']
        candle_low = five_min_candle['low']

        # Initial Breakout Scan (opening candle)
        if candle_time == cfg.MARKET_OPEN_TIME:
            self.scan_stats['initial_scans'] += 1
            if candle_open < pdh and candle_close > candle_open:
                rejection = (candle_high - candle_close) / (candle_high - candle_low) if (
                                                                                                     candle_high - candle_low) > 0 else 0
                if rejection <= cfg.INITIAL_REJECTION_THRESHOLD:
                    self.identified_stocks.add(symbol)
                    self.scan_stats['breakouts_found'] += 1

                    ReportFormatter.print_status(
                        f"Initial breakout detected: {symbol} | Entry: {candle_high:.2f} | Rejection: {rejection:.1%}",
                        "TRADE"
                    )

                    return {
                        'symbol': symbol,
                        'entry_point': candle_high,
                        'type': 'Initial',
                        'scan_time': candle_time,
                        'candle_data': {
                            'open': candle_open,
                            'high': candle_high,
                            'low': candle_low,
                            'close': candle_close,
                            'rejection_pct': rejection
                        }
                    }

        # Stage Breakout Scan (any candle)
        if candle_open < pdh and candle_close >= pdh:
            self.scan_stats['stage_scans'] += 1
            self.identified_stocks.add(symbol)
            self.scan_stats['breakouts_found'] += 1

            ReportFormatter.print_status(
                f"Stage breakout detected: {symbol} at {candle_time} | Entry: {candle_high:.2f} | PDH: {pdh:.2f}",
                "TRADE"
            )

            return {
                'symbol': symbol,
                'entry_point': candle_high,
                'type': 'Stage',
                'scan_time': candle_time,
                'candle_data': {
                    'open': candle_open,
                    'high': candle_high,
                    'low': candle_low,
                    'close': candle_close
                }
            }

        return None

    def get_stats(self):
        """Return scanner statistics."""
        return self.scan_stats


class OrderManager:
    def __init__(self):
        self.trade_log = []
        self.active_trades = []
        self.trade_stats = {
            'total_signals': 0,
            'triggered_trades': 0,
            'pending_trades': 0,
            'closed_trades': 0
        }

    def add_potential_trade(self, trade_idea, date):
        """Add trade with enhanced tracking and logging."""
        trade_type = trade_idea['type']
        entry_price = trade_idea['entry_point']
        sl_pct = cfg.INITIAL_SL_PERCENT if trade_type == 'Initial' else cfg.STAGE_SL_PERCENT
        target_pct = cfg.INITIAL_TARGET_PERCENT if trade_type == 'Initial' else cfg.STAGE_TARGET_PERCENT

        trade = {
            'symbol': trade_idea['symbol'],
            'type': trade_type,
            'date': date.strftime('%Y-%m-%d'),
            'signal_time': trade_idea['scan_time'],
            'entry_point': entry_price,
            'status': 'pending',
            'stop_loss': round(entry_price * (1 - sl_pct / 100), 2),
            'target': round(entry_price * (1 + target_pct / 100), 2),
            'sl_percent': sl_pct,
            'target_percent': target_pct,
            'candle_data': trade_idea.get('candle_data', {}),
            'risk_reward': round(target_pct / sl_pct, 2)
        }

        self.active_trades.append(trade)
        self.trade_stats['total_signals'] += 1
        self.trade_stats['pending_trades'] += 1

        ReportFormatter.print_status(
            f"Trade setup: {trade['symbol']} ({trade_type}) | Entry: {ReportFormatter.format_currency(entry_price)} | R:R = 1:{trade['risk_reward']}",
            "INFO"
        )

    def process_active_trades(self, minute_candle):
        """Enhanced trade processing with detailed decision logging."""
        if not self.active_trades:
            return

        current_time = None
        for symbol, candle in minute_candle.items():
            current_time = candle['time']
            break

        for trade in self.active_trades:
            if trade['status'] == 'closed':
                continue

            symbol = trade['symbol']
            if symbol not in minute_candle:
                continue

            current_candle = minute_candle[symbol]
            high, low = current_candle['high'], current_candle['low']
            current_close = current_candle['close']
            current_time = current_candle['time']

            # Process pending trades for spurt-based entry
            if trade['status'] == 'pending':
                spurt_detected = current_candle.get('Spurt_Signal', False)
                if high >= trade['entry_point'] and spurt_detected:
                    trade['status'] = 'active'
                    trade['entry_time'] = current_time
                    trade['actual_entry_price'] = trade['entry_point']

                    self.trade_stats['triggered_trades'] += 1
                    self.trade_stats['pending_trades'] -= 1

                    ReportFormatter.print_status(
                        f"ENTRY: {symbol} at {current_time} | Price: {ReportFormatter.format_currency(trade['entry_point'])} | Spurt confirmed",
                        "TRADE"
                    )

            # Process active trades for exit conditions
            elif trade['status'] == 'active':
                exit_reason, exit_price = None, None

                if low <= trade['stop_loss']:
                    exit_reason, exit_price = 'STOP_LOSS', trade['stop_loss']
                elif high >= trade['target']:
                    exit_reason, exit_price = 'TARGET', trade['target']
                elif current_time >= cfg.DAY_END_SQUARE_OFF:
                    exit_reason, exit_price = 'EOD_SQUARE_OFF', current_close

                if exit_reason:
                    self._close_trade(trade, current_time, exit_price, exit_reason)

    def _close_trade(self, trade, exit_time, exit_price, exit_reason):
        """Close trade with detailed logging."""
        trade['status'] = 'closed'
        trade['exit_time'] = exit_time
        trade['exit_price'] = exit_price
        trade['exit_reason'] = exit_reason
        trade['pnl_points'] = round(exit_price - trade['entry_point'], 2)
        trade['pnl_percent'] = round((trade['pnl_points'] / trade['entry_point']) * 100, 2)

        # Calculate trade duration
        if 'entry_time' in trade:
            entry_dt = datetime.strptime(f"{trade['date']} {trade['entry_time']}", "%Y-%m-%d %H:%M")
            exit_dt = datetime.strptime(f"{trade['date']} {exit_time}", "%Y-%m-%d %H:%M")
            trade['duration_minutes'] = int((exit_dt - entry_dt).total_seconds() / 60)

        self.trade_log.append(trade)
        self.trade_stats['closed_trades'] += 1

        # Enhanced exit logging
        pnl_color = Fore.GREEN if trade['pnl_percent'] > 0 else Fore.RED
        ReportFormatter.print_status(
            f"EXIT: {trade['symbol']} | {exit_reason} | P&L: {ReportFormatter.format_percentage(trade['pnl_percent'])} | Duration: {trade.get('duration_minutes', 'N/A')}min",
            "TRADE"
        )

    def square_off_open_trades(self, last_candles):
        """Enhanced EOD square-off with proper logging."""
        squared_off = 0
        for trade in self.active_trades:
            if trade['status'] == 'active':
                symbol = trade['symbol']
                if symbol in last_candles:
                    exit_price = last_candles[symbol]['close']
                    self._close_trade(trade, cfg.DAY_END_SQUARE_OFF, exit_price, 'EOD_SQUARE_OFF')
                    squared_off += 1

        if squared_off > 0:
            ReportFormatter.print_status(f"EOD Square-off completed for {squared_off} active positions", "INFO")

    def generate_report(self, output_folder: str):
        """Generate comprehensive trading report."""
        ReportFormatter.print_header("BACKTEST REPORT",
                                     f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if not self.trade_log:
            ReportFormatter.print_status("No trades were executed during the backtest period", "WARNING")
            return

        report_df = pd.DataFrame(self.trade_log)

        # Trade Summary Table
        ReportFormatter.print_section("Trade Execution Summary")

        summary_data = []
        for _, trade in report_df.iterrows():
            duration_str = f"{trade.get('duration_minutes', 'N/A')}min" if pd.notna(
                trade.get('duration_minutes')) else "N/A"
            summary_data.append([
                trade['symbol'],
                trade['type'],
                trade.get('entry_time', 'N/A'),
                ReportFormatter.format_currency(trade['entry_point']),
                trade.get('exit_time', 'N/A'),
                ReportFormatter.format_currency(trade['exit_price']),
                trade['exit_reason'],
                ReportFormatter.format_percentage(trade['pnl_percent']),
                duration_str
            ])

        headers = ["Symbol", "Type", "Entry Time", "Entry Price", "Exit Time", "Exit Price", "Exit Reason", "P&L %",
                   "Duration"]
        logging.info(tabulate(summary_data, headers=headers, tablefmt="rounded_grid"))

        # Performance Statistics
        ReportFormatter.print_section("Performance Analytics")

        total_trades = len(report_df)
        winning_trades = len(report_df[report_df['pnl_percent'] > 0])
        losing_trades = len(report_df[report_df['pnl_percent'] < 0])
        breakeven_trades = len(report_df[report_df['pnl_percent'] == 0])

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        avg_win = report_df[report_df['pnl_percent'] > 0]['pnl_percent'].mean() if winning_trades > 0 else 0
        avg_loss = report_df[report_df['pnl_percent'] < 0]['pnl_percent'].mean() if losing_trades > 0 else 0
        avg_pnl = report_df['pnl_percent'].mean()
        max_win = report_df['pnl_percent'].max() if not report_df.empty else 0
        max_loss = report_df['pnl_percent'].min() if not report_df.empty else 0

        # Calculate expectancy
        expectancy = (win_rate / 100 * avg_win) + ((100 - win_rate) / 100 * avg_loss) if total_trades > 0 else 0

        # Performance metrics table
        performance_data = [
            ["Total Trades", total_trades],
            ["Winning Trades", f"{winning_trades} ({win_rate:.1f}%)"],
            ["Losing Trades", f"{losing_trades} ({(losing_trades / total_trades * 100):.1f}%)"],
            ["Breakeven Trades", breakeven_trades],
            ["Average P&L", f"{avg_pnl:+.2f}%"],
            ["Average Win", f"+{avg_win:.2f}%" if avg_win > 0 else "N/A"],
            ["Average Loss", f"{avg_loss:.2f}%" if avg_loss < 0 else "N/A"],
            ["Best Trade", f"+{max_win:.2f}%"],
            ["Worst Trade", f"{max_loss:.2f}%"],
            ["Expectancy", f"{expectancy:+.2f}%"]
        ]

        logging.info(tabulate(performance_data, headers=["Metric", "Value"], tablefmt="rounded_grid"))

        # Trade Type Analysis
        if 'type' in report_df.columns:
            ReportFormatter.print_section("Strategy Breakdown")

            type_analysis = report_df.groupby('type').agg({
                'pnl_percent': ['count', 'mean', lambda x: (x > 0).sum()],
                'duration_minutes': 'mean'
            }).round(2)

            type_analysis.columns = ['Trades', 'Avg P&L %', 'Wins', 'Avg Duration (min)']
            type_analysis['Win Rate %'] = (type_analysis['Wins'] / type_analysis['Trades'] * 100).round(1)

            logging.info(tabulate(type_analysis, headers=type_analysis.columns, tablefmt="rounded_grid"))

        # Exit Reason Analysis
        if 'exit_reason' in report_df.columns:
            ReportFormatter.print_section("Exit Analysis")

            exit_analysis = report_df['exit_reason'].value_counts()
            exit_pnl = report_df.groupby('exit_reason')['pnl_percent'].mean()

            exit_data = []
            for reason in exit_analysis.index:
                count = exit_analysis[reason]
                avg_pnl = exit_pnl[reason]
                pct_of_total = (count / total_trades * 100)
                exit_data.append([reason, count, f"{pct_of_total:.1f}%", f"{avg_pnl:+.2f}%"])

            headers = ["Exit Reason", "Count", "% of Trades", "Avg P&L %"]
            logging.info(tabulate(exit_data, headers=headers, tablefmt="rounded_grid"))

        # Risk Metrics
        if total_trades > 1:
            ReportFormatter.print_section("Risk Metrics")

            # Calculate additional risk metrics
            returns = report_df['pnl_percent'].values
            volatility = np.std(returns)
            sharpe_ratio = (avg_pnl / volatility) if volatility > 0 else 0

            # Maximum drawdown calculation
            cumulative_returns = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = cumulative_returns - running_max
            max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0

            risk_data = [
                ["Volatility", f"{volatility:.2f}%"],
                ["Sharpe Ratio", f"{sharpe_ratio:.2f}"],
                ["Max Drawdown", f"{max_drawdown:.2f}%"],
                ["Profit Factor",
                 f"{abs(avg_win * winning_trades) / abs(avg_loss * losing_trades):.2f}" if losing_trades > 0 and avg_loss != 0 else "N/A"]
            ]

            logging.info(tabulate(risk_data, headers=["Risk Metric", "Value"], tablefmt="rounded_grid"))

        # Time Analysis
        if 'entry_time' in report_df.columns:
            ReportFormatter.print_section("Timing Analysis")

            # Convert entry times to hours for analysis
            report_df['entry_hour'] = pd.to_datetime(report_df['entry_time'], format='%H:%M').dt.hour
            time_analysis = report_df.groupby('entry_hour').agg({
                'pnl_percent': ['count', 'mean'],
            }).round(2)
            time_analysis.columns = ['Trades', 'Avg P&L %']
            time_analysis.index = [f"{hour:02d}:00-{hour:02d}:59" for hour in time_analysis.index]

            logging.info(tabulate(time_analysis.head(10), headers=time_analysis.columns, tablefmt="rounded_grid"))

        # Save detailed report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(output_folder, f"Detailed_Trade_Report_{timestamp}.csv")
        report_df.to_csv(filename, index=False)

        ReportFormatter.print_section("Report Export")
        ReportFormatter.print_status(f"Detailed report saved to: {filename}", "SUCCESS")

        # System Performance Summary
        system_stats = perf_tracker.get_stats()
        scanner_stats = getattr(self, '_scanner_stats', {})

        logging.info(f"\n{Fore.CYAN}System Performance Summary:{Style.RESET_ALL}")
        perf_data = [
            ["Total Runtime", system_stats.get("Runtime", "N/A")],
            ["API Calls Made", system_stats.get("API Calls", "N/A")],
            ["Symbols Processed", system_stats.get("Symbols Processed", "N/A")],
            ["Filter Success Rate", system_stats.get("Filter Rate", "N/A")],
            ["Breakouts Detected", scanner_stats.get('breakouts_found', 'N/A')],
            ["Trade Trigger Rate",
             f"{(self.trade_stats['triggered_trades'] / max(self.trade_stats['total_signals'], 1) * 100):.1f}%" if
             self.trade_stats['total_signals'] > 0 else "N/A"]
        ]
        logging.info(tabulate(perf_data, headers=["System Metric", "Value"], tablefmt="rounded_grid"))


# ============================================================================
# UTILITY FUNCTIONS (Enhanced)
# ============================================================================
def calculate_spurt_indicators(df):
    """Enhanced spurt calculation with progress indication."""
    if df.empty or len(df) < max(cfg.VOLUME_WINDOW, cfg.ATR_WINDOW):
        return df

    df = df.copy()

    # Calculate True Range (TR) and Average True Range (ATR)
    df['previous_close'] = df['close'].shift(1)
    df['high_low_range'] = df['high'] - df['low']
    df['high_prev_close_range'] = abs(df['high'] - df['previous_close'])
    df['low_prev_close_range'] = abs(df['low'] - df['previous_close'])
    df['tr'] = df[['high_low_range', 'high_prev_close_range', 'low_prev_close_range']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=cfg.ATR_WINDOW, min_periods=cfg.ATR_WINDOW).mean()

    # Calculate Volume Moving Average
    df['volume_ma'] = df['volume'].rolling(window=cfg.VOLUME_WINDOW, min_periods=cfg.VOLUME_WINDOW).mean()

    # Calculate Price Change over 5 minutes
    df['price_change_5min'] = df['close'].diff(periods=5)

    # Identify Spurt Conditions
    df['volume_spurt'] = df['volume'] >= (df['volume_ma'] * cfg.VOLUME_SPURT_FACTOR)
    df['price_spurt'] = abs(df['price_change_5min']) >= (df['atr'] * cfg.PRICE_SPURT_FACTOR)

    # Final Spurt Signal
    df['Spurt_Signal'] = df['volume_spurt'] & df['price_spurt']

    # Clean up intermediate columns
    df = df.drop(columns=['previous_close', 'high_low_range', 'high_prev_close_range', 'low_prev_close_range', 'tr'])

    return df


def get_trading_day(days_back):
    """Get trading day with validation."""
    check_date = datetime.now().date() - timedelta(days=days_back)
    while check_date.weekday() >= 5:  # Skip weekends
        check_date -= timedelta(days=1)
    return check_date


def validate_config():
    """Validate configuration parameters."""
    required_attrs = [
        'SYMBOL_CSV', 'MIN_PRICE', 'MAX_PRICE', 'EMA_LENGTH',
        'MARKET_OPEN_TIME', 'MARKET_CLOSE_TIME', 'DAY_END_SQUARE_OFF',
        'CANDLE_INTERVAL', 'MINUTE_RESOLUTION', 'BACKTEST_DAYS'
    ]

    missing_attrs = [attr for attr in required_attrs if not hasattr(cfg, attr)]
    if missing_attrs:
        ReportFormatter.print_status(f"Missing configuration: {', '.join(missing_attrs)}", "ERROR")
        return False

    return True


def main():
    """Enhanced main function with comprehensive reporting."""
    # --- LOGGING SETUP (FOR THE ENTIRE RUN) ---
    os.makedirs("run_logs", exist_ok=True)
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join("run_logs", f"backtest_run_{run_timestamp}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Console handler (with colors)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)

    # File handler (without colors)
    file_handler = logging.FileHandler(log_filename, 'w', 'utf-8')
    file_handler.setFormatter(StripAnsiFormatter('%(message)s'))
    logger.addHandler(file_handler)

    logging.info(f"Log file for this multi-day run: {log_filename}")

    ReportFormatter.print_header(
        "EQUITY BACKTESTING SYSTEM",
        f"Professional Trading Strategy Analysis | Version 2.0"
    )

    # --- CONFIGURATION VALIDATION AND LOGGING ---
    if not validate_config():
        ReportFormatter.print_status("Configuration validation failed. Please check config.py", "ERROR")
        return

    logging.info("\n" + "="*80)
    logging.info(" " * 25 + "CONFIGURATION SETTINGS")
    logging.info("="*80)
    config_dict = {key: getattr(cfg, key) for key in dir(cfg) if not key.startswith('__')}
    sensitive_keys = ['CLIENT_ID', 'SECRET_KEY']
    for key, value in config_dict.items():
        if key in sensitive_keys:
            masked_value = '********'
            logging.info(f"  {key:<30}: {masked_value}")
        else:
            logging.info(f"  {key:<30}: {value}")
    logging.info("="*80)

    # --- AUTHENTICATION ---
    access_token = FyersAuth().authenticate()
    if not access_token:
        ReportFormatter.print_status("Authentication failed. Cannot proceed", "ERROR")
        return

    fyers_data = FyersData(access_token)

    # --- SYMBOL LOADING ---
    try:
        all_symbols = pd.read_csv(cfg.SYMBOL_CSV)['fyers_symbol'].dropna().tolist()
        ReportFormatter.print_status(f"Loaded {len(all_symbols)} symbols from {cfg.SYMBOL_CSV}", "SUCCESS")
    except Exception as e:
        ReportFormatter.print_status(f"Failed to load symbols: {e}", "ERROR")
        return

    # --- MAIN BACKTESTING LOOP ---
    for day_offset in range(1, cfg.BACKTEST_DAYS + 1):

        backtest_date = get_trading_day(day_offset)

        # Create a date-stamped folder for this specific day's output
        output_folder = backtest_date.strftime('%d%m%y')
        os.makedirs(output_folder, exist_ok=True)

        logging.info("\n\n" + "#"*80)
        logging.info(f"###### STARTING BACKTEST FOR DATE: {backtest_date.strftime('%Y-%m-%d')} ######")
        logging.info("#"*80 + "\n")

        # --- STAGE 1: PRE-FILTERING ---
        pre_filter = PreFilterEngine(fyers_data)
        filtered_stocks, _ = pre_filter.process_symbols(all_symbols)

        if not filtered_stocks:
            ReportFormatter.print_status(f"No stocks qualified for intraday analysis on {backtest_date.strftime('%Y-%m-%d')}. Skipping.", "WARNING")
            continue

        pre_filtered_symbols = [s['symbol'] for s in filtered_stocks]
        pdh_map = {s['symbol']: s['pdh'] for s in filtered_stocks}

        # --- STAGE 2: INTRADAY BACKTESTING ---
        ReportFormatter.print_section(f"Intraday Data Collection for {backtest_date.strftime('%Y-%m-%d')}")
        five_min_data = fyers_data.get_bulk_intraday_data(pre_filtered_symbols, backtest_date, cfg.CANDLE_INTERVAL)
        one_min_data = fyers_data.get_bulk_intraday_data(pre_filtered_symbols, backtest_date, cfg.MINUTE_RESOLUTION)

        ReportFormatter.print_status("Calculating technical indicators for 1-minute data", "INFO")
        indicator_count = 0
        for symbol in one_min_data:
            if not one_min_data[symbol].empty:
                one_min_data[symbol] = calculate_spurt_indicators(one_min_data[symbol])
                indicator_count += 1
        ReportFormatter.print_status(f"Technical indicators calculated for {indicator_count} symbols", "SUCCESS")

        hist_filename = os.path.join(output_folder, f"Historical_Data_{backtest_date.strftime('%d%m%y')}.csv")
        all_5min_df = pd.concat([df.assign(symbol=sym) for sym, df in five_min_data.items() if not df.empty])
        if not all_5min_df.empty:
            all_5min_df.to_csv(hist_filename, index=False)
            ReportFormatter.print_status(f"Historical data exported to {hist_filename}", "SUCCESS")

        scanner = BreakoutScanner()
        order_manager = OrderManager()

        ReportFormatter.print_section(f"Market Simulation for {backtest_date.strftime('%Y-%m-%d')}")
        all_1min_datetimes = []
        for df in one_min_data.values():
            if not df.empty:
                all_1min_datetimes.extend(df['datetime'].tolist())

        if not all_1min_datetimes:
            ReportFormatter.print_status("No 1-minute data available to run simulation.", "WARNING")
        else:
            unique_1min_datetimes = sorted(list(set(all_1min_datetimes)))
            ReportFormatter.print_status(f"Market simulation ready: Processing {len(unique_1min_datetimes)} 1-minute candles.", "INFO")
            for i, candle_start_dt in enumerate(unique_1min_datetimes):
                current_processing_dt = candle_start_dt + timedelta(minutes=1)
                current_processing_time_str = current_processing_dt.strftime("%H:%M")
                if current_processing_time_str > cfg.DAY_END_SQUARE_OFF:
                    break
                print(f"\r{Fore.WHITE}Processing Time: {current_processing_time_str} ({i+1}/{len(unique_1min_datetimes)}){Style.RESET_ALL}", end="")
                if current_processing_dt.minute % int(cfg.CANDLE_INTERVAL) == 0:
                    five_min_candle_start_dt = current_processing_dt - timedelta(minutes=int(cfg.CANDLE_INTERVAL))
                    five_min_candle_start_time_str = five_min_candle_start_dt.strftime("%H:%M")
                    stocks_in_5min_candle = {}
                    for sym, df in five_min_data.items():
                        if not df.empty and five_min_candle_start_time_str in df['time'].values:
                            stocks_in_5min_candle[sym] = df[df['time'] == five_min_candle_start_time_str].iloc[0]
                    for symbol, five_min_candle in stocks_in_5min_candle.items():
                        if symbol in pdh_map:
                            trade_idea = scanner.scan({'symbol': symbol}, five_min_candle, pdh_map[symbol])
                            if trade_idea:
                                order_manager.add_potential_trade(trade_idea, backtest_date)
                one_min_candle_start_time_str = candle_start_dt.strftime("%H:%M")
                stocks_in_1min_candle = {}
                for sym, df in one_min_data.items():
                    if not df.empty and one_min_candle_start_time_str in df['time'].values:
                        stocks_in_1min_candle[sym] = df[df['time'] == one_min_candle_start_time_str].iloc[0]
                if stocks_in_1min_candle:
                    order_manager.process_active_trades(stocks_in_1min_candle)
            print()
            ReportFormatter.print_status("Market simulation complete.", "SUCCESS")

        last_candles = {sym: df.iloc[-1] for sym, df in one_min_data.items() if not df.empty}
        order_manager.square_off_open_trades(last_candles)
        order_manager._scanner_stats = scanner.get_stats()

        # --- STAGE 3: COMPREHENSIVE REPORTING (FOR THE DAY) ---
        order_manager.generate_report(output_folder=output_folder)

    ReportFormatter.print_header("MULTI-DAY BACKTEST RUN COMPLETED")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        if logging.getLogger().hasHandlers():
            logging.info(f"\n{Fore.YELLOW}Process interrupted by user. Graceful shutdown initiated...{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.YELLOW}Process interrupted by user. Graceful shutdown initiated...{Style.RESET_ALL}")
    except Exception as e:
        if logging.getLogger().hasHandlers():
            logging.error(f"A critical error occurred: {e}")
            logging.error(traceback.format_exc())
        else:
            print(f"\n{Fore.RED}A critical error occurred: {e}{Style.RESET_ALL}")
            import traceback
            traceback.print_exc()
