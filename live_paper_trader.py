#!/usr/bin/env python3
"""
EQUITY LIVE PAPER TRADING BOT
=============================
This script monitors the market in real-time, identifies trading opportunities
based on a predefined strategy, and logs them to an Excel file for paper trading.
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
from colorama import Fore, Style, init
from tabulate import tabulate
import warnings
import logging
import re
import queue
import threading

warnings.filterwarnings('ignore')

try:
    import openpyxl
except ImportError:
    sys.exit("`openpyxl` library not found. Please install it using: pip install openpyxl")

import config as cfg

# --- Setup Custom Logging ---
class StripAnsiFormatter(logging.Formatter):
    def format(self, record):
        message = super().format(record)
        return re.sub(r'\x1b\[([0-9]{1,2}(;[0-9]{1,2})?)?[m|K]', '', message)

init(autoreset=True)

try:
    from fyers_apiv3 import fyersModel
except ImportError:
    sys.exit("Fyers SDK not found. Please install: pip install fyers-apiv3")

# ============================================================================
# UTILITY CLASSES & FUNCTIONS
# ============================================================================
class ReportFormatter:
    @staticmethod
    def print_header(title: str, subtitle: str = ""):
        width = 80
        logging.info(f"\n{Fore.CYAN}{'═' * width}")
        logging.info(f"  {title.upper()}")
        if subtitle: logging.info(f"  {subtitle}")
        logging.info(f"{'═' * width}{Style.RESET_ALL}")

    @staticmethod
    def print_section(title: str):
        logging.info(f"\n{Fore.BLUE}▶ {title}{Style.RESET_ALL}")
        logging.info(f"{Fore.BLUE}{'─' * (len(title) + 3)}{Style.RESET_ALL}")

    @staticmethod
    def print_status(message: str, status: str = "INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        colors = {"INFO": Fore.BLUE, "SUCCESS": Fore.GREEN, "WARNING": Fore.YELLOW, "ERROR": Fore.MAGENTA, "TRADE": Fore.CYAN}
        color = colors.get(status, Fore.WHITE)
        logging.info(f"{Fore.WHITE}[{timestamp}] {color}[{status}]{Style.RESET_ALL} {message}")

    @staticmethod
    def format_currency(amount: float, decimals: int = 2) -> str:
        return f"₹{amount:,.{decimals}f}"

class FyersAuth:
    def __init__(self):
        self.access_token = None
        self.token_file = cfg.ACCESS_TOKEN_FILE

    def load_saved_token(self):
        try:
            if os.path.exists(self.token_file):
                with open(self.token_file, 'r') as f: data = json.load(f)
                if data.get('access_token') and time.time() + 600 < data.get('expiry', 0):
                    ReportFormatter.print_status("Using saved token.", "SUCCESS")
                    return data['access_token']
        except Exception as e:
            ReportFormatter.print_status(f"Error loading token: {e}", "ERROR")
        return None

    def save_token(self, token):
        try:
            data = {'access_token': token, 'expiry': int(time.time()) + (23*60*60)}
            with open(self.token_file, 'w') as f: json.dump(data, f)
            ReportFormatter.print_status("Access token saved.", "SUCCESS")
        except Exception as e:
            ReportFormatter.print_status(f"Failed to save token: {e}", "ERROR")

    def authenticate(self):
        ReportFormatter.print_section("Authentication")
        if saved_token := self.load_saved_token():
            return saved_token

        ReportFormatter.print_status("Interactive authentication required", "INFO")
        try:
            session = fyersModel.SessionModel(client_id=cfg.CLIENT_ID, secret_key=cfg.SECRET_KEY, redirect_uri=cfg.REDIRECT_URI, response_type=cfg.RESPONSE_TYPE, grant_type=cfg.GRANT_TYPE)
            auth_url = session.generate_authcode()
            print(f"\n{Fore.CYAN}1. Open this URL in your browser:\n   {auth_url}")
            print(f"2. Log in and get the auth_code from the redirect URL.{Style.RESET_ALL}")
            auth_code = input(f"\n{Fore.YELLOW}Enter auth_code: {Style.RESET_ALL}").strip()
            if not auth_code:
                ReportFormatter.print_status("No auth_code provided", "ERROR")
                return None
            session.set_token(auth_code)
            response = session.generate_token()
            if access_token := response.get("access_token"):
                ReportFormatter.print_status("Authentication successful", "SUCCESS")
                self.save_token(access_token)
                return access_token
            else:
                ReportFormatter.print_status(f"Authentication failed: {response.get('message', 'Unknown')}", "ERROR")
                return None
        except Exception as e:
            ReportFormatter.print_status(f"Authentication error: {e}", "ERROR")
            return None

class FyersData:
    def __init__(self, access_token):
        self.client = fyersModel.FyersModel(client_id=cfg.CLIENT_ID, token=access_token)

    def get_daily_data(self, symbol: str, base_date: datetime.date, days_back: int = 45):
        params = {"symbol": f"NSE:{symbol}-EQ", "resolution": "D", "date_format": "1", "range_from": (base_date - timedelta(days=days_back)).strftime("%Y-%m-%d"), "range_to": base_date.strftime("%Y-%m-%d"), "cont_flag": "1"}
        try:
            response = self.client.history(params)
            if response.get('code') != 200: return None, f"API Error: {response.get('message', 'Unknown')}"
            candles = response.get("candles", [])
            if not candles or len(candles) < cfg.EMA_LENGTH + 3: return None, f"Insufficient data"
            data = [{'date': datetime.fromtimestamp(c[0]).date(), 'open': float(c[1]), 'high': float(c[2]), 'low': float(c[3]), 'close': float(c[4]), 'volume': float(c[5])} for c in candles]
            df = pd.DataFrame(data).sort_values('date').reset_index(drop=True)
            df[f'ema_{cfg.EMA_LENGTH}'] = df['close'].ewm(span=cfg.EMA_LENGTH, adjust=False).mean()
            return df, None
        except Exception as e: return None, str(e)

    def get_intraday_5min_data(self, symbols: List[str], date: datetime.date):
        all_data = {}
        for symbol in symbols:
            params = {"symbol": symbol, "resolution": "5", "date_format": "1", "range_from": date.strftime("%Y-%m-%d"), "range_to": date.strftime("%Y-%m-%d"), "cont_flag": "1"}
            try:
                response = self.client.history(params)
                if response.get('code') == 200 and response.get("candles"):
                    data = [{'datetime': datetime.fromtimestamp(c[0]), 'time': datetime.fromtimestamp(c[0]).strftime("%H:%M"), 'open': float(c[1]), 'high': float(c[2]), 'low': float(c[3]), 'close': float(c[4]), 'volume': float(c[5])} for c in response.get("candles", [])]
                    all_data[symbol] = pd.DataFrame(data).sort_values('datetime')
                else: all_data[symbol] = pd.DataFrame()
            except Exception: all_data[symbol] = pd.DataFrame()
            time.sleep(0.2)
        return all_data

class PreFilterEngine:
    def __init__(self, fyers_data): self.fyers_data = fyers_data
    def filter_stock(self, symbol: str, base_date: datetime.date):
        try:
            daily_df, error = self.fyers_data.get_daily_data(symbol, base_date=base_date, days_back=45)
            if error or daily_df is None or len(daily_df) < 3: return None
            latest, previous, two_days_ago = daily_df.iloc[-1], daily_df.iloc[-2], daily_df.iloc[-3]
            ema_col = f'ema_{cfg.EMA_LENGTH}'
            if ema_col not in daily_df.columns: return None
            ema_1day, ema_2days = previous[ema_col], two_days_ago[ema_col]
            if not (cfg.MIN_PRICE <= latest['close'] <= cfg.MAX_PRICE): return None
            if not (ema_1day > ema_2days): return None
            if not (previous['high'] > two_days_ago['high']): return None
            if not (previous['close'] > previous['open']): return None
            if not (previous['close'] > two_days_ago['open']): return None
            if not (previous['low'] > ema_1day): return None
            return {'symbol': f"NSE:{symbol}-EQ", 'pdh': round(previous['high'], 2)}
        except Exception: return None
    def process_symbols(self, symbols: List[str], base_date: datetime.date):
        ReportFormatter.print_section("Pre-Market Filtering")
        filtered = [self.filter_stock(s, base_date) for s in symbols]
        return [s for s in filtered if s is not None]

class BreakoutScanner:
    def __init__(self): self.identified_stocks = set()
    def scan(self, stock_data, five_min_candle, pdh):
        symbol = stock_data['symbol']
        if symbol in self.identified_stocks: return None
        candle_time, candle_open, candle_high, candle_close, candle_low = five_min_candle['time'], five_min_candle['open'], five_min_candle['high'], five_min_candle['close'], five_min_candle['low']
        if candle_time == cfg.MARKET_OPEN_TIME and candle_open < pdh and candle_close > candle_open:
            rejection = (candle_high - candle_close) / (candle_high - candle_low) if (candle_high - candle_low) > 0 else 0
            if rejection <= cfg.INITIAL_REJECTION_THRESHOLD:
                self.identified_stocks.add(symbol); return {'symbol': symbol, 'entry_point': candle_high, 'type': 'Initial', 'scan_time': candle_time}
        if candle_open < pdh and candle_close >= pdh:
            self.identified_stocks.add(symbol); return {'symbol': symbol, 'entry_point': candle_high, 'type': 'Stage', 'scan_time': candle_time}
        return None

def calculate_spurt_indicators(df):
    if df.empty or len(df) < max(cfg.VOLUME_WINDOW, cfg.ATR_WINDOW): return df
    df = df.copy()
    df['previous_close'] = df['close'].shift(1)
    df['tr'] = df.apply(lambda row: max(row['high'] - row['low'], abs(row['high'] - row['previous_close']), abs(row['low'] - row['previous_close'])), axis=1)
    df['atr'] = df['tr'].rolling(window=cfg.ATR_WINDOW).mean()
    df['volume_ma'] = df['volume'].rolling(window=cfg.VOLUME_WINDOW).mean()
    df['price_change_5min'] = df['close'].diff(periods=5)
    df['volume_spurt'] = df['volume'] >= (df['volume_ma'] * cfg.VOLUME_SPURT_FACTOR)
    df['price_spurt'] = abs(df['price_change_5min']) >= (df['atr'] * cfg.PRICE_SPURT_FACTOR)
    df['Spurt_Signal'] = df['volume_spurt'] & df['price_spurt']
    return df.drop(columns=['previous_close', 'tr'])

class WebsocketClient:
    def __init__(self, access_token: str, symbols: List[str], candle_queue: queue.Queue):
        self.access_token = access_token; self.symbols = symbols; self.candle_queue = candle_queue
        self.data_type = "SymbolUpdate"; self.fyers_socket = None
        self.ticks = {sym: [] for sym in symbols}
        self.candle_history = {sym: pd.DataFrame(columns=['datetime', 'open', 'high', 'low', 'close', 'volume']) for sym in symbols}
        self.last_tick_minute = {sym: None for sym in symbols}
        self.last_cumulative_volume = {sym: 0 for sym in symbols}
        ReportFormatter.print_status("WebsocketClient initialized.", "INFO")
    def connect(self):
        app_id = cfg.CLIENT_ID.split('-')[0]; token = f"{app_id}:{self.access_token}"
        self.fyers_socket = fyersModel.FyersSocket(access_token=token, log_path=os.path.join(os.getcwd(), "run_logs"))
        self.fyers_socket.with_data(on_message=self.on_message, on_open=self.on_open, on_close=self.on_close, on_error=self.on_error)
        ReportFormatter.print_status("Connecting to websocket...", "INFO"); self.fyers_socket.connect()
    def subscribe(self):
        if self.fyers_socket: ReportFormatter.print_status(f"Subscribing to {len(self.symbols)} symbols.", "INFO"); self.fyers_socket.subscribe(symbol=self.symbols, data_type=self.data_type)
    def on_message(self, msg):
        try:
            symbol, ltp, volume = msg.get('symbol'), msg.get('ltp'), msg.get('vol_traded_today')
            if not all([symbol, ltp, volume]): return
            current_dt = datetime.fromtimestamp(msg.get('timestamp', time.time())); current_minute = current_dt.minute
            if self.last_tick_minute.get(symbol) is None: self.last_tick_minute[symbol] = current_minute; self.last_cumulative_volume[symbol] = volume
            if current_minute != self.last_tick_minute[symbol]:
                candle_start_time = current_dt.replace(second=0, microsecond=0) - timedelta(minutes=1)
                self.form_and_process_candle(symbol, candle_start_time); self.last_tick_minute[symbol] = current_minute
            self.ticks[symbol].append({'price': ltp, 'volume': volume})
        except Exception as e: ReportFormatter.print_status(f"Error in on_message for {msg.get('symbol')}: {e}", "ERROR")
    def form_and_process_candle(self, symbol, candle_dt):
        ticks_in_minute = self.ticks.get(symbol, [])
        if not ticks_in_minute: return
        prices = [t['price'] for t in ticks_in_minute]
        volume_for_minute = ticks_in_minute[-1]['volume'] - self.last_cumulative_volume[symbol]
        self.last_cumulative_volume[symbol] = ticks_in_minute[-1]['volume']
        new_candle = {'datetime': candle_dt, 'time': candle_dt.strftime("%H:%M"), 'open': prices[0], 'high': max(prices), 'low': min(prices), 'close': prices[-1], 'volume': volume_for_minute}
        new_candle_df = pd.DataFrame([new_candle])
        self.candle_history[symbol] = pd.concat([self.candle_history[symbol], new_candle_df], ignore_index=True)
        self.candle_history[symbol] = self.candle_history[symbol].tail(max(cfg.ATR_WINDOW, cfg.VOLUME_WINDOW) + 10)
        enriched_history = calculate_spurt_indicators(self.candle_history[symbol])
        if not enriched_history.empty: self.candle_queue.put({'symbol': symbol, 'candle': enriched_history.iloc[-1].to_dict()})
        self.ticks[symbol] = []
    def on_open(self): ReportFormatter.print_status("Websocket connection opened.", "SUCCESS"); self.subscribe()
    def on_close(self): ReportFormatter.print_status("Websocket connection closed.", "WARNING")
    def on_error(self, msg): ReportFormatter.print_status(f"Websocket error: {msg}", "ERROR")
    def stop(self):
        if self.fyers_socket: self.fyers_socket.close_connection()

class OrderLogger:
    def __init__(self):
        self.pending_trades = {}; self.logged_trades = set()
        self.output_file = f"paper_trades_{datetime.now().strftime('%Y-%m-%d')}.xlsx"
        self.lock = threading.Lock(); self._setup_excel_file()
        ReportFormatter.print_status(f"Paper trade log initialized: {self.output_file}", "INFO")
    def _setup_excel_file(self):
        if not os.path.exists(self.output_file):
            pd.DataFrame(columns=["Timestamp", "Symbol", "Type", "Entry Price", "Signal Time"]).to_excel(self.output_file, index=False, sheet_name="Trades")
    def add_potential_trade(self, trade_idea):
        symbol = trade_idea['symbol']
        if symbol not in self.logged_trades and symbol not in self.pending_trades:
            self.pending_trades[symbol] = trade_idea
            ReportFormatter.print_status(f"Trade setup: {symbol} ({trade_idea['type']}) | Entry: {ReportFormatter.format_currency(trade_idea['entry_point'])}", "INFO")
    def process_pending_trades(self, symbol: str, one_min_candle: Dict):
        if symbol not in self.pending_trades: return
        trade_idea = self.pending_trades[symbol]
        if one_min_candle.get('high') >= trade_idea['entry_point'] and one_min_candle.get('Spurt_Signal', False):
            self.log_trade_to_excel(trade_idea)
            del self.pending_trades[symbol]; self.logged_trades.add(symbol)
    def log_trade_to_excel(self, trade_details: Dict):
        with self.lock:
            try:
                new_trade = pd.DataFrame([{"Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "Symbol": trade_details['symbol'], "Type": trade_details['type'], "Entry Price": trade_details['entry_point'], "Signal Time": trade_details['scan_time']}])
                df = pd.read_excel(self.output_file, sheet_name="Trades")
                df = pd.concat([df, new_trade], ignore_index=True)
                df.to_excel(self.output_file, index=False, sheet_name="Trades")
                ReportFormatter.print_status(f"BUY SIGNAL LOGGED: {trade_details['symbol']} at {ReportFormatter.format_currency(trade_details['entry_point'])}", "TRADE")
            except Exception as e: ReportFormatter.print_status(f"Failed to log trade to Excel: {e}", "ERROR")

def main():
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join("run_logs", f"live_run_{run_timestamp}.log")
    os.makedirs("run_logs", exist_ok=True)
    logger = logging.getLogger(); logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(); console_handler.setFormatter(logging.Formatter('%(message)s')); logger.addHandler(console_handler)
    file_handler = logging.FileHandler(log_filename, 'w', 'utf-8'); file_handler.setFormatter(StripAnsiFormatter('%(message)s')); logger.addHandler(file_handler)

    ReportFormatter.print_header("EQUITY LIVE PAPER TRADING BOT", f"Version 3.0 | Run ID: {run_timestamp}")
    access_token = FyersAuth().authenticate()
    if not access_token: return

    fyers_data = FyersData(access_token)
    today = datetime.now().date()

    ReportFormatter.print_section("Pre-Market Analysis")
    all_symbols = [s.split('-EQ')[0].split('NSE:')[1] for s in pd.read_csv(cfg.SYMBOL_CSV)['fyers_symbol'].dropna().tolist()]
    filtered_stocks = PreFilterEngine(fyers_data).process_symbols(all_symbols, today - timedelta(days=1))
    if not filtered_stocks: ReportFormatter.print_status("No stocks passed pre-filter. Exiting.", "WARNING"); return

    watchlist = {s['symbol']: s for s in filtered_stocks}
    pdh_map = {s['symbol']: s['pdh'] for s in filtered_stocks}

    while datetime.now().strftime("%H:%M") < cfg.MARKET_OPEN_TIME:
        ReportFormatter.print_status(f"Waiting for market to open at {cfg.MARKET_OPEN_TIME}...", "INFO"); time.sleep(20)

    ReportFormatter.print_section("Market Open Initial Scan")
    scanner = BreakoutScanner(); order_logger = OrderLogger()

    ReportFormatter.print_status("Waiting for first 5-minute candle to form...", "INFO"); time.sleep(300)
    initial_5min_data = fyers_data.get_intraday_5min_data(list(watchlist.keys()), today)
    for symbol, df in initial_5min_data.items():
        if not df.empty:
            trade_idea = scanner.scan(watchlist[symbol], df.iloc[0], pdh_map[symbol])
            if trade_idea: order_logger.add_potential_trade(trade_idea)

    ReportFormatter.print_section("Live Market Monitoring")
    candle_queue = queue.Queue()
    websocket_client = WebsocketClient(access_token, list(watchlist.keys()), candle_queue)
    ws_thread = threading.Thread(target=websocket_client.connect, daemon=True); ws_thread.start()

    last_5min_scan_time = datetime.now()
    try:
        while datetime.now().strftime("%H:%M") < cfg.MARKET_CLOSE_TIME:
            try:
                data = candle_queue.get(timeout=1)
                order_logger.process_pending_trades(data['symbol'], data['candle'])
            except queue.Empty: pass

            if (datetime.now() - last_5min_scan_time).total_seconds() >= 300:
                ReportFormatter.print_status("Performing periodic 5-min scan for Stage Breakouts...", "INFO")
                latest_5min_data = fyers_data.get_intraday_5min_data(list(watchlist.keys()), today)
                for symbol, df in latest_5min_data.items():
                    if not df.empty:
                        trade_idea = scanner.scan(watchlist[symbol], df.iloc[-1], pdh_map[symbol])
                        if trade_idea: order_logger.add_potential_trade(trade_idea)
                last_5min_scan_time = datetime.now()
    except KeyboardInterrupt: ReportFormatter.print_status("Manual interruption detected.", "WARNING")
    finally:
        ReportFormatter.print_section("Market Closed - Shutting Down")
        websocket_client.stop(); ws_thread.join(timeout=5)
        ReportFormatter.print_status("Bot shutdown complete.", "SUCCESS")

if __name__ == "__main__":
    main()
