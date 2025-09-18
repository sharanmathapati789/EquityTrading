#!/usr/bin/env python3
"""
INTEGRATED EQUITY BACKTESTING SYSTEM
====================================
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

# Import settings from the config file
import config as cfg

# Initialize colorama
init(autoreset=True)

# Try to import Fyers SDK
try:
    from fyers_apiv3 import fyersModel
except ImportError:
    print(f"{Fore.RED}Error: No Fyers SDK found. Please install: pip install fyers-apiv3")
    sys.exit(1)


# ============================================================================
# AUTHENTICATION
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
                    print(f"{Fore.GREEN}Found a valid saved access token.{Style.RESET_ALL}")
                    return token
                else:
                    print(f"{Fore.YELLOW}Saved access token has expired or is invalid.{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error loading saved token: {e}{Style.RESET_ALL}")
        return None

    def save_token(self, token):
        """Save the access token and its expiry time to a file."""
        try:
            expiry_time = int(time.time()) + (23 * 60 * 60)
            data = {'access_token': token, 'expiry': expiry_time, 'created_at': int(time.time())}
            with open(self.token_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"{Fore.GREEN}New access token saved successfully to {self.token_file}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Failed to save access token: {e}{Style.RESET_ALL}")

    def authenticate(self):
        """Orchestrates the Fyers authentication process."""
        saved_token = self.load_saved_token()
        if saved_token:
            self.access_token = saved_token
            return self.access_token

        print(f"\n{Fore.YELLOW}--- Fyers API Authentication Required ---{Style.RESET_ALL}")
        try:
            session = fyersModel.SessionModel(client_id=cfg.CLIENT_ID, secret_key=cfg.SECRET_KEY,
                                              redirect_uri=cfg.REDIRECT_URI, response_type=cfg.RESPONSE_TYPE,
                                              grant_type=cfg.GRANT_TYPE)
            auth_url = session.generate_authcode()
            print(f"\n{Fore.CYAN}Step 1: Please copy the URL below and paste it into your web browser:")
            print(auth_url)
            print(f"\n{Fore.CYAN}Step 2: Log in to Fyers and authorize the application.")
            print(f"\n{Fore.CYAN}Step 3: After authorization, copy the 'auth_code' from the redirect URL.")
            auth_code = input(f"\n{Fore.YELLOW}Please enter the auth_code here: {Style.RESET_ALL}").strip()

            if not auth_code:
                print(f"{Fore.RED}Authentication failed: No authorization code provided.{Style.RESET_ALL}")
                return None

            session.set_token(auth_code)
            response = session.generate_token()

            if response and response.get("access_token"):
                access_token = response["access_token"]
                print(f"{Fore.GREEN}Authentication successful!{Style.RESET_ALL}")
                self.save_token(access_token)
                self.access_token = access_token
                return self.access_token
            else:
                print(f"{Fore.RED}Authentication failed: {response.get('message', 'Unknown error')}{Style.RESET_ALL}")
                return None
        except Exception as e:
            print(f"{Fore.RED}An error occurred during authentication: {e}{Style.RESET_ALL}")
            return None

# ============================================================================
# DATA FETCHER
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
        """Get daily OHLCV data."""
        params = {"symbol": self.format_symbol(symbol), "resolution": "D", "date_format": "1",
                  "range_from": (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d"),
                  "range_to": datetime.now().strftime("%Y-%m-%d"), "cont_flag": "1"}
        try:
            response = self.client.history(params)
            if response.get('code') != 200: return None, f"API Error: {response.get('message', 'Unknown')}"
            candles = response.get("candles", [])
            if not candles or len(candles) < cfg.EMA_LENGTH + 3: return None, f"Insufficient data: got {len(candles)} candles"

            data = [{'date': datetime.fromtimestamp(c[0]).date(), 'open': float(c[1]), 'high': float(c[2]),
                     'low': float(c[3]), 'close': float(c[4]), 'volume': float(c[5])} for c in candles]
            df = pd.DataFrame(data).sort_values('date').reset_index(drop=True)
            df[f'ema_{cfg.EMA_LENGTH}'] = df['close'].ewm(span=cfg.EMA_LENGTH, adjust=False).mean()
            return df, None
        except Exception as e:
            return None, str(e)

    def get_intraday_data_for_day(self, symbol, date, resolution):
        """Get historical intraday data for a specific date and resolution."""
        params = {"symbol": self.format_symbol(symbol), "resolution": resolution, "date_format": "1",
                  "range_from": date.strftime("%Y-%m-%d"), "range_to": date.strftime("%Y-%m-%d"), "cont_flag": "1"}
        try:
            response = self.client.history(params)
            if response.get('code') != 200: return None, f"API Error: {response.get('message', 'Unknown')}"
            candles = response.get("candles", [])
            if not candles: return None, "No intraday data available"

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
        """Fetch intraday data for a list of symbols for a specific date."""
        all_data = {}
        print(f"\nFetching {resolution}-min data for {len(symbols)} symbols on {date.strftime('%Y-%m-%d')}...")
        for i, symbol in enumerate(symbols, 1):
            print(f"\rProgress: {i}/{len(symbols)} - {symbol}", end="")
            df, error = self.get_intraday_data_for_day(symbol, date, resolution)
            if error:
                print(f"\n{Fore.RED}Error fetching {resolution}-min data for {symbol}: {error}{Style.RESET_ALL}")
                all_data[symbol] = pd.DataFrame()
            else:
                all_data[symbol] = df
            time.sleep(0.2)  # Rate limit
        print("\nData fetching complete.")
        return all_data

# ============================================================================
# PRE-FILTER ENGINE
# ============================================================================
class PreFilterEngine:
    def __init__(self, fyers_data):
        self.fyers_data = fyers_data

    def filter_stock(self, symbol):
        """Filter individual stock based on the user-defined criteria."""
        try:
            daily_df, error = self.fyers_data.get_daily_data(symbol, days_back=45)
            if error or daily_df is None or len(daily_df) < 3: return None, error or "Insufficient data"

            latest, previous, two_days_ago = daily_df.iloc[-1], daily_df.iloc[-2], daily_df.iloc[-3]
            ema_col = f'ema_{cfg.EMA_LENGTH}'
            if ema_col not in daily_df.columns: return None, "EMA not calculated"

            ema_1day, ema_2days = previous[ema_col], two_days_ago[ema_col]
            reasons = []
            if not (cfg.MIN_PRICE <= latest['close'] <= cfg.MAX_PRICE): reasons.append("Price out of range")
            if not (ema_1day > ema_2days): reasons.append("EMA slope negative")
            if not (previous['high'] > two_days_ago['high']): reasons.append("PDH <= 2DH")
            if not (previous['close'] > previous['open']): reasons.append("Bearish PD candle")
            if not (previous['close'] > two_days_ago['open']): reasons.append("PD Close <= 2D Open")
            if not (previous['low'] > ema_1day): reasons.append("PD Low <= EMA1D")

            if reasons: return None, "; ".join(reasons)

            return {'symbol': symbol, 'pdh': round(previous['high'], 2)}, None
        except Exception as e:
            return None, f"Exception in filter_stock: {e}"

    def process_symbols(self, symbols: List[str]) -> (List[Dict], List[Dict]):
        """Process all symbols and return filtered results."""
        print(f"\n{Fore.BLUE}Processing {len(symbols)} symbols for Pre-filtering...{Style.RESET_ALL}")
        filtered_stocks, failed_stocks = [], []
        for i, symbol in enumerate(symbols, 1):
            print(f"\r{Fore.WHITE}Progress: {i:>3}/{len(symbols)} - Processing {symbol:<20}{Style.RESET_ALL}", end="")
            stock_data, error = self.filter_stock(symbol)
            if stock_data:
                filtered_stocks.append(stock_data)
            else:
                failed_stocks.append({'symbol': symbol, 'reason': error})
            time.sleep(0.1)
        print(f"\n{Fore.GREEN}Pre-filtering complete.{Style.RESET_ALL}")
        return filtered_stocks, failed_stocks

# ============================================================================
# BREAKOUT SCANNER & ORDER MANAGER
# ============================================================================
class BreakoutScanner:
    def __init__(self):
        self.identified_stocks = set()

    def scan(self, stock_data, five_min_candle, pdh):
        symbol = stock_data['symbol']
        if symbol in self.identified_stocks: return None

        candle_time = five_min_candle['time']
        candle_open, candle_high, candle_close = five_min_candle['open'], five_min_candle['high'], five_min_candle['close']

        # Initial Breakout Scan (only for the first candle)
        if candle_time == cfg.MARKET_OPEN_TIME:
            if candle_open < pdh and candle_close > candle_open:
                rejection = (candle_high - candle_close) / (candle_high - five_min_candle['low']) if (candle_high - five_min_candle['low']) > 0 else 0
                if rejection <= cfg.INITIAL_REJECTION_THRESHOLD:
                    self.identified_stocks.add(symbol)
                    return {'symbol': symbol, 'entry_point': candle_high, 'type': 'Initial'}

        # Stage Breakout Scan (for any candle)
        if candle_open < pdh and candle_close >= pdh:
            self.identified_stocks.add(symbol)
            return {'symbol': symbol, 'entry_point': candle_high, 'type': 'Stage'}

        return None

class OrderManager:
    def __init__(self):
        self.trade_log = []
        self.active_trades = []

    def add_potential_trade(self, trade_idea, date):
        trade_type = trade_idea['type']
        entry_price = trade_idea['entry_point']
        sl_pct = cfg.INITIAL_SL_PERCENT if trade_type == 'Initial' else cfg.STAGE_SL_PERCENT
        target_pct = cfg.INITIAL_TARGET_PERCENT if trade_type == 'Initial' else cfg.STAGE_TARGET_PERCENT

        self.active_trades.append({
            'symbol': trade_idea['symbol'],
            'type': trade_type,
            'date': date.strftime('%Y-%m-%d'),
            'entry_point': entry_price,
            'status': 'pending', # pending -> active -> closed
            'stop_loss': entry_price * (1 - sl_pct / 100),
            'target': entry_price * (1 + target_pct / 100)
        })

    def process_active_trades(self, minute_candle):
        if not self.active_trades: return

        for trade in self.active_trades:
            if trade['status'] == 'closed': continue

            symbol = trade['symbol']
            if symbol not in minute_candle: continue

            current_candle = minute_candle[symbol]
            high, low, current_time = current_candle['high'], current_candle['low'], current_candle['time']

            # Process pending trades to see if they trigger
            if trade['status'] == 'pending':
                if high >= trade['entry_point']:
                    trade['status'] = 'active'
                    trade['entry_time'] = current_time
                    print(f"\n{Fore.GREEN}Trade Triggered: {symbol} ({trade['type']}) at {trade['entry_point']:.2f}{Style.RESET_ALL}")

            # Process active trades for exit
            if trade['status'] == 'active':
                exit_reason, exit_price = None, None
                if low <= trade['stop_loss']:
                    exit_reason, exit_price = 'STOP_LOSS', trade['stop_loss']
                elif high >= trade['target']:
                    exit_reason, exit_price = 'TARGET', trade['target']
                elif current_time >= cfg.DAY_END_SQUARE_OFF:
                    exit_reason, exit_price = 'EOD_SQUARE_OFF', current_candle['close']

                if exit_reason:
                    trade['status'] = 'closed'
                    trade['exit_time'] = current_time
                    trade['exit_price'] = exit_price
                    trade['pnl_points'] = exit_price - trade['entry_point']
                    trade['pnl_percent'] = (trade['pnl_points'] / trade['entry_point']) * 100
                    self.trade_log.append(trade)
                    color = Fore.GREEN if trade['pnl_percent'] > 0 else Fore.RED
                    print(f"\n{color}Trade Closed: {symbol} | {exit_reason} | P&L: {trade['pnl_percent']:.2f}%{Style.RESET_ALL}")

    def square_off_open_trades(self, last_candles):
        for trade in self.active_trades:
            if trade['status'] == 'active':
                symbol = trade['symbol']
                if symbol in last_candles:
                    exit_price = last_candles[symbol]['close']
                    trade['status'] = 'closed'
                    trade['exit_time'] = last_candles[symbol]['time']
                    trade['exit_price'] = exit_price
                    trade['pnl_points'] = exit_price - trade['entry_point']
                    trade['pnl_percent'] = (trade['pnl_points'] / trade['entry_point']) * 100
                    self.trade_log.append(trade)
                    print(f"\n{Fore.YELLOW}EOD Square Off: {symbol} at {exit_price:.2f} | P&L: {trade['pnl_percent']:.2f}%{Style.RESET_ALL}")

    def generate_report(self):
        if not self.trade_log:
            print(f"\n{Fore.YELLOW}No trades were executed.{Style.RESET_ALL}")
            return

        report_df = pd.DataFrame(self.trade_log)
        print(f"\n{Fore.CYAN}{'='*20} BACKTEST REPORT {'='*20}{Style.RESET_ALL}")
        print(report_df[['date', 'symbol', 'type', 'entry_point', 'exit_price', 'pnl_percent', 'entry_time', 'exit_time']])

        # Summary
        win_rate = (report_df['pnl_percent'] > 0).mean() * 100
        avg_pnl = report_df['pnl_percent'].mean()
        print("\n--- Summary ---")
        print(f"Total Trades: {len(report_df)}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Average P&L: {avg_pnl:.2f}%")

        # Save to CSV
        filename = f"Trade_Log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        report_df.to_csv(filename, index=False)
        print(f"\nTrade log saved to {filename}")

# ============================================================================
# MAIN FUNCTION
# ============================================================================
def get_trading_day(days_back):
    """Get the specific trading day to backtest."""
    check_date = datetime.now().date() - timedelta(days=days_back)
    while check_date.weekday() >= 5: # Skip weekends
        check_date -= timedelta(days=1)
    return check_date

def main():
    """Main integrated function for the backtesting system."""
    print(f"\n{Fore.CYAN}EQUITY BACKTESTING SYSTEM STARTING...{Style.RESET_ALL}")

    # --- AUTHENTICATION ---
    access_token = FyersAuth().authenticate()
    if not access_token:
        print(f"{Fore.RED}Authentication failed. Exiting.{Style.RESET_ALL}")
        return

    fyers_data = FyersData(access_token)

    # --- STAGE 1: PRE-FILTERING ---
    all_symbols = pd.read_csv(cfg.SYMBOL_CSV)['fyers_symbol'].dropna().tolist()
    filtered_stocks, _ = PreFilterEngine(fyers_data).process_symbols(all_symbols)

    if not filtered_stocks:
        print(f"{Fore.YELLOW}No stocks passed the pre-filter. Exiting.{Style.RESET_ALL}")
        return

    pre_filtered_symbols = [s['symbol'] for s in filtered_stocks]
    pdh_map = {s['symbol']: s['pdh'] for s in filtered_stocks}

    # --- STAGE 2: INTRADAY BACKTESTING ---
    backtest_date = get_trading_day(cfg.BACKTEST_DAYS)

    # Fetch all data for the day upfront
    five_min_data = fyers_data.get_bulk_intraday_data(pre_filtered_symbols, backtest_date, cfg.CANDLE_INTERVAL)
    one_min_data = fyers_data.get_bulk_intraday_data(pre_filtered_symbols, backtest_date, cfg.MINUTE_RESOLUTION)

    # Save 5-min data to CSV
    all_5min_df = pd.concat([df.assign(symbol=sym) for sym, df in five_min_data.items() if not df.empty])
    if not all_5min_df.empty:
        hist_filename = f"Historical_Data_{backtest_date.strftime('%d%m%y')}.csv"
        all_5min_df.to_csv(hist_filename, index=False)
        print(f"5-minute historical data saved to {hist_filename}")

    # Initialize components
    scanner = BreakoutScanner()
    order_manager = OrderManager()

    # Create iterators for candle data
    five_min_candles_by_time = {time: {sym: row for sym, df in five_min_data.items() if not df.empty and time in df['time'].values and (row := df[df['time'] == time].iloc[0]) is not None} for time in sorted(all_5min_df['time'].unique())}
    one_min_candles_by_time = {time: {sym: row for sym, df in one_min_data.items() if not df.empty and time in df['time'].values and (row := df[df['time'] == time].iloc[0]) is not None} for time in sorted(pd.concat(one_min_data.values())['time'].unique())}

    print(f"\n{Fore.BLUE}Starting backtest loop for {backtest_date.strftime('%Y-%m-%d')}...{Style.RESET_ALL}")

    # Loop through 5-minute candles to find breakouts
    for candle_time, stocks_in_candle in five_min_candles_by_time.items():
        if candle_time > cfg.MARKET_CLOSE_TIME: break
        print(f"\rScanning 5-min candle: {candle_time}", end="")
        for symbol, candle in stocks_in_candle.items():
            trade_idea = scanner.scan({'symbol': symbol}, candle, pdh_map[symbol])
            if trade_idea:
                order_manager.add_potential_trade(trade_idea, backtest_date)

    # Loop through 1-minute candles to process trades
    print("\nProcessing trades on 1-minute data...")
    for candle_time, stocks_in_candle in one_min_candles_by_time.items():
        if candle_time > cfg.DAY_END_SQUARE_OFF: break
        order_manager.process_active_trades(stocks_in_candle)

    # Final square off for any remaining active trades
    last_candles = {sym: df.iloc[-1] for sym, df in one_min_data.items() if not df.empty}
    order_manager.square_off_open_trades(last_candles)

    # --- STAGE 3: REPORTING ---
    order_manager.generate_report()
    print(f"\n{Fore.CYAN}Backtest for {backtest_date.strftime('%Y-%m-%d')} complete.{Style.RESET_ALL}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Process interrupted by user.{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}An unexpected error occurred: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
