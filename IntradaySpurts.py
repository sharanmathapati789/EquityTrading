import os
import sys
import json
import time
import logging
import webbrowser
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Try to import Fyers SDK (newer version first)
try:
    from fyers_apiv3 import fyersModel

    FYERS_SDK_VERSION = "v3"
except ImportError:
    try:
        from fyers_api import fyersModel

        FYERS_SDK_VERSION = "v2"
    except ImportError:
        print("Error: No Fyers SDK found. Please install: pip install fyers-apiv3")
        sys.exit(1)

# === CONFIGURATION ===
SYMBOL_CSV = "/Users/Scripts/Equity/EQ_ST_EMA/Fyers/Equity_Trading/SmallCap250.csv"

# You need to set these values according to your Fyers API credentials
CLIENT_ID = 'JUZ286UURR-100'
SECRET_KEY = 'FTSBTVAEF0'
REDIRECT_URI = 'https://trade.fyers.in/api-login/redirect-to-app'
RESPONSE_TYPE = 'code'
GRANT_TYPE = 'authorization_code'

# Parameters
VOLUME_WINDOW = 14
ATR_WINDOW = 14
VOLUME_SPURT_FACTOR = 3
PRICE_SPURT_FACTOR = 1.5

# Authentication files
ACCESS_TOKEN_FILE = "fyers_access_token.json"

# Date Configuration
DATE_OPTIONS = {
    'TODAY': 0,
    'YESTERDAY': 1,
    'TWO_DAYS_BACK': 2,
    'THREE_DAYS_BACK': 3,
    'ONE_WEEK_BACK': 7,
    'CUSTOM': None
}

SELECTED_DATE_OPTION = 'TODAY'
CUSTOM_DATE = '2025-09-13'


class SimpleAuth:
    """Simplified Fyers authentication system"""

    def __init__(self):
        self.access_token = None
        self.token_file = ACCESS_TOKEN_FILE

    def load_token(self):
        """Load saved access token if valid"""
        try:
            if os.path.exists(self.token_file):
                with open(self.token_file, 'r') as f:
                    data = json.load(f)
                token = data.get('access_token')
                expiry = data.get('expiry', 0)
                if token and time.time() + 600 < expiry:
                    logging.info("Using saved access token")
                    return token
                else:
                    logging.info("Saved token expired")
            return None
        except Exception as e:
            logging.warning(f"Error loading token: {e}")
            return None

    def save_token(self, token, expiry_seconds=86400):
        """Save access token for future use"""
        try:
            data = {
                'access_token': token,
                'expiry': int(time.time() + expiry_seconds),
                'created': int(time.time())
            }
            with open(self.token_file, 'w') as f:
                json.dump(data, f, indent=2)
            logging.info("Access token saved")
        except Exception as e:
            logging.error(f"Failed to save token: {e}")

    def authenticate(self):
        """Main authentication flow"""
        saved_token = self.load_token()
        if saved_token and self.validate_token(saved_token):
            self.access_token = saved_token
            return saved_token

        print("\n" + "=" * 50)
        print("FYERS API AUTHENTICATION REQUIRED")
        print("=" * 50)

        try:
            session = fyersModel.SessionModel(
                client_id=CLIENT_ID,
                secret_key=SECRET_KEY,
                redirect_uri=REDIRECT_URI,
                response_type=RESPONSE_TYPE,
                grant_type=GRANT_TYPE
            )

            auth_url = session.generate_authcode()
            print(f"1. Opening browser to: {auth_url}")
            webbrowser.open(auth_url)
            print("2. Complete login process in browser")
            print("3. Copy authorization code from redirect URL")
            print("-" * 50)

            auth_code = input("Enter authorization code: ").strip()
            if not auth_code:
                print("No authorization code provided")
                return None

            session.set_token(auth_code)
            response = session.generate_token()

            if response and response.get('code') == 200:
                access_token = response['access_token']
                self.save_token(access_token)
                self.access_token = access_token
                print("Authentication successful!")
                return access_token
            else:
                print(f"Authentication failed: {response}")
                return None

        except Exception as e:
            logging.error(f"Authentication error: {e}")
            return None

    def validate_token(self, token):
        """Validate access token with Fyers API"""
        try:
            client = fyersModel.FyersModel(client_id=CLIENT_ID, token=token, log_path="")
            profile = client.get_profile()
            if profile.get('code') == 200:
                logging.info("Token validation successful")
                return True
            else:
                logging.warning(f"Token validation failed: {profile}")
                return False
        except Exception as e:
            logging.warning(f"Token validation error: {e}")
            return False


class FyersData:
    """Enhanced Fyers data fetching with granular calculations"""

    def __init__(self, access_token):
        self.client = fyersModel.FyersModel(client_id=CLIENT_ID, token=access_token, log_path="")
        self.rate_limit_delay = 0.3

    def validate_symbol_format(self, symbol):
        """Ensure proper symbol format for Fyers API"""
        symbol = symbol.strip()
        if symbol.startswith('NSE:') and symbol.endswith('-EQ'):
            return symbol
        if not symbol.startswith('NSE:'):
            symbol = symbol.replace('&', '%26')
            return f"NSE:{symbol}-EQ"
        return symbol

    def format_date_for_fyers(self, date_obj, time_str):
        """Format date for Fyers API"""
        dt_with_time = datetime.strptime(f"{date_obj.strftime('%Y-%m-%d')} {time_str}", '%Y-%m-%d %H:%M:%S')
        timestamp = int(dt_with_time.timestamp())
        return str(timestamp)

    def get_historical_data(self, symbol, target_date, retry_count=3):
        """Get historical data with improved date handling"""
        formatted_symbol = self.validate_symbol_format(symbol)

        start_time = "09:15:00"
        end_time = "15:30:00"

        date_formats_to_try = [
            (self.format_date_for_fyers(target_date, start_time),
             self.format_date_for_fyers(target_date, end_time)),
            (f"{target_date.strftime('%Y-%m-%d')}", f"{target_date.strftime('%Y-%m-%d')}"),
            (f"{target_date.strftime('%Y-%m-%d')}:{start_time}",
             f"{target_date.strftime('%Y-%m-%d')}:{end_time}"),
        ]

        for format_idx, (from_date, to_date) in enumerate(date_formats_to_try):
            for attempt in range(retry_count):
                try:
                    if attempt > 0:
                        time.sleep(self.rate_limit_delay * (attempt + 1))

                    data = {
                        "symbol": formatted_symbol,
                        "resolution": "1",
                        "date_format": "1",
                        "range_from": from_date,
                        "range_to": to_date,
                        "cont_flag": "1"
                    }

                    response = self.client.history(data)

                    if not response:
                        continue

                    if response.get('s') == 'ok':
                        candles = response.get("candles", [])
                        if candles:
                            data_list = []
                            for candle in candles:
                                try:
                                    timestamp = int(candle[0])
                                    if timestamp > 1e12:
                                        timestamp = timestamp // 1000

                                    data_list.append({
                                        "datetime": datetime.fromtimestamp(timestamp),
                                        "open": float(candle[1]),
                                        "high": float(candle[2]),
                                        "low": float(candle[3]),
                                        "close": float(candle[4]),
                                        "volume": float(candle[5]) if len(candle) > 5 else 0
                                    })
                                except (ValueError, IndexError):
                                    continue

                            if data_list:
                                df = pd.DataFrame(data_list)
                                df = df.sort_values('datetime').reset_index(drop=True)
                                return df, None

                    elif response.get('s') == 'no_data':
                        return pd.DataFrame(), "No data available for this date"
                    else:
                        error_msg = response.get('message', 'Unknown error')
                        if 'Invalid input' not in error_msg:
                            if attempt == retry_count - 1:
                                return pd.DataFrame(), f"API Error: {error_msg}"
                        break

                except Exception as e:
                    if attempt == retry_count - 1:
                        error_str = str(e)
                        if 'Invalid input' not in error_str:
                            return pd.DataFrame(), error_str
                        break

        return pd.DataFrame(), "All date formats failed"


def get_target_date():
    """Get the target date based on configuration"""
    if SELECTED_DATE_OPTION == 'CUSTOM':
        try:
            return datetime.strptime(CUSTOM_DATE, '%Y-%m-%d')
        except ValueError:
            print(f"Invalid CUSTOM_DATE format: {CUSTOM_DATE}. Using YESTERDAY instead.")
            days_back = 1
    else:
        days_back = DATE_OPTIONS.get(SELECTED_DATE_OPTION, 1)

    target_date = datetime.now() - timedelta(days=days_back)

    while target_date.weekday() > 4:
        target_date = target_date - timedelta(days=1)

    return target_date


def load_symbols():
    """Load symbols from CSV file"""
    try:
        df = pd.read_csv(SYMBOL_CSV)
        symbols = df['fyers_symbol'].dropna().tolist()
        return symbols
    except Exception as e:
        print(f"Error loading symbols: {e}")
        return []


def calculate_enhanced_spurts(df):
    """Calculate enhanced spurts with granular information"""
    if len(df) < max(VOLUME_WINDOW, ATR_WINDOW, 5):
        return pd.DataFrame()

    df = df.copy()

    # Store the day's opening price
    day_open = df.iloc[0]['open']

    # Calculate True Range and ATR
    df['previous_close'] = df['close'].shift(1)
    df['high_low'] = df['high'] - df['low']
    df['high_pc'] = abs(df['high'] - df['previous_close'])
    df['low_pc'] = abs(df['low'] - df['previous_close'])
    df['TR'] = df[['high_low', 'high_pc', 'low_pc']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=ATR_WINDOW, min_periods=ATR_WINDOW).mean()

    # Calculate Volume Moving Average
    df['Volume_MA'] = df['volume'].rolling(window=VOLUME_WINDOW, min_periods=VOLUME_WINDOW).mean()

    # Calculate Price Changes
    df['Price_Change'] = df['close'] - df['close'].shift(5)

    # Calculate percentage changes
    df['1min_pct_change'] = ((df['close'] - df['open']) / df['open']) * 100
    df['5min_pct_change'] = ((df['close'] - df['close'].shift(4)) / df['close'].shift(4)) * 100
    df['day_pct_change'] = ((df['close'] - day_open) / day_open) * 100

    # Calculate volume change from previous candle
    df['vol_change'] = df['volume'] - df['volume'].shift(1)
    df['vol_pct_change'] = ((df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)) * 100

    # Calculate cumulative volume for the day
    df['cumulative_volume'] = df['volume'].cumsum()

    # Identify Spurts
    df['Volume_Spurt'] = df['volume'] >= (df['Volume_MA'] * VOLUME_SPURT_FACTOR)
    df['Price_Spurt'] = abs(df['Price_Change']) >= (df['ATR'] * PRICE_SPURT_FACTOR)
    df['Spurt_Signal'] = df['Volume_Spurt'] & df['Price_Spurt']

    # Filter spurt signals
    spurt_df = df[df['Spurt_Signal'] == True].dropna()

    return spurt_df


def format_output_data(spurt_df, symbol):
    """Format data according to required columns"""
    if spurt_df.empty:
        return pd.DataFrame()

    result_data = []

    for _, row in spurt_df.iterrows():
        formatted_row = {
            'Symbol': symbol.replace('NSE:', '').replace('-EQ', ''),
            'LTP': f"{row['close']:.2f}",
            'Timestamp': row['datetime'].strftime('%H:%M:%S'),
            '1min%Chg': f"{row['1min_pct_change']:.2f}%",
            '5min%Chg': f"{row['5min_pct_change']:.2f}%",
            '%Chg': f"{row['day_pct_change']:.2f}%",
            'Volume': f"{int(row['cumulative_volume']):,}",
            'Vol%Chg': f"{row['vol_pct_change']:.2f}%"
        }
        result_data.append(formatted_row)

    return pd.DataFrame(result_data)


def display_professional_table(df):
    """Display data in professional tabular format"""
    if df.empty:
        return

    # Set pandas display options for better formatting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 15)

    print("\n" + "=" * 120)
    print("INTRADAY SPURTS DETECTION RESULTS")
    print("=" * 120)

    # Create formatted table
    table_str = df.to_string(index=False, justify='center')
    print(table_str)

    print("=" * 120)
    print(f"Total Spurt Signals Detected: {len(df)}")
    print(f"Unique Symbols with Spurts: {df['Symbol'].nunique()}")
    print("=" * 120)


def backtest_spurts():
    """Enhanced main function with granular output"""

    target_date = get_target_date()

    print("\n" + "=" * 60)
    print("ENHANCED INTRADAY SPURTS DETECTION SYSTEM")
    print("=" * 60)
    print(f"Selected Date Option: {SELECTED_DATE_OPTION}")
    print(f"Target Date: {target_date.strftime('%Y-%m-%d (%A)')}")
    print("=" * 60)

    # Authentication
    auth = SimpleAuth()
    access_token = auth.authenticate()
    if not access_token:
        print("ERROR: Authentication failed")
        return

    # Initialize data fetcher
    fyers_data = FyersData(access_token)

    # Load symbols
    symbols = load_symbols()
    if not symbols:
        print("No symbols loaded. Exiting.")
        return

    all_spurts_data = []
    stats = {'processed': 0, 'successful': 0, 'errors': 0, 'spurts_found': 0}

    print(f"\nProcessing {len(symbols)} symbols for spurt detection...")
    print("-" * 60)

    for i, symbol in enumerate(symbols, 1):
        print(f"Processing [{i:3d}/{len(symbols)}] {symbol:<25}", end=" ", flush=True)

        df, error = fyers_data.get_historical_data(symbol, target_date)

        if error:
            print(f"❌ {error[:30]}...")
            stats['errors'] += 1
        elif df is not None and len(df) > 0:
            stats['successful'] += 1
            spurt_df = calculate_enhanced_spurts(df)

            if not spurt_df.empty:
                formatted_data = format_output_data(spurt_df, symbol)
                if not formatted_data.empty:
                    all_spurts_data.append(formatted_data)
                    stats['spurts_found'] += len(formatted_data)
                    print(f"✅ {len(formatted_data)} spurts detected")
                else:
                    print("⚪ No valid spurts")
            else:
                print("⚪ No spurts detected")
        else:
            print("❌ No data available")
            stats['errors'] += 1

        stats['processed'] += 1
        time.sleep(0.2)

        # Progress update every 50 symbols
        if i % 50 == 0:
            success_rate = (stats['successful'] / stats['processed']) * 100
            print(f"\n--- Progress: {i}/{len(symbols)} | Success Rate: {success_rate:.1f}% ---")

    # Compile and display results
    date_str = target_date.strftime('%Y-%m-%d')

    if all_spurts_data:
        final_df = pd.concat(all_spurts_data, ignore_index=True)

        # Sort by timestamp for chronological display
        final_df['_timestamp'] = pd.to_datetime(final_df['Timestamp'], format='%H:%M:%S')
        final_df = final_df.sort_values('_timestamp').drop('_timestamp', axis=1).reset_index(drop=True)

        # Display professional table
        display_professional_table(final_df)

        # Save to CSV with enhanced filename
        output_file = f'enhanced_spurts_{date_str}.csv'
        # Save without formatting for Excel compatibility
        raw_df = final_df.copy()
        for col in ['LTP', '1min%Chg', '5min%Chg', '%Chg', 'Vol%Chg']:
            if col in raw_df.columns:
                raw_df[col] = raw_df[col].str.replace('%', '').str.replace(',', '').astype(float)
        raw_df.to_csv(output_file, index=False)

        print(f"\nDetailed results saved to: {output_file}")

    else:
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE - NO SPURTS DETECTED")
        print("=" * 60)
        print(f"Date Analyzed: {date_str}")
        print(f"Symbols Processed: {stats['processed']}")
        print(f"Successful Data Fetches: {stats['successful']}")
        print(f"Errors Encountered: {stats['errors']}")
        print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format='%(message)s')

    try:
        backtest_spurts()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nTroubleshooting checklist:")
        print("1. Verify CLIENT_ID and SECRET_KEY")
        print("2. Check internet connection")
        print("3. Ensure selected date is a trading day")
        print("4. Verify symbol CSV file path")
