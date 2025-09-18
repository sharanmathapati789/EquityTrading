# Fyers API Credentials
# IMPORTANT: Please fill in your actual Fyers API credentials
CLIENT_ID = 'JUZ286UURR-100'  # Replace with your Client ID
SECRET_KEY = 'FTSBTVAEF0'      # Replace with your Secret Key
REDIRECT_URI = 'https://trade.fyers.in/api-login/redirect-to-app' # Keep this as is
RESPONSE_TYPE = 'code'
GRANT_TYPE = 'authorization_code'

# File Paths
SYMBOL_CSV = "symbols.csv" # Using local file for testing
ACCESS_TOKEN_FILE = "fyers_access_token.json"

# Pre-filter Criteria
MIN_PRICE = 23.0
MAX_PRICE = 2300.0
EMA_LENGTH = 15

# Backtest Settings
BACKTEST_DAYS = 1  # Number of past trading days to backtest
CANDLE_INTERVAL = "5"  # 5-minute candles for monitoring
MINUTE_RESOLUTION = "1"  # 1-minute data for detailed backtesting

# Initial Breakout Strategy Settings
INITIAL_REJECTION_THRESHOLD = 0.30  # 30% rejection rule for the first candle
INITIAL_TARGET_PERCENT = 1.2
INITIAL_SL_PERCENT = 0.6

# Stage Breakout Strategy Settings
STAGE_TARGET_PERCENT = 0.8
STAGE_SL_PERCENT = 0.4

# Market Timings (IST)
MARKET_OPEN_TIME = "09:15"
MARKET_CLOSE_TIME = "15:15"
DAY_END_SQUARE_OFF = "15:20" # Time to square off any open positions

# Spurt Indicator Settings (for 1-min data)
# NOTE: Added by agent as these were required by the backtester script but missing.
VOLUME_WINDOW = 20
ATR_WINDOW = 14
VOLUME_SPURT_FACTOR = 2.5
PRICE_SPURT_FACTOR = 1.5
