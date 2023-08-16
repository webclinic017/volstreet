import urllib
import requests
import pandas as pd
import logging
from datetime import datetime
import os
import joblib
import ast


def get_ticker_file():
    url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    data = urllib.request.urlopen(url).read().decode()
    df = pd.read_json(data)
    return df


def fetch_holidays():
    url = "https://upstox.com/stocks-market/nse-bse-share-market-holiday-calendar-2023-india/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/80.0.3987.132 Safari/537.36"
    }
    r = requests.get(url, headers=headers)

    holiday_df = pd.read_html(r.text)[0]
    holiday_df["Date"] = pd.to_datetime(holiday_df["Date"], format="%d %B %Y")
    holidays = holiday_df["Date"].values
    holidays = holidays.astype("datetime64[D]")
    return holidays


def get_symbols():
    try:
        freeze_qty_url = "https://archives.nseindia.com/content/fo/qtyfreeze.xls"
        response = requests.get(freeze_qty_url, timeout=10)  # Set the timeout value
        response.raise_for_status()  # Raise an exception if the response contains an HTTP error status
        df = pd.read_excel(response.content)
        df.columns = df.columns.str.strip()
        df["SYMBOL"] = df["SYMBOL"].str.strip()
    except Exception as e:
        print(f"Error in fetching qtyfreeze.xls: {e}")
        df = pd.read_csv("data/qtyfreeze.csv")
        df.columns = df.columns.str.strip()
        df["SYMBOL"] = df["SYMBOL"].str.strip()
    return df


def load_rf_models():
    random_forest_models = {}
    for file in os.listdir("iv_models"):
        if file.endswith(".joblib"):
            str_literal = file.split("_")[-1].rstrip(".joblib")
            segment = ast.literal_eval(str_literal)
            random_forest_models[segment] = joblib.load(f"iv_models/{file}")
    return random_forest_models


def create_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    today = datetime.now().strftime("%Y-%m-%d")
    info_log_filename = f"error-{today}.log"
    info_handler = logging.FileHandler(info_log_filename)
    formatter = logging.Formatter(
        "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
    )
    info_handler.setFormatter(formatter)
    info_handler.setLevel(logging.INFO)
    logger.addHandler(info_handler)

    error_log_filename = f"info-{today}.log"
    error_handler = logging.FileHandler(error_log_filename)
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.ERROR)
    logger.addHandler(error_handler)
    return logger


# Get the list of scrips
scrips = get_ticker_file()
scrips["expiry_dt"] = pd.to_datetime(
    scrips[scrips.expiry != ""]["expiry"], format="%d%b%Y"
)
scrips["expiry_formatted"] = scrips["expiry_dt"].dt.strftime("%d%b%y")
scrips["expiry_formatted"] = scrips["expiry_formatted"].str.upper()

# Create a dictionary of token and symbol
token_symbol_dict = dict(zip(scrips["token"], scrips["symbol"]))

# Get the list of holidays
holidays = fetch_holidays()

# Get the list of symbols
symbol_df = get_symbols()

# Load the iv models
iv_models = load_rf_models()

# Create logger
logger = create_logger("volstreet")
