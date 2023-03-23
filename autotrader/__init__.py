import urllib
import requests
import pandas as pd


def get_ticker_file():

    url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    data = urllib.request.urlopen(url).read().decode()
    df = pd.read_json(data)
    return df


def fetch_holidays():

    url = 'https://upstox.com/stocks-market/nse-bse-share-market-holiday-calendar-2023-india/'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                             'AppleWebKit/537.36 (KHTML, like Gecko) '
                             'Chrome/80.0.3987.132 Safari/537.36'}
    r = requests.get(url, headers=headers)

    holiday_df = pd.read_html(r.text)[0]
    holiday_df['Date'] = pd.to_datetime(holiday_df['Date'], format='%d %B %Y')
    holidays = holiday_df['Date'].values
    holidays = holidays.astype('datetime64[D]')
    return holidays


scrips = get_ticker_file()
holidays = fetch_holidays()
