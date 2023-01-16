import urllib
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from math import exp, log, sqrt
from scipy.stats import norm
from time import sleep
import requests
import json
from smartapi import SmartConnect
import pyotp
from threading import Thread
from SmartWebSocketV2 import SmartWebSocketV2


def login(user, pin, apikey, authkey, webhook_url=None):
    global obj, login_data
    authkey = pyotp.TOTP(authkey)
    obj = SmartConnect(api_key=apikey)
    login_data = obj.generateSession(user, pin, authkey.now())
    if login_data['message'] != 'SUCCESS':
        for attempt in range(2, 7):
            sleep(10)
            notifier(f'Login attempt {attempt}.', webhook_url)
            login_data = obj.generateSession(user, pin, authkey.now())
            if login_data['message'] == 'SUCCESS':
                break
            if attempt == 6:
                notifier('Login failed.', webhook_url)
                raise Exception('Login failed.')
    notifier(f'Date: {currenttime().strftime("%d %b %Y %H:%M:%S")}\nLogged in successfully.', webhook_url)


def start_websocket(exchangetype=1, tokens=None):

    websocket_started = False

    if tokens is None:
        tokens = ['26000', '26009']

    global sws, price_dict
    price_dict = {}
    auth_token = login_data['data']['jwtToken']
    feed_token = obj.getfeedToken()
    sws = SmartWebSocketV2(auth_token, obj.api_key, obj.userId, feed_token)

    correlation_id = 'websocket'
    mode = 1
    token_list = [{'exchangeType': exchangetype, 'tokens': tokens}]

    def on_data(wsapp, message):
        price_dict[message['token']] = {'ltp': message['last_traded_price']/100,
                                        'timestamp': datetime.fromtimestamp(
                                            message['exchange_timestamp']/1000).strftime('%H:%M:%S')}

    def on_open(wsapp):
        nonlocal websocket_started
        print("Starting Websocket")
        sws.subscribe(correlation_id, mode, token_list)
        websocket_started = True

    def on_error(wsapp, error):
        print(error)

    def on_close(wsapp):
        print("Close")

    # Assign the callbacks.
    sws.on_open = on_open
    sws.on_data = on_data
    sws.on_error = on_error
    sws.on_close = on_close

    Thread(target=sws.connect).start()

    while not websocket_started:
        print('Waiting for websocket to start')
        sleep(1)


# Ticker file
def get_ticker_file():
    global scrips
    url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    data = urllib.request.urlopen(url).read().decode()
    scrips = pd.read_json(data)
    return scrips


def fetch_holidays():

    url = 'https://upstox.com/stocks-market/nse-bse-share-market-holiday-calendar-2023-india/'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                             'AppleWebKit/537.36 (KHTML, like Gecko) '
                             'Chrome/80.0.3987.132 Safari/537.36'}
    r = requests.get(url, headers=headers)

    holiday_df = pd.read_html(r.text)[0]
    holiday_df['Date'] = pd.to_datetime(holiday_df['Date'], format='%d %B %Y')
    holidays = holiday_df['Date'].values
    return holidays


def fetch_book(book):
    if book == 'orderbook':
        for attempt in range(1, 7):
            try:
                data = obj.orderBook()['data']
                return data
            except:
                if attempt == 6:
                    raise Exception('Failed to fetch orderbook.')
                else:
                    print(f'Error no {attempt}. Failed to fetch orderbook. Retrying in 2 seconds.')
                    sleep(2)

    elif book == 'positions' or book == 'position':
        for attempt in range(1, 7):
            try:
                data = obj.position()['data']
                return data
            except:
                if attempt == 6:
                    raise Exception('Failed to fetch positions.')
                else:
                    print(f'Error no {attempt}. Failed to fetch positions. Retrying in 2 seconds.')
                    sleep(2)


# Look up and return function for position and orderbook
def lookup_and_return(book, fieldtolookup, valuetolookup, fieldtoreturn):

    """Specify the dictionary as 'positions' or 'orderbook' or pass a dictionary. The valuetolookup can be
    a list or a single value. If provided a list, the function will return a numpy array of values. If provided a
    single value, the function will return a single value. Will return an empty array or 0 if the value is
    not found."""

    if isinstance(book, list):

        if isinstance(valuetolookup, list):

            bucket = [entry[fieldtoreturn] for entry in book
                      if entry[fieldtolookup] in valuetolookup
                      and entry[fieldtolookup] != '']

        elif isinstance(valuetolookup, str):

            bucket = [entry[fieldtoreturn] for entry in book
                      if entry[fieldtolookup] == valuetolookup
                      and entry[fieldtolookup] != '']
        else:
            raise ValueError('Invalid valuetolookup')

        # Logic for returning
        if isinstance(valuetolookup, list):
            assert len(bucket) == len(valuetolookup)
            return np.array(bucket)
        else:
            if len(bucket) == 0:
                return 0
            elif len(bucket) == 1:
                return bucket[0]
            else:
                return np.array(bucket)

    elif isinstance(book, str):

        for attempt in range(3):

            try:
                if book == 'orderbook':
                    if isinstance(valuetolookup, list):

                        bucket = [order[fieldtoreturn] for order in obj.orderBook()['data']
                                  if order[fieldtolookup] in valuetolookup and order[fieldtolookup] != '']
                    elif isinstance(valuetolookup, str):
                        bucket = [order[fieldtoreturn] for order in obj.orderBook()['data']
                                  if order[fieldtolookup] == valuetolookup and order[fieldtolookup] != '']
                    else:
                        raise ValueError('Invalid valuetolookup')

                elif book == 'positions':

                    if isinstance(valuetolookup, list):
                        bucket = [order[fieldtoreturn] for order in obj.position()['data']
                                  if order[fieldtolookup] in valuetolookup and order[fieldtolookup] != '']
                    elif isinstance(valuetolookup, str):
                        bucket = [order[fieldtoreturn] for order in obj.position()['data']
                                  if order[fieldtolookup] == valuetolookup and order[fieldtolookup] != '']
                    else:
                        raise ValueError('Invalid valuetolookup')
                else:
                    raise ValueError('Invalid dictionary')

            except Exception as e:
                if attempt == 2:
                    print(f'Error in lookup_and_return: {e}')
                else:
                    print(f'Error {attempt} in lookup_and_return: {e}\nRetrying again in 1 second')
                    sleep(1)

            # Logic for returning
            else:
                if isinstance(valuetolookup, list):
                    assert len(bucket) == len(valuetolookup)
                    return np.array(bucket)
                else:
                    if len(bucket) == 0:
                        return 0
                    elif len(bucket) == 1:
                        return bucket[0]
                    else:
                        return np.array(bucket)

    else:
        raise ValueError('Invalid dictionary')


# Discord messenger
def notifier(message, webhook_url=None):
    if webhook_url is None:
        print(message)
        return
    else:
        notification_url = webhook_url
        data = {'content': message}
        requests.post(notification_url, data=json.dumps(data), headers={'Content-Type': 'application/json'})
        print(message)
    return


# Market Hours
def markethours():
    if time(9, 10) < currenttime().time() < time(15, 30):
        return True
    else:
        return False


# Defining current time
def currenttime():
    return datetime.now()


def fetch_lot_size(name):
    return int(scrips.loc[(scrips.symbol.str.startswith(name)) & (scrips.exch_seg == 'NFO'), 'lotsize'].iloc[0])


def fetch_symbol_token(name):

    """Fetches symbol & token for a given scrip name. Provide just a single world if
    you want to fetch the symbol & token for the cash segment. If you want to fetch the
    symbol & token for the options segment, provide name in the format '{name} {strike} {expiry} {optiontype}'.
    Expiry should be in the DDMMMYY format. Optiontype should be CE or PE."""

    if len(name.split()) == 1:
        if name in ['BANKNIFTY', 'NIFTY']:
            symbol, token = scrips.loc[(scrips.name == name) & (scrips.exch_seg == 'NSE'), ['symbol', 'token']].values[
                0]
        elif name == 'FINNIFTY':
            futures = scrips.loc[(scrips.name == name) & (scrips.instrumenttype == 'FUTIDX'),
            ['expiry', 'symbol', 'token']]
            sorted_expiry_array = pd.to_datetime(futures.expiry, format='%d%b%Y').sort_values()
            futures = futures.loc[sorted_expiry_array.index]
            symbol, token = futures.iloc[0][['symbol', 'token']].values
        else:
            symbol, token = scrips.loc[
                (scrips.name == name) &
                (scrips.exch_seg == 'NSE') &
                (scrips.symbol.str.endswith('EQ')), ['symbol', 'token']
            ].values[0]
    elif len(name.split()) == 4:
        name, strike, expiry, optiontype = name.split()
        symbol = name + expiry + str(strike) + optiontype
        token = scrips[scrips.symbol == symbol]['token'].tolist()[0]

    else:
        raise ValueError('Invalid name')

    return symbol, token


# LTP function
def fetchltp(exchange_seg, symbol, token):
    for attempt in range(3):
        try:
            price = obj.ltpData(exchange_seg, symbol, token)['data']['ltp']
            return price
        except Exception as e:
            if attempt == 2:
                print(f'Error in fetchltp: {e}')
            else:
                print(f'Error {attempt} in fetchltp: {e}\nRetrying again in 1 second')
                sleep(1)


def fetchpreviousclose(exchange_seg, symbol, token):
    for attempt in range(3):
        try:
            previousclose = obj.ltpData(exchange_seg, symbol, token)['data']['close']
            return previousclose
        except Exception as e:
            if attempt == 2:
                print(f'Error in fetchpreviousclose: {e}')
            else:
                print(f'Error {attempt} in fetchpreviousclose: {e}\nRetrying again in 1 second')
                sleep(1)


def fetch_straddle_price(name, expiry, strike, return_total_price=False):

    """Fetches the price of the straddle for a given name, expiry and strike. Expiry should be in the DDMMMYY format.
    If return_total_price is True, then the total price of the straddle is returned. If return_total_price is False,
    then the price of the call and put is returned as a tuple."""

    call_symbol, call_token = fetch_symbol_token(f'{name} {strike} {expiry} CE')
    put_symbol, put_token = fetch_symbol_token(f'{name} {strike} {expiry} PE')
    call_ltp = fetchltp('NFO', call_symbol, call_token)
    put_ltp = fetchltp('NFO', put_symbol, put_token)
    if return_total_price:
        return call_ltp + put_ltp
    else:
        return call_ltp, put_ltp


def fetch_strangle_price(name, expiry, call_strike, put_strike, return_total_price=False):

    """Fetches the price of the strangle for a given name, expiry and strike. Expiry should be in the DDMMMYY format.
    If return_total_price is True, then the total price of the strangle is returned. If return_total_price is False,
    then the price of the call and put is returned as a tuple."""

    call_symbol, call_token = fetch_symbol_token(f'{name} {call_strike} {expiry} CE')
    put_symbol, put_token = fetch_symbol_token(f'{name} {put_strike} {expiry} PE')
    call_ltp = fetchltp('NFO', call_symbol, call_token)
    put_ltp = fetchltp('NFO', put_symbol, put_token)
    if return_total_price:
        return call_ltp + put_ltp
    else:
        return call_ltp, put_ltp


# Finding ATM strike
def findstrike(x, base):
    return base * round(x / base)


# BLACK SCHOLES BELOW #

def d1(S, K, T, r, sigma):
    return (log(S / K) + (r + sigma ** 2 / 2.) * T) / (sigma * sqrt(T))


def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * sqrt(T)


def bs_call(S, K, T, r, sigma):
    return S * norm.cdf(d1(S, K, T, r, sigma)) - K * exp(-r * T) * norm.cdf(d2(S, K, T, r, sigma))


def bs_put(S, K, T, r, sigma):
    return K * exp(-r * T) - S * bs_call(S, K, T, r, sigma)


def call_implied_volatility(Price, S, K, T, r):
    sigma = 0.001
    while sigma < 1:
        Price_implied = S * \
                        norm.cdf(d1(S, K, T, r, sigma)) - K * exp(-r * T) * \
                        norm.cdf(d2(S, K, T, r, sigma))
        if Price - Price_implied < 0.01:
            return sigma * 100
        sigma += 0.001
    return 100


def put_implied_volatility(Price, S, K, T, r):
    sigma = 0.001
    while sigma < 1:
        Price_implied = K * exp(-r * T) - S + bs_call(S, K, T, r, sigma)
        if Price - Price_implied < 0.01:
            return sigma * 100
        sigma += 0.001
    return 100


def call_delta(S, K, T, r, sigma):
    delta = norm.cdf(d1(S, K, T, r, sigma))
    return round(delta, 5)


def call_gamma(S, K, T, r, sigma):
    gamma = norm.pdf(d1(S, K, T, r, sigma)) / (S * sigma * sqrt(T))
    return round(gamma, 5)


def put_delta(S, K, T, r, sigma):
    return -norm.cdf(-d1(S, K, T, r, sigma))


def put_gamma(S, K, T, r, sigma):
    return norm.pdf(d1(S, K, T, r, sigma)) / (S * sigma * sqrt(T))


def timetoexpiry(expiry):
    """Return time left to expiry"""
    time_to_expiry = ((datetime.strptime(expiry, '%d%b%y') + pd.DateOffset(minutes=930)) - currenttime()) / timedelta(
        days=365)
    return time_to_expiry


def straddleiv(callprice, putprice, spot, strike, timeleft):
    call_iv = call_implied_volatility(callprice, spot, strike, timeleft, 0.05)
    put_iv = put_implied_volatility(putprice, spot, strike, timeleft, 0.05)
    avg_iv = (call_iv + put_iv) / 2

    return round(avg_iv, 2)


def fetch_price_iv_delta(position_string, underlying_price=None):
    """Fetches the price, iv and delta of a stock"""

    name, strike, expiry, option_type = position_string.split()
    strike = int(strike)
    time_left = timetoexpiry(expiry)

    if underlying_price is None:
        underlying_symbol, underlying_token = fetch_symbol_token(name)
        underlying_price = fetchltp('NSE', underlying_symbol, underlying_token)

    position_symbol, position_token = fetch_symbol_token(position_string)
    position_price = fetchltp('NFO', position_symbol, position_token)

    if option_type == 'CE':
        iv = call_implied_volatility(position_price, underlying_price, strike, time_left, 0.05)
        delta = call_delta(underlying_price, strike, time_left, 0.05, iv / 100)
    elif option_type == 'PE':
        iv = put_implied_volatility(position_price, underlying_price, strike, time_left, 0.05)
        delta = put_delta(underlying_price, strike, time_left, 0.05, iv / 100)
    else:
        raise Exception('Invalid option type')

    return position_price, iv, delta


def place_synthetic_fut(name, strike, expiry, buy_or_sell, quantity, price='MARKET'):
    """Places a synthetic future order. Quantity is in number of shares."""

    call_symbol, call_token = fetch_symbol_token(f'{name} {strike} {expiry} CE')
    put_symbol, put_token = fetch_symbol_token(f'{name} {strike} {expiry} PE')

    if buy_or_sell == 'BUY':
        order_id_call = placeorder(call_symbol, call_token, quantity, 'BUY', 'MARKET')
        order_id_put = placeorder(put_symbol, put_token, quantity, 'SELL', 'MARKET')
    elif buy_or_sell == 'SELL':
        order_id_call = placeorder(call_symbol, call_token, quantity, 'SELL', 'MARKET')
        order_id_put = placeorder(put_symbol, put_token, quantity, 'BUY', 'MARKET')
    else:
        raise Exception('Invalid buy or sell')

    order_statuses = lookup_and_return('orderbook', 'orderid', [order_id_call, order_id_put], 'status')

    if not all(order_statuses == 'complete'):
        raise Exception('Syntehtic Futs: Orders not completed')
    else:
        print(f'Synthetic Futs: {buy_or_sell} Order for {quantity} quantity completed.')


# ORDER FUNCTIONS BELOW #

def placeorder(symbol, token, qty, buyorsell, orderprice, ordertag=""):
    """Provide symbol, token, qty (shares), buyorsell, orderprice, ordertag (optional)"""

    if orderprice == 'MARKET':

        params = {"variety": "NORMAL",
                  "tradingsymbol": symbol,
                  "symboltoken": token,
                  "transactiontype": buyorsell,
                  "exchange": "NFO",
                  "ordertype": "MARKET",
                  "producttype": "CARRYFORWARD",
                  "duration": "DAY",
                  "price": 0,
                  "squareoff": "0",
                  "stoploss": "0",
                  "quantity": int(qty),
                  "ordertag": ordertag}

    else:

        if buyorsell == 'BUY' and orderprice < 1:
            orderprice = 1
        elif buyorsell == 'SELL' and orderprice < 1:
            orderprice = 0.05

        params = {"variety": "NORMAL",
                  "tradingsymbol": symbol,
                  "symboltoken": token,
                  "transactiontype": buyorsell,
                  "exchange": "NFO",
                  "ordertype": "LIMIT",
                  "producttype": "CARRYFORWARD",
                  "duration": "DAY",
                  "price": round(orderprice, 1),
                  "squareoff": "0",
                  "stoploss": "0",
                  "quantity": int(qty),
                  "ordertag": ordertag}

    order_id = obj.placeOrder(params)
    return order_id


def placeSLorder(symbol, token, qty, buyorsell, triggerprice, ordertag=""):
    executionprice = triggerprice * 1.1

    params = {"variety": "STOPLOSS",
              "tradingsymbol": symbol,
              "symboltoken": token,
              "transactiontype": buyorsell,
              "exchange": "NFO",
              "ordertype": "STOPLOSS_LIMIT",
              "producttype": "CARRYFORWARD",
              "duration": "DAY",
              "triggerprice": round(triggerprice, 1),
              "price": round(executionprice, 1),
              "squareoff": "0",
              "stoploss": "0",
              "quantity": int(qty),
              "ordertag": ordertag}

    order_id = obj.placeOrder(params)
    return order_id


class Index:

    """Initialize an index with the name of the index in uppercase"""

    def __init__(self, name, webhook_url=None, subscribe_to_ws=False):

        self.name = name
        self.ltp = None
        self.previous_close = None
        self.current_strike = None
        self.previous_strike = None
        self.order_list = []
        self.current_expiry = None
        self.next_expiry = None
        self.month_expiry = None
        self.fut_expiry = None
        self.points_captured = None
        self.stoploss = None
        self.webhook_url = webhook_url

        self.symbol, self.token = fetch_symbol_token(name)
        self.lot_size = fetch_lot_size(name)
        self.fetch_expirys()
        self.freeze_qty = self.fetch_freeze_limit()

        if self.name == 'BANKNIFTY':
            self.base = 100
            self.exchange_type = 1
        elif self.name == 'NIFTY':
            self.base = 50
            self.exchange_type = 1
        elif self.name == 'FINNIFTY':
            self.base = 50
            self.exchange_type = 2
        else:
            raise ValueError('Index name not valid')

        if subscribe_to_ws:
            try:
                sws.subscribe('websocket', 1, [{'exchangeType': self.exchange_type, 'tokens': [self.token]}])
                print(f'{self.name}: Subscribed underlying to the websocket')
            except NameError:
                print('Websocket not initialized. Please initialize the websocket before subscribing to it.')

    def fetch_freeze_limit(self):
        freeze_qty_url = 'https://www1.nseindia.com/content/fo/qtyfreeze.xls'
        df = pd.read_excel(freeze_qty_url)
        df.columns = df.columns.str.strip()
        df['SYMBOL'] = df['SYMBOL'].str.strip()
        freeze_qty = df[df['SYMBOL'] == self.name]['VOL_FRZ_QTY'].values[0]
        freeze_qty_in_lots = freeze_qty / self.lot_size
        return freeze_qty_in_lots

    def fetch_expirys(self):

        """Fetch Specified Expiry"""

        expirymask = (scrips.expiry != '') & (scrips.exch_seg == 'NFO') & (scrips.name == self.name)
        expirydates = pd.to_datetime(scrips[expirymask].expiry).astype('datetime64[ns]').sort_values().unique()
        allexpiries = [(pd.to_datetime(exps) + pd.DateOffset(minutes=930)) for exps in expirydates]
        allexpiries = [*filter(lambda expiry: expiry > currenttime(), allexpiries)]
        allexpiries = [*map(lambda expiry: expiry.strftime('%d%b%y').upper(), allexpiries)]

        expiry_month_list = [int(datetime.strptime(i[2:5], '%b').strftime('%m')) for i in allexpiries]
        monthmask = np.array([second - first for first, second in zip(expiry_month_list, expiry_month_list[1:])])
        monthmask = np.where(monthmask == 0, monthmask, 1)

        monthexpiries = [b for a, b in zip(monthmask, allexpiries) if a == 1]

        currentexpiry = allexpiries[0]
        nextexpiry = allexpiries[1]

        if monthexpiries[0] == allexpiries[0]:
            monthexpiry = monthexpiries[1]
            futexpiry = allexpiries[0]
        else:
            monthexpiry = monthexpiries[0]
            futexpiry = monthexpiries[0]

        self.current_expiry = currentexpiry
        self.next_expiry = nextexpiry
        self.month_expiry = monthexpiry
        self.fut_expiry = futexpiry

    def fetch_ltp(self):
        """Fetch LTP of the index. Uses futures for FINNIFTY"""
        if self.name == 'FINNIFTY':
            self.ltp = fetchltp('NFO', self.symbol, self.token)
        else:
            self.ltp = fetchltp('NSE', self.symbol, self.token)
        return self.ltp

    def fetch_previous_close(self):
        self.previous_close = fetchpreviousclose('NSE', self.symbol, self.token)
        return self.previous_close

    def place_straddle_order(self, strike, expiry, buy_or_sell, quantity_in_lots, return_avg_price=False, order_tag=""):

        """
        Place a straddle order on the index.

        Params:
        strike: Strike price of the option
        expiry: Expiry of the option
        buy_or_sell: BUY or SELL
        quantity_in_lots: Quantity in lots
        return_avg_price: If True, returns the average price of the order
        order_tag: Tag to be added to the order"""

        call_symbol, call_token = fetch_symbol_token(f'{self.name} {strike} {expiry} CE')
        put_symbol, put_token = fetch_symbol_token(f'{self.name} {strike} {expiry} PE')
        call_price = fetchltp('NFO', call_symbol, call_token)
        put_price = fetchltp('NFO', put_symbol, put_token)

        limit_price_extender = 1.1 if buy_or_sell == 'BUY' else 0.9

        if quantity_in_lots > self.freeze_qty:
            loops = int(quantity_in_lots / self.freeze_qty)
            if loops > 10:
                raise Exception('Order too big. This error was raised to prevent accidental large order placement.')

            remainder = quantity_in_lots % self.freeze_qty
            spliced_orders = [self.freeze_qty] * loops + [remainder]
        else:
            spliced_orders = [quantity_in_lots]

        call_order_id_list = []
        put_order_id_list = []
        for quantity in spliced_orders:

            call_order_id = placeorder(call_symbol, call_token,
                                       quantity * self.lot_size,
                                       buy_or_sell, call_price * limit_price_extender,
                                       ordertag=order_tag)
            put_order_id = placeorder(put_symbol, put_token,
                                      quantity * self.lot_size,
                                      buy_or_sell, put_price * limit_price_extender,
                                      ordertag=order_tag)
            call_order_id_list.append(call_order_id)
            put_order_id_list.append(put_order_id)
            sleep(0.3)

        orderbook = fetch_book('orderbook')

        call_order_statuses = lookup_and_return(orderbook, 'orderid', call_order_id_list, 'status')
        put_order_statuses = lookup_and_return(orderbook, 'orderid', put_order_id_list, 'status')

        if all(call_order_statuses == 'complete') and all(put_order_statuses == 'complete'):
            notifier(f'{order_tag}: Order(s) placed successfully for {buy_or_sell} {self.name} ' +
                     f'{strike} {expiry} {quantity_in_lots} lot(s).', self.webhook_url)
            call_order_avg_price = lookup_and_return(orderbook, 'orderid',
                                                     call_order_id_list, 'averageprice').astype(float).mean()
            put_order_avg_price = lookup_and_return(orderbook, 'orderid',
                                                    put_order_id_list, 'averageprice').astype(float).mean()
            order_log = {'Date': currenttime().strftime('%d-%m-%Y %H:%M:%S'), 'Index': self.name,
                         'Put Strike': strike, 'Call Strike': strike, 'Expiry': expiry,
                         'Action Type': buy_or_sell, 'Call Price': call_order_avg_price,
                         'Put Price': put_order_avg_price, 'Total Price': call_order_avg_price + put_order_avg_price,
                         'Tag': order_tag}
            self.order_list.append(order_log)
            if return_avg_price:
                return call_order_avg_price, put_order_avg_price
            else:
                return

        elif all(call_order_statuses == 'rejected') and all(put_order_statuses == 'rejected'):
            notifier(f'{order_tag}: All orders rejected for {buy_or_sell} {self.name} ' +
                     f'{strike} {expiry} {quantity_in_lots} lot(s).', self.webhook_url)
            raise Exception('Orders rejected')

        else:
            notifier(f'{order_tag}: ERROR. Order statuses uncertain for {buy_or_sell} {self.name} ' +
                     f'{strike} {expiry} {quantity_in_lots} lot(s).', self.webhook_url)
            raise Exception('Order statuses uncertain')

    def place_strangle_order(self, call_strike, put_strike, expiry, buy_or_sell, quantity_in_lots,
                             return_avg_price=False, order_tag=""):

        """
        Places a strangle order on the index.

        Params:
        call_strike: Strike price of the call
        put_strike: Strike price of the put
        expiry: Expiry of the option
        buy_or_sell: BUY or SELL
        quantity_in_lots: Quantity in lots
        return_avg_price: If True, returns the average price of the order
        order_tag: Tag to be added to the order"""


        call_symbol, call_token = fetch_symbol_token(f'{self.name} {call_strike} {expiry} CE')
        put_symbol, put_token = fetch_symbol_token(f'{self.name} {put_strike} {expiry} PE')
        call_price = fetchltp('NFO', call_symbol, call_token)
        put_price = fetchltp('NFO', put_symbol, put_token)

        limit_price_extender = 1.1 if buy_or_sell == 'BUY' else 0.9

        if quantity_in_lots > self.freeze_qty:
            loops = int(quantity_in_lots / self.freeze_qty)
            if loops > 10:
                raise Exception('Order too big. This error was raised to prevent accidental large order placement.')

            remainder = quantity_in_lots % self.freeze_qty
            spliced_orders = [self.freeze_qty] * loops + [remainder]
        else:
            spliced_orders = [quantity_in_lots]

        call_order_id_list = []
        put_order_id_list = []
        for quantity in spliced_orders:
            call_order_id = placeorder(call_symbol, call_token,
                                       quantity * self.lot_size,
                                       buy_or_sell, call_price * limit_price_extender,
                                       ordertag=order_tag)
            put_order_id = placeorder(put_symbol, put_token,
                                      quantity * self.lot_size,
                                      buy_or_sell, put_price * limit_price_extender,
                                      ordertag=order_tag)
            call_order_id_list.append(call_order_id)
            put_order_id_list.append(put_order_id)
            sleep(0.3)

        orderbook = fetch_book('orderbook')

        call_order_statuses = lookup_and_return(orderbook, 'orderid', call_order_id_list, 'status')
        put_order_statuses = lookup_and_return(orderbook, 'orderid', put_order_id_list, 'status')

        if all(call_order_statuses == 'complete') and all(put_order_statuses == 'complete'):
            notifier(f'{order_tag}: Order(s) placed successfully for {buy_or_sell} {self.name} ' +
                     f'{call_strike} CE and {put_strike} PE ' +
                     f'{expiry} {quantity_in_lots} lot(s).', self.webhook_url)
            call_order_avg_price = lookup_and_return(orderbook, 'orderid',
                                                     call_order_id_list, 'averageprice').astype(float).mean()
            put_order_avg_price = lookup_and_return(orderbook, 'orderid',
                                                    put_order_id_list, 'averageprice').astype(float).mean()
            order_log = {'Date': currenttime().strftime('%d-%m-%Y %H:%M:%S'), 'Index': self.name,
                         'Put Strike': put_strike, 'Call Strike': call_strike, 'Expiry': expiry,
                         'Action Type': buy_or_sell, 'Call Price': call_order_avg_price,
                         'Put Price': put_order_avg_price, 'Total Price': call_order_avg_price + put_order_avg_price,
                         'Tag': order_tag}
            self.order_list.append(order_log)
            if return_avg_price:
                return call_order_avg_price, put_order_avg_price
            else:
                return

        elif all(call_order_statuses == 'rejected') and all(put_order_statuses == 'rejected'):
            notifier(f'{order_tag}: All orders rejected for {buy_or_sell} {self.name} ' +
                     f'{call_strike} CE and {put_strike} PE ' +
                     f'{expiry} {quantity_in_lots} lot(s).', self.webhook_url)
            raise Exception('Orders rejected')

        else:
            notifier(f'{order_tag}: ERROR. Order statuses uncertain for {buy_or_sell} {self.name} ' +
                     f'{call_strike} CE and {put_strike} PE ' +
                     f'{expiry} {quantity_in_lots} lot(s).', self.webhook_url)
            raise Exception('Order statuses uncertain')

    def rollover_overnight_short_straddle(self, quantity_in_lots, strike_offset=1):

        """Buys the previous day's strike and sells the current strike"""

        # Deciding which expirys to trade
        daystocurrentexpiry = timetoexpiry(self.current_expiry) * 365
        if 2 > daystocurrentexpiry > 1:
            expirytobuy = self.current_expiry
            expirytosell = self.next_expiry
        elif daystocurrentexpiry < 1:
            expirytobuy = self.next_expiry
            expirytosell = self.next_expiry
        else:
            expirytobuy = self.current_expiry
            expirytosell = self.current_expiry

        # Setting Strikes
        sell_strike = findstrike(self.fetch_ltp() * strike_offset, self.base)
        buy_strike = findstrike(self.fetch_previous_close() * strike_offset, self.base)

        # Matching estimated previous strike with actual sold strike
        with open(f"{self.name}_daily_sold_strike.txt", "r") as file:
            sold_strike = int(file.read())
        if sold_strike != buy_strike:
            notifier(f'Estimated sold strike {buy_strike} does not match ' +
                     f'actual sold strike {sold_strike}. Changing strike to buy to {sold_strike}. ', self.webhook_url)
            buy_strike = sold_strike

        # Placing orders
        if buy_strike == sell_strike and expirytobuy == expirytosell:
            notifier('No trade today.', self.webhook_url)
            call_symbol, call_token = fetch_symbol_token(f'{self.name} {sell_strike} {expirytosell} CE')
            put_symbol, put_token = fetch_symbol_token(f'{self.name} {sell_strike} {expirytosell} PE')
            call_ltp = fetchltp('NFO', call_symbol, call_token)
            put_ltp = fetchltp('NFO', put_symbol, put_token)
            order_log = {'Date': currenttime().strftime('%d-%m-%Y %H:%M:%S'), 'Index': self.name,
                         'Put Strike': buy_strike, 'Call Strike': buy_strike, 'Expiry': expirytobuy,
                         'Action Type': 'No trade required', 'Call Price': call_ltp,
                         'Put Price': put_ltp, 'Total Price': call_ltp + put_ltp,
                         'Tag': 'Daily overnight short straddle'}
            self.order_list.append(order_log)
        else:
            notifier(f'Buying {buy_strike} and selling {sell_strike}.', self.webhook_url)
            self.place_straddle_order(buy_strike, expirytobuy, 'BUY', quantity_in_lots,
                                      order_tag='Daily overnight short straddle')
            self.place_straddle_order(sell_strike, expirytosell, 'SELL', quantity_in_lots,
                                      order_tag='Daily overnight short straddle')

        # Updating daily_sold_strike.txt
        with open(f"{self.name}_daily_sold_strike.txt", "w") as file:
            file.write(str(sell_strike))
        return

    def buy_weekly_hedge(self, quantity_in_lots, type_of_hedge='strangle', **kwargs):

        ltp = self.fetch_ltp()
        if type_of_hedge == 'strangle':
            call_strike = findstrike(ltp * kwargs['call_offset'], self.base)
            put_strike = findstrike(ltp * kwargs['put_offset'], self.base)
            self.place_strangle_order(call_strike, put_strike, self.next_expiry,
                                      'BUY', quantity_in_lots, order_tag='Weekly hedge')
        elif type_of_hedge == 'straddle':
            strike = findstrike(ltp * kwargs['strike_offset'], self.base)
            self.place_straddle_order(strike, self.next_expiry,
                                      'BUY', quantity_in_lots, order_tag='Weekly hedge')

    def intraday_straddle(self, quantity_in_lots, exit_time=(15, 28), wait_for_equality=False,
                          monitor_sl=False, **kwargs):

        """Params:
        quantity_in_lots: Quantity of straddle to trade
        exit_time: Time to exit the trade
        wait_for_equality: If True, waits for call and put prices to be equal before placing orders
        monitor_sl: If True, monitors stop loss and moves sl to cost on the other leg if one leg is hit
        kwargs: 'target_disparity', 'stoploss'
        """

        if quantity_in_lots > self.freeze_qty:
            loops = int(quantity_in_lots / self.freeze_qty)
            if loops > 10:
                raise Exception('Order too big. This error was raised to prevent accidental large order placement.')
            remainder = quantity_in_lots % self.freeze_qty
            spliced_orders = [self.freeze_qty] * loops + [remainder]
        else:
            spliced_orders = [quantity_in_lots]

        def scanner():

            """Scans the market for the best strike to trade"""

            ltp = price_dict.get(self.token, 0)['ltp']
            current_strike = findstrike(ltp, self.base)
            strike_range = np.arange(current_strike - self.base * 6, current_strike + self.base * 6, self.base)

            call_token_list = []
            put_token_list = []
            call_symbol_list = []
            put_symbol_list = []
            strike_list = []
            for strike in strike_range:
                call_symbol, call_token = fetch_symbol_token(f'{self.name} {strike} {self.current_expiry} CE')
                put_symbol, put_token = fetch_symbol_token(f'{self.name} {strike} {self.current_expiry} PE')
                call_token_list.append(call_token)
                put_token_list.append(put_token)
                call_symbol_list.append(call_symbol)
                put_symbol_list.append(put_symbol)
                strike_list.append(strike)

            call_and_put_token_list = call_token_list + put_token_list
            token_list_subscribe = [{'exchangeType': 2, 'tokens': call_and_put_token_list}]
            sws.subscribe('websocket', 1, token_list_subscribe)
            sleep(3)

            # Fetching the last traded prices of different strikes from the global price_dict using token values
            call_ltps = np.array([price_dict.get(call_token, {'ltp': 0})['ltp'] for call_token in call_token_list])
            put_ltps = np.array([price_dict.get(put_token, {'ltp': 0})['ltp'] for put_token in put_token_list])
            disparities = np.abs(call_ltps - put_ltps)/np.minimum(call_ltps, put_ltps)*100

            # If wait_for_equality is True, waits for call and put prices to be equal before selecting a strike
            if wait_for_equality:

                loop_number = 1
                while np.min(disparities) > kwargs['target_disparity']:
                    call_ltps = np.array([price_dict.get(call_token, {'ltp': 0})['ltp'] for call_token in call_token_list])
                    put_ltps = np.array([price_dict.get(put_token, {'ltp': 0})['ltp'] for put_token in put_token_list])
                    disparities = np.abs(call_ltps - put_ltps)/np.minimum(call_ltps, put_ltps)*100
                    if loop_number % 200000 == 0:
                        print(f'Time: {currenttime().strftime("%H:%M:%S")}\n' +
                              f'Index: {self.name}\n' +
                              f'Current lowest disparity: {np.min(disparities):.2f}\n' +
                              f'Strike: {strike_list[np.argmin(disparities)]}\n')
                    if (currenttime() + timedelta(minutes=5)).time() > time(*exit_time):
                        notifier('Intraday straddle exited due to time limit.', self.webhook_url)
                        return
                    loop_number += 1

            # Selecting the strike with the lowest disparity
            strike_to_trade = strike_list[np.argmin(disparities)]
            call_symbol = call_symbol_list[np.argmin(disparities)]
            put_symbol = put_symbol_list[np.argmin(disparities)]
            call_token = call_token_list[np.argmin(disparities)]
            put_token = put_token_list[np.argmin(disparities)]
            call_ltp = call_ltps[np.argmin(disparities)]
            put_ltp = put_ltps[np.argmin(disparities)]

            # Unsubscribing from the tokens
            tokens_to_unsubscribe = [token for token in call_and_put_token_list if token not in [call_token, put_token]]
            token_list_unsubscribe = [{'exchangeType': 2, 'tokens': tokens_to_unsubscribe}]
            sws.unsubscribe('websocket', 1, token_list_unsubscribe)
            for token in tokens_to_unsubscribe:
                del price_dict[token]
            print(f'{self.name}: Unsubscribed from tokens')
            return strike_to_trade, call_symbol, put_symbol, call_token, put_token, call_ltp, put_ltp

        equal_strike, call_symbol, put_symbol, call_token, put_token, call_price, put_price = scanner()
        expiry = self.current_expiry
        #print(f'Index: {self.name}, Strike: {equal_strike}, Call: {call_price}, Put: {put_price}')
        notifier(f'{self.name}: Initiating intraday trade on {equal_strike} strike.', self.webhook_url)

        # Placing orders
        call_avg_price, put_avg_price = self.place_straddle_order(equal_strike, expiry, 'SELL', quantity_in_lots,
                                                                  return_avg_price=True, order_tag='Intraday straddle')

        if 'stoploss' in kwargs:
            sl = kwargs['stoploss']
        else:
            if timetoexpiry(self.current_expiry) * 365 < 1 or self.name == 'BANKNIFTY':
                sl = 1.7
            else:
                sl = 1.5

        # Placing stoploss orders
        call_stoploss_order_ids = []
        put_stoploss_order_ids = []
        for quantity in spliced_orders:
            call_sl_order_id = placeSLorder(call_symbol, call_token, quantity * self.lot_size,
                                            'BUY', call_avg_price * sl, 'Stoploss call')
            put_sl_order_id = placeSLorder(put_symbol, put_token, quantity * self.lot_size,
                                           'BUY', put_avg_price * sl, 'Stoploss put')
            call_stoploss_order_ids.append(call_sl_order_id)
            put_stoploss_order_ids.append(put_sl_order_id)
            sleep(0.3)

        orderbook = fetch_book('orderbook')
        call_sl_statuses = lookup_and_return(orderbook, 'orderid', call_stoploss_order_ids, 'status')
        put_sl_statuses = lookup_and_return(orderbook, 'orderid', put_stoploss_order_ids, 'status')

        if all(call_sl_statuses == 'trigger pending') and all(put_sl_statuses == 'trigger pending'):
            notifier(f'{self.name} stoploss orders placed successfully', self.webhook_url)
        else:
            notifier(f'{self.name} stoploss orders not placed successfully', self.webhook_url)
            raise Exception('Stoploss orders not placed successfully')

        summary_message = '\n'.join(f'{k}: {v}' for k, v in self.order_list[0].items())
        notifier(summary_message, self.webhook_url)
        sleep(1)

        # After placing the orders and stoploss orders
        in_trade = True
        call_sl_hit = False
        put_sl_hit = False

        def price_tracker():

            nonlocal call_price, put_price
            loop_number = 0
            while in_trade:
                underlying_price = price_dict.get(self.token, 0)['ltp']
                call_price = price_dict.get(call_token, 0)['ltp']
                put_price = price_dict.get(put_token, 0)['ltp']
                iv = straddleiv(call_price, put_price, underlying_price, equal_strike, timetoexpiry(expiry))

                if loop_number % 100 == 0:
                    print(f'Index: {self.name}\nTime: {currenttime().time()}\nStrike: {equal_strike}\n' +
                          f'Call Price: {call_price}\nPut Price: {put_price}\n' +
                          f'Total price: {call_price + put_price}\nIV: {iv}\n')
                loop_number += 1

        price_updater = Thread(target=price_tracker)
        price_updater.start()

        if monitor_sl:

            def check_sl_orders(order_ids, side):

                nonlocal orderbook
                orderbook = fetch_book('orderbook')
                statuses = lookup_and_return(orderbook, 'orderid', order_ids, 'status')
                sleep(1)

                if all(statuses == 'trigger pending'):
                    return False, False

                elif all(statuses == 'rejected') or all(statuses == 'cancelled'):
                    rejection_reason = lookup_and_return(orderbook, 'orderid', order_ids, 'text')
                    if all(rejection_reason == '17070 : The Price is out of the LPP range'):
                        notifier(f'{self.name} {side} stoploss orders triggered but rejected ' +
                                 f'due to LPP range.', self.webhook_url)
                    else:
                        notifier(f'{self.name} {side} stoploss orders rejected due to other reason.', self.webhook_url)
                        raise Exception(f'{side} stoploss orders rejected')
                    return True, False

                elif all(statuses == 'pending'):

                    # Confirm pending orders
                    sleep(1)
                    orderbook = fetch_book('orderbook')
                    statuses = lookup_and_return(orderbook, 'orderid', order_ids, 'status')

                    if all(statuses == 'pending'):

                        notifier(f'{self.name} {side} stoploss orders triggered but pending. ' +
                                 f'Cancelling orders and placing market orders', self.webhook_url)
                        for order_id in order_ids:
                            try:
                                obj.cancelOrder(order_id, 'STOPLOSS')
                            except Exception as e:
                                notifier(f'{self.name} {side} stoploss order {order_id} cancellation failed. ' +
                                         f'Error: {e}', self.webhook_url)
                                raise Exception(f'{side} stoploss order {order_id} cancellation failed. Error: {e}')
                        return True, False
                    elif all(statuses == 'complete'):
                        notifier(f'{self.name} {side} stoploss orders triggered and completed.', self.webhook_url)
                        return True, True
                    else:
                        notifier(f'{self.name} {side} stoploss orders pending due to price jump. ' +
                                 f'But order status is not pending or complete. Statuses: {statuses}', self.webhook_url)
                        raise Exception(f'{side} stoploss orders pending due to price jump. ' +
                                        f'But order status is not pending or complete. Statuses: {statuses}')

                elif all(statuses == 'complete'):
                    notifier(f'{self.name} {side} stoploss orders triggered and completed.', self.webhook_url)
                    return True, True

                else:
                    notifier(f'{self.name} {side} stoploss orders in unknown state.', self.webhook_url)
                    raise Exception(f'{side} stoploss orders in unknown state.')

            while currenttime().time() < time(*exit_time) and not call_sl_hit and not put_sl_hit:

                call_sl_hit, call_sl_orders_complete = check_sl_orders(call_stoploss_order_ids, 'call')
                put_sl_hit, put_sl_orders_complete = check_sl_orders(put_stoploss_order_ids, 'put')

                if call_sl_hit:

                    if call_sl_orders_complete:
                        pass
                    else:
                        for quantity in spliced_orders:
                            placeorder(call_symbol, call_token, quantity * self.lot_size,
                                       'BUY', 'MARKET')

                    # Cancelling and placing new put sl orders
                    for orderid in put_stoploss_order_ids:
                        obj.cancelOrder(orderid, 'STOPLOSS')
                    put_stoploss_order_ids = []
                    for quantity in spliced_orders:
                        put_sl_order_id = placeSLorder(put_symbol, put_token, quantity * self.lot_size,
                                                       'BUY', put_avg_price, 'Stoploss put')
                        put_stoploss_order_ids.append(put_sl_order_id)

                    # Monitoring second stoploss
                    while currenttime().time() < time(*exit_time):
                        put_sl_hit, put_sl_orders_complete = check_sl_orders(put_stoploss_order_ids, 'put')
                        if put_sl_hit:
                            if put_sl_orders_complete:
                                pass
                            else:
                                for quantity in spliced_orders:
                                    placeorder(put_symbol, put_token, quantity * self.lot_size,
                                               'BUY', 'MARKET')
                            break
                        else:
                            print(f'{self.name} put stoploss not triggered yet')
                            sleep(5)

                elif put_sl_hit:

                    if put_sl_orders_complete:
                        pass
                    else:
                        for quantity in spliced_orders:
                            placeorder(put_symbol, put_token, quantity * self.lot_size,
                                       'BUY', 'MARKET')

                    # Cancelling and placing new put sl orders
                    for orderid in call_stoploss_order_ids:
                        obj.cancelOrder(orderid, 'STOPLOSS')
                    call_stoploss_order_ids = []
                    for quantity in spliced_orders:
                        call_sl_order_id = placeSLorder(call_symbol, call_token, quantity * self.lot_size,
                                                        'BUY', call_avg_price, 'Stoploss call')
                        call_stoploss_order_ids.append(call_sl_order_id)

                    # Monitoring second stoploss
                    while currenttime().time() < time(*exit_time):
                        call_sl_hit, call_sl_orders_complete = check_sl_orders(call_stoploss_order_ids, 'call')
                        if call_sl_hit:
                            if call_sl_orders_complete:
                                pass
                            else:
                                for quantity in spliced_orders:
                                    placeorder(call_symbol, call_token, quantity * self.lot_size,
                                               'BUY', 'MARKET')
                            break
                        else:
                            print(f'{self.name} call stoploss not triggered yet')
                            sleep(5)
                else:
                    current_time = currenttime().time().strftime('%H:%M:%S')
                    print(f'{current_time} {self.name} stoplosses not triggered')
                    sleep(5)

        else:

            while currenttime().time() < time(*exit_time):
                current_time = currenttime().time().strftime('%H:%M:%S')
                print(f'{current_time} {self.name} - Waiting for exit time')
                sleep(15)

            # Checking which stoplosses orders were triggered
            sleep(1)
            orderbook = fetch_book('orderbook')

            call_sl_statuses = lookup_and_return(orderbook, 'orderid', call_stoploss_order_ids, 'status')
            put_sl_statuses = lookup_and_return(orderbook, 'orderid', put_stoploss_order_ids, 'status')

            call_sl_hit = all(call_sl_statuses == 'complete')
            put_sl_hit = all(put_sl_statuses == 'complete')

        # Cancelling pending stoploss orders
        if call_sl_hit and put_sl_hit:
            pending_sl_orders = False
        elif call_sl_hit:
            pending_sl_orders = put_stoploss_order_ids
        elif put_sl_hit:
            pending_sl_orders = call_stoploss_order_ids
        else:
            pending_sl_orders = call_stoploss_order_ids + put_stoploss_order_ids

        if pending_sl_orders:
            for orderid in pending_sl_orders:
                obj.cancelOrder(orderid, 'STOPLOSS')

        # Exit sequence below
        call_price = fetchltp('NFO', call_symbol, call_token)
        put_price = fetchltp('NFO', put_symbol, put_token)

        if call_sl_hit and put_sl_hit:
            notifier(f'{self.name}: Both stoplosses were triggered.', self.webhook_url)
            put_exit_price = lookup_and_return(orderbook, 'orderid',
                                               put_stoploss_order_ids, 'averageprice').astype(float).mean()
            call_exit_price = lookup_and_return(orderbook, 'orderid',
                                                call_stoploss_order_ids, 'averageprice').astype(float).mean()
            self.points_captured = (put_avg_price - put_exit_price) + (call_avg_price - call_exit_price)
            self.stoploss = 'Both'

        elif call_sl_hit:
            for quantity in spliced_orders:
                placeorder(put_symbol, put_token, quantity * self.lot_size, 'BUY', put_price * 1.1,
                           'Exit order')
                sleep(0.3)
            notifier(f'{self.name}: Exited put. Call stoploss was triggered.', self.webhook_url)
            call_exit_price = lookup_and_return(orderbook, 'orderid',
                                                call_stoploss_order_ids, 'averageprice').astype(float).mean()
            self.points_captured = (call_avg_price - call_exit_price) + (put_avg_price - put_price)
            self.stoploss = 'Call'

        elif put_sl_hit:
            for quantity in spliced_orders:
                placeorder(call_symbol, call_token, quantity * self.lot_size, 'BUY', call_price * 1.1,
                           'Exit order')
                sleep(0.3)
            notifier(f'{self.name}: Exited call. Put stoploss was triggered.', self.webhook_url)
            put_exit_price = lookup_and_return(orderbook, 'orderid',
                                               put_stoploss_order_ids, 'averageprice').astype(float).mean()
            self.points_captured = (put_avg_price - put_exit_price) + (call_avg_price - call_price)
            self.stoploss = 'Put'

        else:
            self.place_straddle_order(equal_strike, expiry, 'BUY', quantity_in_lots)
            notifier(f'{self.name}: Exited positions. No stoploss was triggered.', self.webhook_url)
            self.points_captured = (call_avg_price + put_avg_price) - (call_price + put_price)
            self.stoploss = 'None'

        self.order_list[0]['Points Captured'] = self.points_captured
        self.order_list[0]['Stoploss'] = self.stoploss

        in_trade = False
        sws.close_connection()

    def rollover_short_butterfly(self, quantity_in_lots, ce_hedge_offset=1.02,
                                 pe_hedge_offset=0.98):

        """Shorts a butterfly spread."""

        # Deciding which expirys to trade
        daystocurrentexpiry = timetoexpiry(self.current_expiry) * 365
        if 2 > daystocurrentexpiry > 1:
            atm_expiry_to_buy = self.current_expiry
            otm_expiry_to_sell = self.current_expiry
            atm_expiry_to_sell = self.next_expiry
            otm_expiry_to_buy = self.next_expiry
        elif daystocurrentexpiry < 1:
            atm_expiry_to_buy = self.next_expiry
            otm_expiry_to_sell = self.next_expiry
            atm_expiry_to_sell = self.next_expiry
            otm_expiry_to_buy = self.next_expiry
        else:
            atm_expiry_to_buy = self.current_expiry
            otm_expiry_to_sell = self.current_expiry
            atm_expiry_to_sell = self.current_expiry
            otm_expiry_to_buy = self.current_expiry

        # Setting the new short butterfly spread
        ltp = self.fetch_ltp()
        atm_sell_strike = findstrike(ltp, self.base)
        otm_call_buy_strike = findstrike(ltp * ce_hedge_offset, self.base)
        otm_put_buy_strike = findstrike(ltp * pe_hedge_offset, self.base)

        # Fetching the previous strikes
        with open(f'{self.name}_daily_butterfly.txt', 'r') as file:
            yesterday_strikes = file.readlines()
            yesterday_strikes = [int(strike.rstrip().split(':')[1]) for strike in yesterday_strikes]
            atm_buy_strike = yesterday_strikes[0]
            otm_call_sell_strike = yesterday_strikes[1]
            otm_put_sell_strike = yesterday_strikes[2]

        # Placing orders
        if atm_sell_strike == atm_buy_strike and atm_expiry_to_buy == atm_expiry_to_sell:
            notifier('No trade today.', self.webhook_url)
            atm_call_price, atm_put_price = fetch_straddle_price(self.name, atm_expiry_to_buy, atm_buy_strike)
            otm_call_price, otm_put_price = fetch_strangle_price(self.name, atm_expiry_to_buy, otm_call_sell_strike,
                                                                 otm_put_sell_strike)

            order_log = [

                {'Date': currenttime().strftime('%d-%m-%Y %H:%M:%S'), 'Index': self.name,
                 'Put Strike': atm_buy_strike, 'Call Strike': atm_buy_strike, 'Expiry': atm_expiry_to_buy,
                 'Action Type': 'No trade required', 'Call Price': atm_call_price,
                 'Put Price': atm_put_price, 'Total Price': atm_call_price + atm_put_price},
                {'Date': currenttime().strftime('%d-%m-%Y %H:%M:%S'), 'Index': self.name,
                 'Put Strike': otm_put_sell_strike, 'Call Strike': otm_call_sell_strike, 'Expiry': atm_expiry_to_buy,
                 'Action Type': 'No trade required', 'Call Price': otm_call_price,
                 'Put Price': otm_put_price, 'Total Price': otm_call_price + otm_put_price}
            ]
            self.order_list.extend(order_log)
        else:
            notifier(f'Daily short Butterfly: Buying {atm_buy_strike} and selling {atm_sell_strike}. ' +
                     f'Hedges - Selling {otm_call_sell_strike}CE and {otm_put_sell_strike}PE and '
                     f'buying {otm_call_buy_strike}CE and {otm_put_buy_strike}PE.', self.webhook_url)

            self.place_straddle_order(atm_buy_strike, atm_expiry_to_buy, 'BUY', quantity_in_lots,
                                      order_tag='Daily short Butterfly main')
            self.place_straddle_order(atm_sell_strike, atm_expiry_to_sell, 'SELL', quantity_in_lots,
                                      order_tag='Daily short Butterfly main')
            self.place_strangle_order(otm_call_buy_strike, otm_put_buy_strike, otm_expiry_to_buy, 'BUY',
                                      quantity_in_lots, order_tag='Daily short Butterfly hedges')
            self.place_strangle_order(otm_call_sell_strike, otm_put_sell_strike, otm_expiry_to_sell, 'SELL',
                                      quantity_in_lots, order_tag='Daily short Butterfly hedges')

        # Updating daily_butterfly.txt
        with open(f"{self.name}_daily_butterfly.txt", "w") as file:
            file.write(f'ATM: {atm_sell_strike}\nCall hedge: {otm_call_buy_strike}\nPut hedge: {otm_put_buy_strike}')
        return

    def intraday_butterfly_delta_hedged(self, quantity_in_lots, ce_hedge_offset=1.02,
                                        pe_hedge_offset=0.98, exit_time=(15, 20)):

        """Params: quantity_in_lots, ce_hedge_offset, pe_hedge_offset, exit_time (tuple of hour and minute)"""

        if timetoexpiry(self.current_expiry) * 365 < 1:
            expiry = self.next_expiry
        else:
            expiry = self.current_expiry

        # Setting up the strikes
        ltp = self.fetch_ltp()
        atm_strike = findstrike(ltp, self.base)
        hedge_call_strike = findstrike(ltp * ce_hedge_offset, self.base)
        hedge_put_strike = findstrike(ltp * pe_hedge_offset, self.base)

        # Placing orders
        notifier(f'Intraday Butterfly: Selling {atm_strike} and buying {hedge_call_strike}CE and {hedge_put_strike}PE.',
                 self.webhook_url)
        self.place_straddle_order(atm_strike, expiry, 'SELL', quantity_in_lots,
                                  order_tag='Intraday Butterfly main')
        self.place_strangle_order(hedge_call_strike, hedge_put_strike, expiry, 'BUY', quantity_in_lots,
                                  order_tag='Intraday Butterfly hedges')

        positions = {
            f'{self.name} {atm_strike} {expiry} CE': -1 * quantity_in_lots * self.lot_size,
            f'{self.name} {atm_strike} {expiry} PE': -1 * quantity_in_lots * self.lot_size,
            f'{self.name} {hedge_call_strike} {expiry} CE': quantity_in_lots * self.lot_size,
            f'{self.name} {hedge_put_strike} {expiry} PE': quantity_in_lots * self.lot_size
        }
        positions = pd.DataFrame(positions, index=['quantity']).T

        synthetic_fut_call = f'{self.name} {atm_strike} {expiry} CE'
        synthetic_fut_put = f'{self.name} {atm_strike} {expiry} PE'
        delta_threshold = 1 * self.lot_size
        delta_position_dict = {synthetic_fut_call: 0, synthetic_fut_put: 0}

        while currenttime().time() < time(*exit_time):

            underlying_price = self.fetch_ltp()
            delta_position = pd.DataFrame(delta_position_dict, index=['quantity']).T
            merged_positions = positions.combine(delta_position, lambda x, y: x + y, fill_value=0)
            merged_positions[['ltp', 'iv', 'delta']] = merged_positions.index.map(fetch_price_iv_delta).to_list()
            merged_positions['delta'] = merged_positions.delta * merged_positions.quantity
            current_delta = merged_positions.delta.sum()

            print(f'\n**** Starting Loop ****\n' +
                  f'Delta Position:\n{delta_position}\n' +
                  f'Current position:\n{merged_positions}\nCurrent delta: {current_delta}\n')

            if abs(current_delta) > delta_threshold:

                if current_delta > 0:  # We are long
                    lots_to_sell = round(abs(current_delta) / self.lot_size, 0)
                    notifier(f'Delta greater than {delta_threshold}. Selling {lots_to_sell} ' +
                             f'synthetic futures to reduce delta.\n', self.webhook_url)
                    place_synthetic_fut(self.name, atm_strike, expiry, 'SELL', lots_to_sell * self.lot_size)
                    delta_position_dict[synthetic_fut_call] += -1 * lots_to_sell * self.lot_size
                    delta_position_dict[synthetic_fut_put] += lots_to_sell * self.lot_size

                else:  # We are short
                    lots_to_buy = round(abs(current_delta) / self.lot_size, 0)
                    notifier(f'Delta less than {-delta_threshold}. Buying {lots_to_buy} ' +
                             f'synthetic futures to reduce delta.\n', self.webhook_url)
                    place_synthetic_fut(self.name, atm_strike, expiry, 'BUY', lots_to_buy * self.lot_size)
                    delta_position_dict[synthetic_fut_call] += lots_to_buy * self.lot_size
                    delta_position_dict[synthetic_fut_put] += -1 * lots_to_buy * self.lot_size

            sleep(2)

        # Closing the main positions along with the delta positions if any are open
        notifier(f'Intraday Butterfly: Closing positions.', self.webhook_url)
        self.place_straddle_order(atm_strike, expiry, 'BUY', quantity_in_lots,
                                  order_tag='Intraday Butterfly main')
        self.place_strangle_order(hedge_call_strike, hedge_put_strike, expiry, 'SELL', quantity_in_lots,
                                  order_tag='Intraday Butterfly hedges')

        # Squaring off the delta positions
        if delta_position_dict[synthetic_fut_call] != 0 and delta_position_dict[synthetic_fut_put] != 0:
            assert delta_position_dict[synthetic_fut_call] == -1 * delta_position_dict[synthetic_fut_put]
            quantity_to_square_up = abs(delta_position_dict[synthetic_fut_call])

            if delta_position_dict[synthetic_fut_call] > 0:
                action = 'BUY'
            else:
                action = 'SELL'

            place_synthetic_fut(self.name, atm_strike, expiry, action, quantity_to_square_up)
            notifier(f'Intraday Butterfly: Squared off delta positions. ' +
                     f'{action} {quantity_to_square_up} synthetic futures.', self.webhook_url)
        elif delta_position_dict[synthetic_fut_call] == 0 and delta_position_dict[synthetic_fut_put] == 0:
            notifier('No delta positions to square off.', self.webhook_url)
        else:
            raise AssertionError('Delta positions are not balanced.')
