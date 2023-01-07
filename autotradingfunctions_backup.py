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


def login(user, pin, apikey, authkey, webhook_url=None):
    global obj
    authkey = pyotp.TOTP(authkey)
    obj = SmartConnect(api_key=apikey)
    data = obj.generateSession(user, pin, authkey.now())
    if data['message'] != 'SUCCESS':
        for attempt in range(2, 7):
            notifier(f'Login attempt {attempt}.', webhook_url)
            data = obj.generateSession(user, pin, authkey.now())
            if data['message'] == 'SUCCESS':
                break
            if attempt == 6:
                notifier('Login failed.', webhook_url)
                raise Exception('Login failed.')
            sleep(30)
    notifier(f'Date: {currenttime().strftime("%d %b %Y %H:%M:%S")}\nLogged in successfully.', webhook_url)


# Ticker file
def get_ticker_file():
    global scrips
    url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    data = urllib.request.urlopen(url).read().decode()
    scrips = pd.read_json(data)
    return scrips


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
                    print('Failed to fetch orderbook. Retrying in 1 second.')
                    sleep(1.2)

    elif book == 'positions' or book == 'position':
        for attempt in range(1, 7):
            try:
                data = obj.position()['data']
                return data
            except:
                if attempt == 6:
                    raise Exception('Failed to fetch positions.')
                else:
                    print('Failed to fetch positions. Retrying in 1 second.')
                    sleep(1.2)


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
    if time(9, 15) < currenttime().time() < time(15, 30):
        return True
    else:
        return False


# Defining current time
def currenttime():
    return datetime.now() + timedelta(minutes = 330)


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


# ORDER FUNCTIONS BELOW #

def placeorder(symbol, token, qty, buyorsell, orderprice, ordertag=""):
    """Provide symbol, token, qty (shares), buyorsell, orderprice, ordertag (optional)"""

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

    def __init__(self, name, webhook_url=None):

        self.name = name
        self.symbol, self.token = fetch_symbol_token(name)
        self.lot_size = fetch_lot_size(name)
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
        self.fetch_expirys()

        if self.name == 'BANKNIFTY':
            self.base = 100
        elif self.name == 'NIFTY':
            self.base = 50
        elif self.name == 'FINNIFTY':
            self.base = 50
        else:
            raise ValueError('Index name not valid')

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

    def place_straddle_order(self, strike, expiry, buy_or_sell, quantity_in_lots,
                             multiple_of_orders=1, return_avg_price=False, order_tag=""):

        """Places a straddle order on the index on the given strike, expiry, buy or sell,
        quantity in lots and multiple of orders. Provide the expiry in DDMMMYY format"""

        call_symbol, call_token = fetch_symbol_token(f'{self.name} {strike} {expiry} CE')
        put_symbol, put_token = fetch_symbol_token(f'{self.name} {strike} {expiry} PE')
        call_price = fetchltp('NFO', call_symbol, call_token)
        put_price = fetchltp('NFO', put_symbol, put_token)

        limit_price_extender = 1.1 if buy_or_sell == 'BUY' else 0.9

        call_order_id_list = []
        put_order_id_list = []
        for order in range(1, multiple_of_orders + 1):
            call_order_id = placeorder(call_symbol, call_token,
                                       quantity_in_lots * self.lot_size,
                                       buy_or_sell, call_price * limit_price_extender,
                                       ordertag=order_tag)
            put_order_id = placeorder(put_symbol, put_token,
                                      quantity_in_lots * self.lot_size,
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
                     f'{strike} {expiry} {quantity_in_lots * multiple_of_orders} lot(s).', self.webhook_url)
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
                     f'{strike} {expiry} {quantity_in_lots * multiple_of_orders} lot(s).', self.webhook_url)
            raise Exception('Orders rejected')

        else:
            notifier(f'{order_tag}: ERROR. Order statuses uncertain for {buy_or_sell} {self.name} ' +
                     f'{strike} {expiry} {quantity_in_lots * multiple_of_orders} lot(s).', self.webhook_url)
            raise Exception('Order statuses uncertain')

    def place_strangle_order(self, call_strike, put_strike, expiry, buy_or_sell, quantity_in_lots,
                             multiple_of_orders=1, return_avg_price=False, order_tag=""):

        """Places a strangle order on the index on the given call and put strike, expiry, buy or sell,
        quantity in lots and multiple of orders. Provide the expiry in DDMMMYY format"""

        call_symbol, call_token = fetch_symbol_token(f'{self.name} {call_strike} {expiry} CE')
        put_symbol, put_token = fetch_symbol_token(f'{self.name} {put_strike} {expiry} PE')
        call_price = fetchltp('NFO', call_symbol, call_token)
        put_price = fetchltp('NFO', put_symbol, put_token)

        limit_price_extender = 1.1 if buy_or_sell == 'BUY' else 0.9

        call_order_id_list = []
        put_order_id_list = []
        for order in range(1, multiple_of_orders + 1):
            call_order_id = placeorder(call_symbol, call_token,
                                       quantity_in_lots * self.lot_size,
                                       buy_or_sell, call_price * limit_price_extender,
                                       ordertag=order_tag)
            put_order_id = placeorder(put_symbol, put_token,
                                      quantity_in_lots * self.lot_size,
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
                     f'{expiry} {quantity_in_lots * multiple_of_orders} lot(s).', self.webhook_url)
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
                     f'{expiry} {quantity_in_lots * multiple_of_orders} lot(s).', self.webhook_url)
            raise Exception('Orders rejected')

        else:
            notifier(f'{order_tag}: ERROR. Order statuses uncertain for {buy_or_sell} {self.name} ' +
                     f'{call_strike} CE and {put_strike} PE ' +
                     f'{expiry} {quantity_in_lots * multiple_of_orders} lot(s).', self.webhook_url)
            raise Exception('Order statuses uncertain')

    def rollover_overnight_short_straddle(self, quantity_in_lots, multiple_of_orders, strike_offset=1):

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
        with open("daily_sold_strike.txt", "r") as file:
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
                                      multiple_of_orders, order_tag='Daily overnight short straddle')
            self.place_straddle_order(sell_strike, expirytosell, 'SELL', quantity_in_lots,
                                      multiple_of_orders, order_tag='Daily overnight short straddle')

        # Updating daily_sold_strike.txt
        with open("daily_sold_strike.txt", "w") as file:
            file.write(str(sell_strike))
        return

    def buy_weekly_hedge(self, quantity_in_lots, multiple_of_orders, type_of_hedge='strangle', **kwargs):

        ltp = self.fetch_ltp()
        if type_of_hedge == 'strangle':
            call_strike = findstrike(ltp * kwargs['call_offset'], self.base)
            put_strike = findstrike(ltp * kwargs['put_offset'], self.base)
            self.place_strangle_order(call_strike, put_strike, self.next_expiry,
                                      'BUY', quantity_in_lots, multiple_of_orders,
                                      order_tag='Weekly hedge')
        elif type_of_hedge == 'straddle':
            strike = findstrike(ltp * kwargs['strike_offset'], self.base)
            self.place_straddle_order(strike, self.next_expiry,
                                      'BUY', quantity_in_lots, multiple_of_orders,
                                      order_tag='Weekly hedge')

    def intraday_straddle(self, quantity_in_lots, multiple_of_orders, exit_time=(15, 28)):

        def fetch_disparity_dict():
            ltp = self.fetch_ltp()
            current_strike = findstrike(ltp, self.base)
            next_strike = current_strike + self.base
            previous_strike = current_strike - self.base
            disparity_dict = {}
            for strike in [current_strike, next_strike, previous_strike]:
                call_symbol, call_token = fetch_symbol_token(f'{self.name} {strike} {self.current_expiry} CE')
                put_symbol, put_token = fetch_symbol_token(f'{self.name} {strike} {self.current_expiry} PE')
                call_price = fetchltp('NFO', call_symbol, call_token)
                put_price = fetchltp('NFO', put_symbol, put_token)
                disparity = abs(call_price - put_price) / min(call_price, put_price) * 100
                disparity_dict[strike] = disparity, call_symbol, call_token, put_symbol, put_token
            return disparity_dict

        disparities = fetch_disparity_dict()
        equal_strike = min(disparities, key=disparities.get)
        expiry = self.current_expiry
        notifier(f'Initiating intraday trade on {equal_strike} strike.', self.webhook_url)
        call_avg_price, put_avg_price = self.place_straddle_order(equal_strike, expiry, 'SELL', quantity_in_lots,
                                                                  multiple_of_orders, return_avg_price=True,
                                                                  order_tag='Intraday straddle')

        if timetoexpiry(self.current_expiry) * 365 < 1 or self.name == 'BANKNIFTY':
            sl = 1.7
        else:
            sl = 1.5

        call_symbol, call_token, put_symbol, put_token = disparities[equal_strike][1:5]

        call_stoploss_order_ids = []
        put_stoploss_order_ids = []
        for order in range(multiple_of_orders):
            call_sl_order_id = placeSLorder(call_symbol, call_token, quantity_in_lots * self.lot_size,
                                            'BUY', call_avg_price * sl, 'Stoploss call')
            put_sl_order_id = placeSLorder(put_symbol, put_token, quantity_in_lots * self.lot_size,
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
        while currenttime().time() < time(*exit_time):
            current_time = currenttime().time().strftime('%H:%M:%S')
            print(f'{current_time} - Waiting for exit time')
            sleep(15)

        # Checking which stoplosses orders were triggered
        orderbook = fetch_book('orderbook')

        call_sl_statuses = lookup_and_return(orderbook, 'orderid', call_stoploss_order_ids, 'status')
        put_sl_statuses = lookup_and_return(orderbook, 'orderid', put_stoploss_order_ids, 'status')

        call_sl_hit = all(call_sl_statuses == 'complete')
        put_sl_hit = all(put_sl_statuses == 'complete')

        # Cancelling pending stoploss orders
        if call_sl_hit:
            pending_sl_orders = put_stoploss_order_ids
        elif put_sl_hit:
            pending_sl_orders = call_stoploss_order_ids
        else:
            pending_sl_orders = call_stoploss_order_ids + put_stoploss_order_ids

        for orderid in pending_sl_orders:
            obj.cancelOrder(orderid, 'STOPLOSS')

        # Squaring up open positions if any
        call_ltp = fetchltp('NFO', call_symbol, call_token)
        put_ltp = fetchltp('NFO', put_symbol, put_token)

        if call_sl_hit and put_sl_hit:
            notifier(f'{self.name}: Both stoplosses were triggered.', self.webhook_url)
            put_exit_price = lookup_and_return(orderbook, 'orderid',
                                               put_stoploss_order_ids, 'averageprice').astype(float).mean()
            call_exit_price = lookup_and_return(orderbook, 'orderid',
                                                call_stoploss_order_ids, 'averageprice').astype(float).mean()
            self.points_captured = (put_avg_price - put_exit_price) + (call_avg_price - call_exit_price)
            self.stoploss = 'Both'

        elif call_sl_hit:
            for order in range(multiple_of_orders):
                placeorder(put_symbol, put_token, quantity_in_lots * self.lot_size, 'BUY', put_ltp * 1.1, 'Exit order')
                sleep(0.3)
            notifier(f'{self.name}: Exited put. Call stoploss was triggered.', self.webhook_url)
            call_exit_price = lookup_and_return(orderbook, 'orderid',
                                                call_stoploss_order_ids, 'averageprice').astype(float).mean()
            self.points_captured = (call_avg_price - call_exit_price) + (put_avg_price - put_ltp)
            self.stoploss = 'Call'

        elif put_sl_hit:
            for order in range(multiple_of_orders):
                placeorder(call_symbol, call_token, quantity_in_lots * self.lot_size, 'BUY', call_ltp * 1.1,
                           'Exit order')
                sleep(0.3)
            notifier(f'{self.name}: Exited call. Put stoploss was triggered.', self.webhook_url)
            put_exit_price = lookup_and_return(orderbook, 'orderid',
                                               put_stoploss_order_ids, 'averageprice').astype(float).mean()
            self.points_captured = (put_avg_price - put_exit_price) + (call_avg_price - call_ltp)
            self.stoploss = 'Put'

        else:
            self.place_straddle_order(equal_strike, expiry, 'BUY', quantity_in_lots, multiple_of_orders)
            notifier(f'{self.name}: Exited positions. No stoploss was triggered.', self.webhook_url)
            self.points_captured = (call_avg_price + put_avg_price) - (call_ltp + put_ltp)
            self.stoploss = 'None'

    def rollover_short_butterfly(self, quantity_in_lots, multiple_of_orders, ce_hedge_offset=1.02,
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
        with open('daily_butterfly.txt', 'r') as file:
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
                                      multiple_of_orders, order_tag='Daily short Butterfly main')
            self.place_straddle_order(atm_sell_strike, atm_expiry_to_sell, 'SELL', quantity_in_lots,
                                      multiple_of_orders, order_tag='Daily short Butterfly main')
            self.place_strangle_order(otm_call_buy_strike, otm_put_buy_strike, otm_expiry_to_buy, 'BUY',
                                      quantity_in_lots, multiple_of_orders, order_tag='Daily short Butterfly hedges')
            self.place_strangle_order(otm_call_sell_strike, otm_put_sell_strike, otm_expiry_to_sell, 'SELL',
                                      quantity_in_lots, multiple_of_orders, order_tag='Daily short Butterfly hedges')

        # Updating daily_butterfly.txt
        with open("daily_butterfly.txt", "w") as file:
            file.write(f'ATM: {atm_sell_strike}\nCall hedge: {otm_call_buy_strike}\nPut hedge: {otm_put_buy_strike}')
        return
