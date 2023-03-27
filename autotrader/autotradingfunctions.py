import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta, timezone
from time import sleep
import requests
import json
from smartapi import SmartConnect
from smartapi.smartExceptions import DataException
import pyotp
from threading import Thread
from autotrader.SmartWebSocketV2 import SmartWebSocketV2
from autotrader import scrips, holidays, blackscholes as bs, nsefunctions as nse
from collections import defaultdict

global login_data, obj, sws, price_dict


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
        price_dict[message['token']] = {'ltp': message['last_traded_price'] / 100,
                                        'timestamp': datetime.fromtimestamp(
                                            message['exchange_timestamp'] / 1000).strftime('%H:%M:%S')}

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


def fetch_price_dict():
    global price_dict
    new_price_dict = {scrips.loc[scrips.token == token]['symbol'].values[0]: value for token, value in
                      price_dict.items()}
    return new_price_dict


def fetch_book(book):
    if book == 'orderbook':
        for attempt in range(1, 7):
            try:
                data = obj.orderBook()['data']
                return data
            except DataException:
                if attempt == 6:
                    raise Exception('Failed to fetch orderbook.')
                else:
                    sleep(2)
                    continue
            except Exception as e:
                if attempt == 6:
                    raise Exception('Failed to fetch orderbook.')
                else:
                    print(f'Error {attempt} in fetching orderbook: {e}')
                    sleep(2)
                    continue

    elif book == 'positions' or book == 'position':
        for attempt in range(1, 7):
            try:
                data = obj.position()['data']
                return data
            except DataException:
                if attempt == 6:
                    raise Exception('Failed to fetch positions.')
                else:
                    sleep(2)
                    continue
            except Exception as e:
                if attempt == 6:
                    raise Exception('Failed to fetch positions.')
                else:
                    print(f'Error {attempt} in fetching positions: {e}')
                    sleep(2)
                    continue


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

            if book == 'orderbook':
                if isinstance(valuetolookup, list):

                    bucket = [order[fieldtoreturn] for order in fetch_book('orderbook')
                              if order[fieldtolookup] in valuetolookup and order[fieldtolookup] != '']
                elif isinstance(valuetolookup, str):
                    bucket = [order[fieldtoreturn] for order in fetch_book('orderbook')
                              if order[fieldtolookup] == valuetolookup and order[fieldtolookup] != '']
                else:
                    raise ValueError('Invalid valuetolookup')

            elif book == 'positions':

                if isinstance(valuetolookup, list):
                    bucket = [order[fieldtoreturn] for order in fetch_book('positions')
                              if order[fieldtolookup] in valuetolookup and order[fieldtolookup] != '']
                elif isinstance(valuetolookup, str):
                    bucket = [order[fieldtoreturn] for order in fetch_book('positions')
                              if order[fieldtolookup] == valuetolookup and order[fieldtolookup] != '']
                else:
                    raise ValueError('Invalid valuetolookup')
            else:
                raise ValueError('Invalid dictionary')

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
    # Adjusting for timezones
    ist = timezone(timedelta(hours=5, minutes=30))
    return datetime.now(ist)


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
    for attempt in range(1, 6):
        try:
            price = obj.ltpData(exchange_seg, symbol, token)['data']['ltp']
            return price
        except DataException:
            if attempt == 5:
                raise DataException('Failed to fetch LTP')
            else:
                sleep(1)
                continue
        except Exception as e:
            if attempt == 5:
                raise e
            else:
                print(f'Error {attempt} in fetching LTP: {e}')
                sleep(1)
                continue


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


def timetoexpiry(expiry, effective_time=False, in_days=False):
    """Return time left to expiry"""
    if in_days:
        multiplier = 365
    else:
        multiplier = 1

    expiry = datetime.strptime(expiry, '%d%b%y')
    time_to_expiry = ((expiry + pd.DateOffset(minutes=930)) - currenttime()) / timedelta(days=365)

    # Subtracting holidays and weekends
    if effective_time:
        date_range = pd.date_range(currenttime().date(), expiry - timedelta(days=1))
        numer_of_weekdays = sum(date_range.dayofweek > 4)
        number_of_holidays = sum(date_range.isin(holidays))
        time_to_expiry -= (numer_of_weekdays + number_of_holidays) / 365
        # print(f'Number of weekdays: {numer_of_weekdays} and number of holidays: {number_of_holidays}')
    return time_to_expiry * multiplier


def straddleiv(callprice, putprice, spot, strike, timeleft):
    call_iv = bs.implied_volatility(callprice, spot, strike, timeleft, 0.05, 'c')
    put_iv = bs.implied_volatility(putprice, spot, strike, timeleft, 0.05, 'p')
    avg_iv = (call_iv + put_iv) / 2 * 100

    return round(avg_iv, 2)


def calc_greeks(position_string, position_price, underlying_price):
    """Fetches the price, iv and delta of a stock"""

    name, strike, expiry, option_type = position_string.split()
    strike = int(strike)
    time_left = timetoexpiry(expiry)

    iv = bs.implied_volatility(position_price, underlying_price, strike, time_left, 0.05, option_type) * 100
    delta = bs.delta(underlying_price, strike, time_left, 0.05, iv, option_type)
    gamma = bs.gamma(underlying_price, strike, time_left, 0.05, iv)

    return iv, delta, gamma


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

    for attempt in range(1, 4):

        try:
            order_id = obj.placeOrder(params)
            return order_id
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            if attempt == 3:
                raise e
            print(f'Error {attempt} in placing order for {symbol}: {e}')
            sleep(2)
            continue


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

    for attempt in range(1, 4):

        try:
            order_id = obj.placeOrder(params)
            return order_id
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            if attempt == 3:
                raise e
            print(f'Error {attempt} in placing SL order for {symbol}: {e}')
            sleep(2)
            continue


def place_synthetic_fut_order(name, strike, expiry, buy_or_sell, quantity, price='MARKET'):
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


class SharedData:
    def __init__(self):
        self.position_data = None
        self.orderbook_data = None
        self.updated_time = None
        self.error_info = None

    def fetch_data(self):
        try:
            self.position_data = fetch_book('position')
            self.orderbook_data = fetch_book('orderbook')
            self.updated_time = currenttime()
        except Exception as e:
            self.position_data = None
            self.orderbook_data = None
            self.error_info = e

    def update_data(self, sleep_time=5, exit_time=(15, 30)):
        while currenttime().time() < time(*exit_time):
            self.fetch_data()
            sleep(sleep_time)


class Index:
    """Initialize an index with the name of the index in uppercase"""

    def __init__(self, name, webhook_url=None, subscribe_to_ws=False):

        self.name = name
        self.ltp = None
        self.previous_close = None
        self.current_expiry = None
        self.next_expiry = None
        self.month_expiry = None
        self.fut_expiry = None
        self.order_log = defaultdict(list)
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
                sleep(2)
                print(f'{self.name}: Subscribed underlying to the websocket')
            except NameError:
                print('Websocket not initialized. Please initialize the websocket before subscribing to it.')

    def fetch_freeze_limit(self):
        try:
            freeze_qty = nse.nse_fno(self.name)['vfq']
            freeze_qty_in_lots = int(freeze_qty / self.lot_size)
            assert freeze_qty_in_lots > 0
            print('Freeze qty in lots: ', freeze_qty_in_lots, ' for ', self.name, ' fetched from nse api')
            return freeze_qty_in_lots
        except Exception as e:
            print(e)
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

    def log_order(self, strike, expiry, buy_or_sell, call_price, put_price, order_tag):
        dict_format = {'Date': currenttime().strftime('%d-%m-%Y %H:%M:%S'), 'Index': self.name,
                       'Put Strike': strike, 'Call Strike': strike, 'Expiry': expiry,
                       'Action Type': buy_or_sell, 'Call Price': call_price,
                       'Put Price': put_price, 'Total Price': call_price + put_price,
                       'Tag': order_tag}
        self.order_log[order_tag].append(dict_format)

    def splice_orders(self, quantity_in_lots):

        if quantity_in_lots > self.freeze_qty:
            loops = int(quantity_in_lots / self.freeze_qty)
            if loops > 10:
                raise Exception('Order too big. This error was raised to prevent accidental large order placement.')

            remainder = quantity_in_lots % self.freeze_qty
            if remainder == 0:
                spliced_orders = [self.freeze_qty] * loops
            else:
                spliced_orders = [self.freeze_qty] * loops + [remainder]
        else:
            spliced_orders = [quantity_in_lots]
        return spliced_orders

    def place_combined_order(self, expiry, buy_or_sell, quantity_in_lots, strike=None, call_strike=None,
                             put_strike=None, return_avg_price=False, order_tag=""):

        """
        Places a straddle or strangle order on the index.
        Params:
        strike: Strike price of the option (for straddle)
        expiry: Expiry of the option
        buy_or_sell: BUY or SELL
        quantity_in_lots: Quantity in lots
        call_strike: Strike price of the call (for strangle)
        put_strike: Strike price of the put (for strangle)
        return_avg_price: If True, returns the average price of the order
        order_tag: Tag to be added to the order
        """

        if strike is None:
            if call_strike is None and put_strike is None:
                raise ValueError('Strike price not specified')
            strike_info = f'{call_strike}CE {put_strike}PE'
        elif call_strike is None and put_strike is None:
            call_strike = strike
            put_strike = strike
            strike_info = f'{strike}'
        else:
            raise ValueError('Strike price specified twice')

        call_symbol, call_token = fetch_symbol_token(f'{self.name} {call_strike} {expiry} CE')
        put_symbol, put_token = fetch_symbol_token(f'{self.name} {put_strike} {expiry} PE')
        call_price = fetchltp('NFO', call_symbol, call_token)
        put_price = fetchltp('NFO', put_symbol, put_token)

        limit_price_extender = 1.1 if buy_or_sell == 'BUY' else 0.9

        spliced_orders = self.splice_orders(quantity_in_lots)

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
                     f'{strike_info} {expiry} {quantity_in_lots} lot(s).', self.webhook_url)
            call_order_avg_price = lookup_and_return(orderbook, 'orderid',
                                                     call_order_id_list, 'averageprice').astype(float).mean()
            put_order_avg_price = lookup_and_return(orderbook, 'orderid',
                                                    put_order_id_list, 'averageprice').astype(float).mean()
            if return_avg_price:
                return call_order_avg_price, put_order_avg_price
            else:
                return
        elif all(call_order_statuses == 'rejected') and all(put_order_statuses == 'rejected'):
            notifier(f'{order_tag}: All orders rejected for {buy_or_sell} {self.name} ' +
                     f'{strike_info} {expiry} {quantity_in_lots} lot(s).', self.webhook_url)
            raise Exception('Orders rejected')
        else:
            notifier(f'{order_tag}: ERROR. Order statuses uncertain for {buy_or_sell} {self.name} ' +
                     f'{strike_info} {expiry} {quantity_in_lots} lot(s).', self.webhook_url)
            raise Exception('Order statuses uncertain')

    def place_synthetic_fut_order(self, strike, expiry, buy_or_sell, quantity, price='MARKET'):

        """Places a synthetic future order. Quantity is in number of shares."""

        freeze_qty_in_shares = self.freeze_qty * self.lot_size
        if quantity > freeze_qty_in_shares:
            quotient, remainder = divmod(quantity, freeze_qty_in_shares)
            if quotient > 10:
                raise Exception('Order too big. This error was raised to prevent accidental large order placement.')
            if remainder == 0:
                spliced_orders = [self.freeze_qty] * quotient
            else:
                spliced_orders = [self.freeze_qty] * quotient + [remainder]
        else:
            spliced_orders = [quantity]

        call_symbol, call_token = fetch_symbol_token(f'{self.name} {strike} {expiry} CE')
        put_symbol, put_token = fetch_symbol_token(f'{self.name} {strike} {expiry} PE')

        call_order_id_list = []
        put_order_id_list = []
        for quantity in spliced_orders:
            if buy_or_sell == 'BUY':
                order_id_call = placeorder(call_symbol, call_token, quantity, 'BUY', 'MARKET')
                order_id_put = placeorder(put_symbol, put_token, quantity, 'SELL', 'MARKET')
            elif buy_or_sell == 'SELL':
                order_id_call = placeorder(call_symbol, call_token, quantity, 'SELL', 'MARKET')
                order_id_put = placeorder(put_symbol, put_token, quantity, 'BUY', 'MARKET')
            else:
                raise Exception('Invalid buy or sell')
            call_order_id_list.append(order_id_call)
            put_order_id_list.append(order_id_put)

        order_statuses = lookup_and_return('orderbook', 'orderid', call_order_id_list + put_order_id_list, 'status')

        if not all(order_statuses == 'complete'):
            raise Exception('Syntehtic Futs: Orders not completed')
        else:
            print(f'Synthetic Futs: {buy_or_sell} Order for {quantity} quantity completed.')

    def find_equal_strike(self, exit_time, websocket, wait_for_equality, **kwargs):

        """Finds the strike price that is equal to the current index price at the time of exit.

        Parameters:
        exit_time (str): Cutoff time
        websocket (bool): If True, the function will use websocket to find the strike price.
        wait_for_equality (bool): If True, the function will wait for the prices to be equal.
        kwargs (dict): Additional keyword arguments to be passed to the websocket function. Includes: 'target_disparity'.

        Returns:
        strike, call symbol, put symbol , call token, put token, call ltp, put ltp"""

        if websocket:

            ltp = price_dict.get(self.token, 0)['ltp']
            current_strike = findstrike(ltp, self.base)
            strike_range = np.arange(current_strike - self.base * 2, current_strike + self.base * 2, self.base)

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
            disparities = np.abs(call_ltps - put_ltps) / np.minimum(call_ltps, put_ltps) * 100

            # If wait_for_equality is True, waits for call and put prices to be equal before selecting a strike
            if wait_for_equality:

                loop_number = 1
                while np.min(disparities) > kwargs['target_disparity']:
                    call_ltps = np.array(
                        [price_dict.get(call_token, {'ltp': 0})['ltp'] for call_token in call_token_list])
                    put_ltps = np.array([price_dict.get(put_token, {'ltp': 0})['ltp'] for put_token in put_token_list])
                    disparities = np.abs(call_ltps - put_ltps) / np.minimum(call_ltps, put_ltps) * 100
                    if loop_number % 200000 == 0:
                        print(f'Time: {currenttime().strftime("%H:%M:%S")}\n' +
                              f'Index: {self.name}\n' +
                              f'Current lowest disparity: {np.min(disparities):.2f}\n' +
                              f'Strike: {strike_list[np.argmin(disparities)]}\n')
                    if (currenttime() + timedelta(minutes=5)).time() > time(*exit_time):
                        notifier('Equal strike tracker exited due to time limit.', self.webhook_url)
                        raise Exception('Equal strike tracker exited due to time limit.')
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
            return strike_to_trade, call_symbol, call_token, put_symbol, put_token, call_ltp, put_ltp

        else:

            ltp = self.fetch_ltp()
            current_strike = findstrike(ltp, self.base)
            if self.name == 'FINNIFTY':
                strike_range = np.arange(current_strike - self.base * 2, current_strike + self.base * 2, self.base)
            else:
                strike_range = np.arange(current_strike - self.base, current_strike + self.base * 2, self.base)
            disparity_dict = {}

            def update_disparity_dict(stk, mydict):
                c_symbol, c_token = fetch_symbol_token(f'{self.name} {stk} {self.current_expiry} CE')
                p_symbol, p_token = fetch_symbol_token(f'{self.name} {stk} {self.current_expiry} PE')
                c_price = fetchltp('NFO', c_symbol, c_token)
                p_price = fetchltp('NFO', p_symbol, p_token)
                disparity = abs(c_price - p_price) / min(c_price, p_price) * 100
                mydict[stk] = disparity, c_symbol, c_token, \
                    p_symbol, p_token, c_price, p_price

            for strike in strike_range:
                update_disparity_dict(strike, disparity_dict)

            if wait_for_equality:

                while min(disparity_dict.values())[0] > kwargs['target_disparity']:
                    for strike in strike_range:
                        update_disparity_dict(strike, disparity_dict)
                    if (currenttime() + timedelta(minutes=5)).time() > time(*exit_time):
                        notifier('Equal strike tracker exited due to time limit.', self.webhook_url)
                        raise Exception('Equal strike tracker exited due to time limit.')

            strike_to_trade = min(disparity_dict, key=disparity_dict.get)
            call_symbol, call_token, put_symbol, put_token, call_ltp, put_ltp = disparity_dict[strike_to_trade][1:]
            return strike_to_trade, call_symbol, call_token, put_symbol, put_token, call_ltp, put_ltp

    def rollover_overnight_short_straddle(self, quantity_in_lots, strike_offset=1, iv_threshold=0.8):

        """ Rollover overnight short straddle to the next expiry.
        Args:
            quantity_in_lots (int): Quantity of the straddle in lots.
            strike_offset (float): Strike offset from the current strike.
            iv_threshold (float): IV threshold compared to vix.
        """

        def load_data():
            try:
                with open('positions.json', 'r') as f:
                    data = json.load(f)
                    return data
            except FileNotFoundError:
                data = {'NIFTY': None, 'BANKNIFTY': None}
                notifier('No positions found for OSS. Creating new file.', self.webhook_url)
                with open('positions.json', 'w') as f:
                    json.dump(data, f)
                return data
            except Exception as e:
                notifier(f'Error while reading positions.json: {e}', self.webhook_url)
                raise Exception('Error while reading positions.json')

        def save_data(data):
            with open('positions.json', 'w') as f:
                json.dump(data, f)

        order_tag = 'Overnight Short Straddle'

        if timetoexpiry(self.current_expiry, effective_time=True, in_days=True) > 4:  # far from expiry
            ltp = self.fetch_ltp()
            sell_strike = findstrike(ltp * strike_offset, self.base)
            call_ltp, put_ltp = fetch_straddle_price(self.name, self.current_expiry, sell_strike)
            iv = straddleiv(call_ltp, put_ltp, ltp, sell_strike, timetoexpiry(self.current_expiry))
            vix = nse.indiavix()
            if iv < vix * iv_threshold:
                notifier(f'IV is too low compared to VIX: IV {iv}, Vix {vix}.', self.webhook_url)
                return
            else:
                notifier(f'IV is fine compared to VIX: IV {iv}, Vix {vix}.', self.webhook_url)
        elif timetoexpiry(self.current_expiry, effective_time=True, in_days=True) < 2:  # only exit
            sell_strike = None
        else:
            ltp = self.fetch_ltp()
            sell_strike = findstrike(ltp * strike_offset, self.base)

        trade_data = load_data()
        buy_strike = trade_data[self.name]

        if not isinstance(buy_strike, int) and not isinstance(buy_strike, float) and buy_strike is not None:
            notifier(f'Invalid strike found for {self.name}.', self.webhook_url)
            raise Exception(f'Invalid strike found for {self.name}.')

        # Placing orders
        if buy_strike is None and sell_strike is None:
            notifier('No trade required.', self.webhook_url)
            return
        elif sell_strike is None:  # only exiting current position
            notifier(f'Exiting current position on strike {buy_strike}.', self.webhook_url)
            call_buy_avg, put_buy_avg = self.place_combined_order(self.current_expiry, 'BUY', quantity_in_lots,
                                                                  strike=buy_strike, return_avg_price=True,
                                                                  order_tag=order_tag)
            self.log_order(buy_strike, self.current_expiry, 'BUY', call_buy_avg, put_buy_avg, order_tag)
        elif buy_strike is None:  # only entering new position
            notifier(f'Entering new position on strike {sell_strike}.', self.webhook_url)
            call_sell_avg, put_sell_avg = self.place_combined_order(self.current_expiry, 'SELL', quantity_in_lots,
                                                                    strike=sell_strike, return_avg_price=True,
                                                                    order_tag=order_tag)
            self.log_order(sell_strike, self.current_expiry, 'SELL', call_sell_avg, put_sell_avg, order_tag)
        else:  # both entering and exiting positions
            if buy_strike == sell_strike:
                notifier('No trade required as strike is same.', self.webhook_url)
                call_ltp, put_ltp = fetch_straddle_price(self.name, self.current_expiry, sell_strike)
                self.log_order(buy_strike, self.current_expiry, 'BUY', call_ltp, put_ltp, order_tag)
                self.log_order(sell_strike, self.current_expiry, 'SELL', call_ltp, put_ltp, order_tag)
            else:
                notifier(f'Buying {buy_strike} and selling {sell_strike}.', self.webhook_url)
                call_buy_avg, put_buy_avg = self.place_combined_order(self.current_expiry, 'BUY', quantity_in_lots,
                                                                      strike=buy_strike, return_avg_price=True,
                                                                      order_tag=order_tag)
                call_sell_avg, put_sell_avg = self.place_combined_order(self.current_expiry, 'SELL', quantity_in_lots,
                                                                        strike=sell_strike, return_avg_price=True,
                                                                        order_tag=order_tag)
                self.log_order(buy_strike, self.current_expiry, 'BUY', call_buy_avg, put_buy_avg, order_tag)
                self.log_order(sell_strike, self.current_expiry, 'SELL', call_sell_avg, put_sell_avg, order_tag)

        trade_data[self.name] = sell_strike
        save_data(trade_data)

    def buy_weekly_hedge(self, quantity_in_lots, type_of_hedge='strangle', **kwargs):

        ltp = self.fetch_ltp()
        if type_of_hedge == 'strangle':
            call_strike = findstrike(ltp * kwargs['call_offset'], self.base)
            put_strike = findstrike(ltp * kwargs['put_offset'], self.base)
            self.place_combined_order(self.next_expiry, 'BUY', quantity_in_lots, call_strike=call_strike,
                                      put_strike=put_strike, order_tag='Weekly hedge')
        elif type_of_hedge == 'straddle':
            strike = findstrike(ltp * kwargs['strike_offset'], self.base)
            self.place_combined_order(self.next_expiry, 'BUY', quantity_in_lots, strike=strike,
                                      order_tag='Weekly hedge')

    def intraday_straddle(self, quantity_in_lots, exit_time=(15, 28), websocket=False, wait_for_equality=False,
                          monitor_sl=False, move_sl=False, shared_data=None, **kwargs):

        """Params:
        quantity_in_lots: Quantity in lots
        exit_time: Time to exit the trade in (hours, minutes, seconds) format | Default: (15, 28)
        websocket: Whether to use websocket or not | Default: False
        wait_for_equality: Whether to wait for the option prices to be equal | Default: False
        monitor_sl: Whether to monitor stop loss | Default: False
        move_sl: Whether to move stop loss | Default: False
        kwargs: 'stoploss': Stop loss  | Default: 1.5 and 1.7 on expiries, 'target_disparity':
        Target disparity"""

        order_tag = 'Intraday straddle'

        # Splicing orders
        spliced_orders = self.splice_orders(quantity_in_lots)

        # Finding equal strike, setting expiry
        equal_strike, call_symbol, \
            call_token, put_symbol, \
            put_token, call_price, put_price = self.find_equal_strike(exit_time=exit_time, websocket=websocket,
                                                                      wait_for_equality=wait_for_equality, **kwargs)
        expiry = self.current_expiry

        notifier(f'{self.name}: Initiating intraday trade on {equal_strike} strike.', self.webhook_url)

        # Placing orders
        call_avg_price, put_avg_price = self.place_combined_order(expiry, 'SELL', quantity_in_lots, strike=equal_strike,
                                                                  return_avg_price=True, order_tag=order_tag)

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

        self.log_order(equal_strike, expiry, 'SELL', call_avg_price, put_avg_price, order_tag)
        summary_message = '\n'.join(f'{k}: {v}' for k, v in self.order_log[order_tag][0].items())
        notifier(summary_message, self.webhook_url)
        sleep(1)

        # After placing the orders and stoploss orders
        in_trade = True
        call_sl_hit = False
        put_sl_hit = False
        error_faced = False

        def price_tracker():

            nonlocal call_price, put_price
            loop_number = 0
            while in_trade and not error_faced:
                if websocket:
                    underlying_price = price_dict.get(self.token, 0)['ltp']
                    call_price = price_dict.get(call_token, 0)['ltp']
                    put_price = price_dict.get(put_token, 0)['ltp']
                else:
                    underlying_price = self.fetch_ltp()
                    call_price = fetchltp('NFO', call_symbol, call_token)
                    put_price = fetchltp('NFO', put_symbol, put_token)
                iv = straddleiv(call_price, put_price, underlying_price, equal_strike, timetoexpiry(expiry))
                if loop_number % 25 == 0:
                    print(f'Index: {self.name}\nTime: {currenttime().time()}\nStrike: {equal_strike}\n' +
                          f'Call SL: {call_sl_hit}\nPut SL: {put_sl_hit}\n' +
                          f'Call Price: {call_price}\nPut Price: {put_price}\n' +
                          f'Total price: {call_price + put_price}\nIV: {iv}\n')
                loop_number += 1

        price_updater = Thread(target=price_tracker)

        if monitor_sl:

            price_updater.start()

            def check_sl_orders(order_ids, side, data=shared_data):

                nonlocal orderbook
                # If first time checking, fetch orderbook and do not use shared data
                if data is None:
                    orderbook = fetch_book('orderbook')
                # If not first time checking, use shared data if it is not too old
                else:
                    if currenttime() - data.updated_time < timedelta(seconds=15) and data.orderbook_data is not None:
                        # print(f'{self.name} {side} Using shared data. Updated at {data.updated_time:%H:%M:%S}')
                        orderbook = data.orderbook_data
                    else:
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

            sleep(10)
            # Monitoring begins here
            while currenttime().time() < time(*exit_time) and not call_sl_hit and not put_sl_hit:
                try:
                    call_sl_hit, call_sl_orders_complete = check_sl_orders(call_stoploss_order_ids, 'call')
                    put_sl_hit, put_sl_orders_complete = check_sl_orders(put_stoploss_order_ids, 'put')
                except Exception as e:
                    notifier(f'{self.name} Error: {e}', self.webhook_url)
                    error_faced = True
                    price_updater.join()
                    raise Exception(f'Error: {e}')

                if call_sl_hit:

                    if call_sl_orders_complete:
                        pass
                    else:
                        for quantity in spliced_orders:
                            placeorder(call_symbol, call_token, quantity * self.lot_size,
                                       'BUY', 'MARKET')

                    if move_sl:
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
                            # print(f'{self.name} put stoploss not triggered yet')
                            sleep(5)

                elif put_sl_hit:

                    if put_sl_orders_complete:
                        pass
                    else:
                        for quantity in spliced_orders:
                            placeorder(put_symbol, put_token, quantity * self.lot_size,
                                       'BUY', 'MARKET')

                    if move_sl:
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
                            # print(f'{self.name} call stoploss not triggered yet')
                            sleep(5)
                else:
                    current_time = currenttime().time().strftime('%H:%M:%S')
                    # print(f'{current_time} {self.name} stoplosses not triggered')
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
            points_captured = (put_avg_price - put_exit_price) + (call_avg_price - call_exit_price)
            stoploss_type = 'Both'

        elif call_sl_hit:
            for quantity in spliced_orders:
                placeorder(put_symbol, put_token, quantity * self.lot_size, 'BUY', put_price * 1.1,
                           'Exit order')
                sleep(0.3)
            notifier(f'{self.name}: Exited put. Call stoploss was triggered.', self.webhook_url)
            call_exit_price = lookup_and_return(orderbook, 'orderid',
                                                call_stoploss_order_ids, 'averageprice').astype(float).mean()
            put_exit_price = put_price
            points_captured = (call_avg_price - call_exit_price) + (put_avg_price - put_exit_price)
            stoploss_type = 'Call'

        elif put_sl_hit:
            for quantity in spliced_orders:
                placeorder(call_symbol, call_token, quantity * self.lot_size, 'BUY', call_price * 1.1,
                           'Exit order')
                sleep(0.3)
            notifier(f'{self.name}: Exited call. Put stoploss was triggered.', self.webhook_url)
            put_exit_price = lookup_and_return(orderbook, 'orderid',
                                               put_stoploss_order_ids, 'averageprice').astype(float).mean()
            call_exit_price = call_price
            points_captured = (put_avg_price - put_exit_price) + (call_avg_price - call_exit_price)
            stoploss_type = 'Put'

        else:
            self.place_combined_order(expiry, 'BUY', quantity_in_lots, strike=equal_strike)
            notifier(f'{self.name}: Exited positions. No stoploss was triggered.', self.webhook_url)
            call_exit_price = call_price
            put_exit_price = put_price
            points_captured = (call_avg_price + put_avg_price) - (call_exit_price + put_exit_price)
            stoploss_type = 'None'

        exit_dict = {'Call exit price': call_exit_price, 'Put exit price': put_exit_price,
                     'Total exit price': call_exit_price + put_exit_price, 'Points captured': points_captured,
                     'Stoploss': stoploss_type}

        try:
            self.order_log[order_tag][0].update(exit_dict)
        except Exception as e:
            notifier(f'{self.name}: Error updating order list with exit details. {e}', self.webhook_url)

        in_trade = False
        if websocket:
            sws.close_connection()

    def intraday_straddle_delta_hedged(self, quantity_in_lots, exit_time=(15, 30), websocket=False,
                                       wait_for_equality=False, delta_threshold=1, **kwargs):

        # Finding equal strike
        equal_strike, call_symbol, put_symbol, call_token, put_token, \
            call_price, put_price = self.find_equal_strike(exit_time, websocket, wait_for_equality, **kwargs)
        expiry = self.current_expiry
        print(f'Index: {self.name}, Strike: {equal_strike}, Call: {call_price}, Put: {put_price}')
        notifier(f'{self.name}: Initiating intraday trade on {equal_strike} strike.', self.webhook_url)

        # Placing orders
        self.place_combined_order(expiry, 'SELL', quantity_in_lots, strike=equal_strike, return_avg_price=True,
                                  order_tag='Intraday straddle with delta')

        positions = {

            f'{self.name} {equal_strike} {expiry} CE': {'token': call_token,
                                                        'quantity': -1 * quantity_in_lots * self.lot_size,
                                                        'delta_quantity': 0},
            f'{self.name} {equal_strike} {expiry} PE': {'token': put_token,
                                                        'quantity': -1 * quantity_in_lots * self.lot_size,
                                                        'delta_quantity': 0}
        }

        synthetic_fut_call = f'{self.name} {equal_strike} {expiry} CE'
        synthetic_fut_put = f'{self.name} {equal_strike} {expiry} PE'
        delta_threshold = delta_threshold * self.lot_size

        while currenttime().time() < time(*exit_time):

            position_df = pd.DataFrame(positions).T
            if websocket:
                underlying_price = price_dict.get(self.token, 0)['ltp']
                position_df['ltp'] = position_df['token'].apply(lambda x: price_dict.get(x, 'None')['ltp'])
            else:
                underlying_price = self.fetch_ltp()
                position_df['ltp'] = position_df.index.map(lambda x: fetchltp('NFO', *fetch_symbol_token(x)))

            position_df[['iv', 'delta', 'gamma']] = position_df.apply(lambda row:
                                                                      calc_greeks(row.name, row.ltp, underlying_price),
                                                                      axis=1).tolist()

            position_df['total_quantity'] = position_df['quantity'] + position_df['delta_quantity']
            position_df['delta'] = position_df.delta * position_df.total_quantity
            position_df['gamma'] = position_df.gamma * position_df.total_quantity
            position_df.loc['Total'] = position_df.agg({'delta': 'sum', 'gamma': 'sum',
                                                        'iv': 'mean', 'ltp': 'mean'})
            current_delta = position_df.loc['Total', 'delta']
            current_gamma = position_df.loc['Total', 'gamma']

            print(f'\n**** Starting Loop ****\n{position_df.drop(["token"], axis=1).to_string()}\n' +
                  f'\nCurrent delta: {current_delta}\n')

            if abs(current_delta) > delta_threshold:

                if current_delta > 0:  # We are long
                    lots_to_sell = round(abs(current_delta) / self.lot_size, 0)
                    notifier(f'Delta greater than {delta_threshold}. Selling {lots_to_sell} ' +
                             f'synthetic futures to reduce delta.\n', self.webhook_url)
                    place_synthetic_fut_order(self.name, equal_strike, expiry, 'SELL', lots_to_sell * self.lot_size)
                    positions[synthetic_fut_call]['delta_quantity'] -= lots_to_sell * self.lot_size
                    positions[synthetic_fut_put]['delta_quantity'] += lots_to_sell * self.lot_size

                else:  # We are short
                    lots_to_buy = round(abs(current_delta) / self.lot_size, 0)
                    notifier(f'Delta less than {-delta_threshold}. Buying {lots_to_buy} ' +
                             f'synthetic futures to reduce delta.\n', self.webhook_url)
                    place_synthetic_fut_order(self.name, equal_strike, expiry, 'BUY', lots_to_buy * self.lot_size)
                    positions[synthetic_fut_call]['delta_quantity'] += lots_to_buy * self.lot_size
                    positions[synthetic_fut_put]['delta_quantity'] -= lots_to_buy * self.lot_size

            sleep(2)

        # Closing the main positions along with the delta positions if any are open
        notifier(f'Intraday straddle with delta: Closing positions.', self.webhook_url)
        self.place_combined_order(expiry, 'BUY', quantity_in_lots, strike=equal_strike,
                                  order_tag='Intraday straddle with delta')

        # Squaring off the delta positions
        call_delta_quantity = positions[synthetic_fut_call]['delta_quantity']
        put_delta_quantity = positions[synthetic_fut_put]['delta_quantity']

        if call_delta_quantity != 0 and put_delta_quantity != 0:
            assert call_delta_quantity == -1 * put_delta_quantity
            quantity_to_square_up = abs(call_delta_quantity)

            if call_delta_quantity > 0:
                action = 'SELL'
            else:
                action = 'BUY'

            self.place_synthetic_fut_order(equal_strike, expiry, action, quantity_to_square_up)
            notifier(f'Intraday Straddle with delta: Squared off delta positions. ' +
                     f'{action} {quantity_to_square_up} synthetic futures.', self.webhook_url)
        elif call_delta_quantity == 0 and put_delta_quantity == 0:
            notifier('No delta positions to square off.', self.webhook_url)
        else:
            raise AssertionError('Delta positions are not balanced.')
