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
from autotrader import scrips, holidays, blackscholes as bs
from collections import defaultdict
import yfinance as yf
import re
import logging
import functools

global login_data, obj

large_order_threshold = 10

today = datetime.now().strftime('%Y-%m-%d %H-%M')
log_filename = f'errors-{today}.log'

logging.basicConfig(filename=log_filename, level=logging.ERROR, format='%(asctime)s:%(levelname)s:%(message)s')


def log_errors(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in function {func.__name__}: {e}")
            raise
    return wrapper


def convert_to_serializable(data):
    if isinstance(data, dict):
        return {k: convert_to_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_serializable(item) for item in data]
    elif hasattr(data, 'tolist'):  # Check for numpy arrays
        return data.tolist()
    elif hasattr(data, 'item'):  # Check for numpy scalar types, e.g., numpy.int32
        return data.item()
    else:
        return data


def append_data_to_json(data_dict: defaultdict, file_name: str):
    # Attempt to read the existing data from the JSON file
    try:
        with open(file_name, "r") as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        # If the file doesn't exist or has invalid JSON content, create an empty list and write it to the file
        data = []
        with open(file_name, "w") as file:
            json.dump(data, file)

    # Convert the defaultdict to a regular dict, make it JSON serializable, and append it to the list
    serializable_data = convert_to_serializable(dict(data_dict))
    data.append(serializable_data)

    # Write the updated data back to the JSON file with indentation
    with open(file_name, "w") as file:
        json.dump(data, file, indent=4, default=str)


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


def parse_symbol(symbol):
    match = re.match(r"([A-Za-z]+)(\d{2}[A-Za-z]{3}\d{2})(\d+)(\w+)", symbol)
    if match:
        return match.groups()
    return None


def fetch_book(book):
    def fetch_data(fetch_func, description, max_attempts=6, sleep_duration=2):
        for attempt in range(1, max_attempts + 1):
            try:
                data = fetch_func()['data']
                return data
            except DataException:
                if attempt == max_attempts:
                    raise Exception(f'Failed to fetch {description}.')
                else:
                    sleep(sleep_duration)
            except Exception as e:
                if attempt == max_attempts:
                    raise Exception(f'Failed to fetch {description}.')
                else:
                    print(f'Error {attempt} in fetching {description}: {e}')
                    sleep(sleep_duration)

    if book == 'orderbook':
        return fetch_data(obj.orderBook, 'orderbook')
    elif book in {'positions', 'position'}:
        return fetch_data(obj.position, 'positions')
    else:
        raise ValueError(f"Invalid book type '{book}'.")


def lookup_and_return(book, field_to_lookup, value_to_lookup, field_to_return):
    def filter_and_return(data: list):
        if isinstance(value_to_lookup, list):
            bucket = [entry[field_to_return] for entry in data
                      if entry[field_to_lookup] in value_to_lookup
                      and entry[field_to_lookup] != '']
            assert len(bucket) == len(value_to_lookup)
            return np.array(bucket)
        elif isinstance(value_to_lookup, str):
            bucket = [entry[field_to_return] for entry in data
                      if entry[field_to_lookup] == value_to_lookup
                      and entry[field_to_lookup] != '']
            return 0 if len(bucket) == 0 else (bucket[0] if len(bucket) == 1 else np.array(bucket))
        else:
            raise ValueError('Invalid valuetolookup')

    if isinstance(book, list):
        return filter_and_return(book)
    elif isinstance(book, str) and book in {'orderbook', 'positions'}:
        for _ in range(3):
            book_data = fetch_book(book)
            return filter_and_return(book_data)
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
    return datetime.now(ist).replace(tzinfo=None)


def simulate_option_movement(spot, strike, time_to_expiry, flag, simulated_move=0.2, r=0.06, vol=None, price=None,
                             print_results=False):
    if price is None and vol is None:
        raise ValueError('Either price or vol must be specified.')
    flag = flag.lower()[0]
    price_func = bs.put if flag.upper().startswith('P') else bs.call
    simulated_move = simulated_move / 100
    simulated_spot = (spot * (1 + simulated_move)) if flag == 'p' else (spot * (1 - simulated_move))
    if vol is None:
        try:
            vol = bs.implied_volatility(price, spot, strike, time_to_expiry, r, flag)
        except ValueError:
            return None
    if price is None:
        price = price_func(spot, strike, time_to_expiry, r, vol)
    current_delta = bs.delta(spot, strike, time_to_expiry, r, vol, flag)
    new_delta = bs.delta(simulated_spot, strike, time_to_expiry, r, vol, flag)
    delta_change = new_delta - current_delta
    average_delta = abs((new_delta + current_delta) / 2)
    new_price = price_func(simulated_spot, strike, time_to_expiry, r, vol)
    price_gain = price - new_price

    if print_results:
        print(f'Current Delta: {current_delta:.2f}\nNew Delta: {new_delta:.2f}\n'
              f'Delta Change: {delta_change:.2f}\nAverage Delta: {average_delta:.2f}\n'
              f'New Price: {new_price:.2f}\nPrice Change: {price_gain:.2f}\n'
              f'Volatility: {vol:.2f}\nSimulated Spot: {simulated_spot:.2f}\n'
              f'Simulated Move: {simulated_move * 100:.2f}%\n')

    return [price_gain, average_delta]


def spot_price_from_future(future_price, interest_rate, time_to_future):
    """
    Calculate the spot price from the future price, interest rate, and time.

    :param future_price: float, the future price of the asset
    :param interest_rate: float, the annual interest rate (as a decimal, e.g., 0.05 for 5%)
    :param time_to_future: float, the time to maturity (in years)
    :return: float, the spot price of the asset
    """
    spot_price = future_price / ((1 + interest_rate) ** time_to_future)
    return spot_price

def fetch_lot_size(name):
    return int(scrips.loc[(scrips.symbol.str.startswith(name)) & (scrips.exch_seg == 'NFO'), 'lotsize'].iloc[0])


def fetch_symbol_token(name):
    """Fetches symbol & token for a given scrip name. Provide just a single world if
    you want to fetch the symbol & token for the cash segment. If you want to fetch the
    symbol & token for the options segment, provide name in the format '{name} {strike} {expiry} {option_type}'.
    Expiry should be in the DDMMMYY format. Optiontype should be CE or PE."""

    if len(name.split()) == 1:
        if name in ['BANKNIFTY', 'NIFTY']:
            symbol, token = scrips.loc[(scrips.name == name) &
                                       (scrips.exch_seg == 'NSE'), ['symbol', 'token']].values[0]
        elif name == 'FINNIFTY':
            futures = scrips.loc[(scrips.name == name) &
                                 (scrips.instrumenttype == 'FUTIDX'), ['expiry', 'symbol', 'token']]
            futures["expiry"] = pd.to_datetime(futures["expiry"], format='%d%b%Y')
            futures = futures.sort_values(by="expiry")
            symbol, token = futures.iloc[0][['symbol', 'token']].values
        else:
            symbol, token = scrips.loc[
                (scrips.name == name) &
                (scrips.exch_seg == 'NSE') &
                (scrips.symbol.str.endswith('EQ')), ['symbol', 'token']
            ].values[0]
    elif len(name.split()) == 4:
        name, strike, expiry, option_type = name.split()
        symbol = name + expiry + str(strike) + option_type
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


def calculate_iv(opt_price, spot, strike, tte, opt_type):
    try:
        return bs.implied_volatility(opt_price, spot, strike, tte, 0.06,
                                     opt_type)
    except ValueError:
        return None


def straddle_iv(callprice, putprice, spot, strike, timeleft):

    call_iv = calculate_iv(callprice, spot, strike, timeleft, 'CE')
    put_iv = calculate_iv(putprice, spot, strike, timeleft, 'PE')

    if call_iv is not None and put_iv is not None:
        avg_iv = (call_iv + put_iv) / 2
    else:
        avg_iv = call_iv if put_iv is not None else call_iv

    return call_iv, put_iv, avg_iv


def calc_greeks(position_string, position_price, underlying_price):
    """Fetches the price, iv and delta of a stock"""

    name, strike, expiry, option_type = position_string.split()
    strike = int(strike)
    time_left = timetoexpiry(expiry)

    iv = bs.implied_volatility(position_price, underlying_price, strike, time_left, 0.05, option_type) * 100
    delta = bs.delta(underlying_price, strike, time_left, 0.05, iv, option_type)
    gamma = bs.gamma(underlying_price, strike, time_left, 0.05, iv)

    return iv, delta, gamma


def charges(buy_premium, contract_size, num_contracts, freeze_quantity=None):

    if freeze_quantity:
        number_of_orders = np.ceil(num_contracts/freeze_quantity)
    else:
        number_of_orders = 1

    buy_brokerage = 40*number_of_orders
    sell_brokerage = 40*number_of_orders
    transaction_charge_rate = 0.05 / 100
    stt_ctt_rate = 0.0625 / 100
    gst_rate = 18 / 100

    buy_transaction_charges = buy_premium * contract_size * num_contracts * transaction_charge_rate
    sell_transaction_charges = buy_premium * contract_size * num_contracts * transaction_charge_rate
    stt_ctt = buy_premium * contract_size * num_contracts * stt_ctt_rate

    buy_gst = (buy_brokerage + buy_transaction_charges) * gst_rate
    sell_gst = (sell_brokerage + sell_transaction_charges) * gst_rate

    total_charges = buy_brokerage + sell_brokerage + buy_transaction_charges + sell_transaction_charges + stt_ctt + buy_gst + sell_gst
    charges_per_share = total_charges / (num_contracts * contract_size)

    return round(charges_per_share, 1)


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


def cancel_pending_orders(order_ids, variety="STOPLOSS"):
    if order_ids:
        if isinstance(order_ids, (list, np.ndarray)):
            for order_id in order_ids:
                obj.cancelOrder(order_id, variety)
        else:
            obj.cancelOrder(order_ids, variety)


class MyWebSocketApp(SmartWebSocketV2):

    global obj, login_data

    def __init__(self, webhook_url=None):
        auth_token = login_data['data']['jwtToken']
        feed_token = obj.getfeedToken()
        api_key = obj.api_key
        client_code = obj.userId
        super().__init__(auth_token, api_key, client_code, feed_token)
        self.price_dict = {}
        self.option_watchlist = {}
        self.updated_time = None
        self.iv_log = defaultdict(dict)
        self.webhook_url = webhook_url

    def start_websocket(self):

        websocket_started = False

        tokens = ['26000', '26009']

        correlation_id = 'websocket'
        mode = 1
        token_list = [{'exchangeType': 1, 'tokens': tokens}]

        def on_data(wsapp, message):
            self.price_dict[message['token']] = {'ltp': message['last_traded_price'] / 100,
                                                 'timestamp': datetime.fromtimestamp(
                                                  message['exchange_timestamp'] / 1000).strftime('%H:%M:%S')}

        def on_open(wsapp):
            nonlocal websocket_started
            print("Starting Websocket")
            self.subscribe(correlation_id, mode, token_list)
            websocket_started = True

        def on_error(wsapp, error):
            print(error)

        def on_close(wsapp):
            print("Close")

        # Assign the callbacks.
        self.on_open = on_open
        self.on_data = on_data
        self.on_error = on_error
        self.on_close = on_close

        Thread(target=self.connect).start()

        while not websocket_started:
            print('Waiting for websocket to start')
            sleep(1)

    def parse_price_dict(self):
        new_price_dict = {scrips.loc[scrips.token == token]['symbol'].values[0]: value for token, value in
                          self.price_dict.items()}
        return new_price_dict

    def add_options(self, *indices, range_of_strikes=10):

        for index in indices:
            def help_func(strike):
                _, c_token = fetch_symbol_token(f'{index.name} {strike} {index.current_expiry} CE')
                _, p_token = fetch_symbol_token(f'{index.name} {strike} {index.current_expiry} PE')
                return c_token, p_token

            ltp = index.fetch_ltp()
            current_strike = findstrike(ltp, index.base)
            strike_range = np.arange(current_strike - (index.base * (range_of_strikes / 2)),
                                     current_strike + (index.base * (range_of_strikes / 2)), index.base)
            strike_range = map(int, strike_range)
            data = [help_func(strike) for strike in strike_range]
            call_token_list, put_token_list = zip(*data)
            self.subscribe('websocket', 1, [{'exchangeType': 2,
                                             'tokens': list(call_token_list) + list(put_token_list)}])
        sleep(3)

    def option_screener(self, n_values=36, iv_threshold=1.3, sleep_time=5, exit_time=(15, 30)):

        while currenttime().time() < time(*exit_time):

            parsed_dict = self.parse_price_dict()

            indices = ['NIFTY', 'BANKNIFTY']
            interest_rate = 0.065  # Replace this with the actual interest rate

            # Iterating over the indices
            for index in indices:

                instrument = parsed_dict[index]
                self.option_watchlist[index] = {}
                expiry = [parse_symbol(symbol)[1] for symbol in parsed_dict.keys()
                          if symbol.startswith(index) and 'CE' in symbol][0]
                time_to_expiry = timetoexpiry(expiry)

                # Iterating over the strikes and calculating the IV
                for symbol, info in parsed_dict.items():

                    if symbol.startswith(index) and 'CE' in symbol:
                        put_symbol = symbol.replace('CE', 'PE')
                        put_option = parsed_dict[put_symbol]

                        s = instrument['ltp']
                        k = float(parse_symbol(symbol)[2])
                        t = time_to_expiry
                        r = interest_rate

                        call_price = info['ltp']
                        put_price = put_option['ltp']

                        try:
                            call_iv = bs.implied_volatility(call_price, s, k, t, r, 'C')
                        except ValueError:
                            call_iv = None
                        try:
                            put_iv = bs.implied_volatility(put_price, s, k, t, r, 'P')
                        except ValueError:
                            put_iv = None

                        # Update the IV log
                        if call_iv is not None or put_iv is not None:
                            if k not in self.iv_log[index]:
                                self.iv_log[index][k] = {'call_ivs': [], 'put_ivs': [], 'total_ivs': [],
                                                         'times': [], 'count': 0, 'last_notified_time': currenttime()}
                            if call_iv is not None:
                                self.iv_log[index][k]['call_ivs'].append(call_iv)
                            if put_iv is not None:
                                self.iv_log[index][k]['put_ivs'].append(put_iv)
                            if call_iv is not None and put_iv is not None:
                                self.iv_log[index][k]['total_ivs'].append((call_iv + put_iv) / 2)

                            self.iv_log[index][k]['times'].append(datetime.strptime(info['timestamp'],
                                                                                    '%H:%M:%S').time())
                            self.iv_log[index][k]['count'] += 1

                        # Calculate the running average

                        call_ivs = self.iv_log[index][k]['call_ivs'][-n_values:] if k in self.iv_log[
                            index] else []
                        put_ivs = self.iv_log[index][k]['put_ivs'][-n_values:] if k in self.iv_log[
                            index] else []
                        total_ivs = self.iv_log[index][k]['total_ivs'][-n_values:] if k in self.iv_log[
                            index] else []

                        running_avg_call_iv = sum(call_ivs) / len(call_ivs) if call_ivs else None
                        running_avg_put_iv = sum(put_ivs) / len(put_ivs) if put_ivs else None
                        running_avg_total_iv = sum(total_ivs) / len(total_ivs) if total_ivs else None

                        if call_iv and call_iv > iv_threshold * running_avg_call_iv \
                                and self.iv_log[index][k]['last_notified_time'] < currenttime() - timedelta(minutes=5):
                            notifier(f'Call IV for {index} {k} greater than average', self.webhook_url)
                            self.iv_log[index][k]['last_notified_time'] = currenttime()

                        if put_iv and put_iv > iv_threshold * running_avg_put_iv \
                                and self.iv_log[index][k]['last_notified_time'] < currenttime() - timedelta(minutes=5):
                            notifier(f'Put IV for {index} {k} greater than average', self.webhook_url)
                            self.iv_log[index][k]['last_notified_time'] = currenttime()

                        self.option_watchlist[index][k] = {'call_price': call_price, 'put_price': put_price,
                                                           'call_iv': call_iv, 'put_iv': put_iv,
                                                           'running_avg_call_iv': running_avg_call_iv,
                                                           'running_avg_put_iv': running_avg_put_iv,
                                                           'running_avg_total_iv': running_avg_total_iv}

            self.updated_time = max([datetime.strptime(info['timestamp'], '%H:%M:%S').time()
                                     for info in parsed_dict.values()])

            sleep(sleep_time)


class SharedData:
    def __init__(self):
        self.position_data = None
        self.orderbook_data = None
        self.updated_time = None
        self.error_info = None
        self.force_stop = False

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
        while currenttime().time() < time(*exit_time) and not self.force_stop:
            self.fetch_data()
            sleep(sleep_time)


class Index:
    """Initialize an index with the name of the index in uppercase"""

    def __init__(self, name, webhook_url=None, websocket=None):

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

        self.intraday_straddle_forced_exit = False

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

        if websocket:
            try:
                websocket.subscribe('websocket', 1, [{'exchangeType': self.exchange_type, 'tokens': [self.token]}])
                sleep(2)
                print(f'{self.name}: Subscribed underlying to the websocket')
            except Exception as e:
                print(f'{self.name}: Websocket subscription failed. {e}')

    def fetch_freeze_limit(self):
        try:
            freeze_qty_url = 'https://archives.nseindia.com/content/fo/qtyfreeze.xls'
            response = requests.get(freeze_qty_url, timeout=10)  # Set the timeout value
            response.raise_for_status()  # Raise an exception if the response contains an HTTP error status
            df = pd.read_excel(response.content)
            df.columns = df.columns.str.strip()
            df['SYMBOL'] = df['SYMBOL'].str.strip()
            freeze_qty = df[df['SYMBOL'] == self.name]['VOL_FRZ_QTY'].values[0]
            freeze_qty_in_lots = freeze_qty / self.lot_size
            return freeze_qty_in_lots
        except requests.exceptions.Timeout as e:
            notifier(f'Timeout error in fetching freeze limit for {self.name}: {e}', self.webhook_url)
            freeze_qty_in_lots = 30
            return freeze_qty_in_lots
        except requests.exceptions.HTTPError as e:
            notifier(f'HTTP error in fetching freeze limit for {self.name}: {e}', self.webhook_url)
            freeze_qty_in_lots = 30
            return freeze_qty_in_lots
        except Exception as e:
            notifier(f'Error in fetching freeze limit for {self.name}: {e}', self.webhook_url)
            freeze_qty_in_lots = 30
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
            ltp = fetchltp('NFO', self.symbol, self.token)
            self.ltp = spot_price_from_future(ltp, 0.06, timetoexpiry(self.fut_expiry))
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
            if loops > large_order_threshold:
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

        order_prefix = f'{order_tag}: ' if order_tag else ''

        if all(call_order_statuses == 'complete') and all(put_order_statuses == 'complete'):
            notifier(f'{order_prefix}Order(s) placed successfully for {buy_or_sell} {self.name} ' +
                     f'{strike_info} {expiry} {quantity_in_lots} lot(s).', self.webhook_url)
            call_order_avg_price = lookup_and_return(orderbook, 'orderid', call_order_id_list,
                                                     'averageprice').astype(float).mean()
            put_order_avg_price = lookup_and_return(orderbook, 'orderid', put_order_id_list,
                                                    'averageprice').astype(float).mean()
            if return_avg_price:
                return call_order_avg_price, put_order_avg_price
            else:
                return
        elif all(call_order_statuses == 'rejected') and all(put_order_statuses == 'rejected'):
            notifier(f'{order_prefix}All orders rejected for {buy_or_sell} {self.name} ' +
                     f'{strike_info} {expiry} {quantity_in_lots} lot(s).', self.webhook_url)
            raise Exception('Orders rejected')
        else:
            notifier(f'{order_prefix}ERROR. Order statuses uncertain for {buy_or_sell} {self.name} ' +
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

    def find_equal_strike(self, exit_time, websocket, wait_for_equality, target_disparity, expiry=None):

        expiry = expiry or self.current_expiry
        ltp = self.fetch_ltp() if not websocket else websocket.price_dict.get(self.token, 0)['ltp']
        current_strike = findstrike(ltp, self.base)
        strike_range = np.arange(current_strike - self.base * 2, current_strike + self.base * 2, self.base)

        def fetch_data(strike, exp):
            c_symbol, c_token = fetch_symbol_token(f'{self.name} {strike} {exp} CE')
            p_symbol, p_token = fetch_symbol_token(f'{self.name} {strike} {exp} PE')
            return c_symbol, c_token, p_symbol, p_token

        def fetch_ltps(tokens, symbols, socket):
            if socket:
                return np.array([websocket.price_dict.get(token, {'ltp': 0})['ltp'] for token in tokens])
            else:
                return np.array([fetchltp('NFO', symbol, token) for symbol, token in zip(symbols, tokens)])

        def compute_disparities(c_ltps, p_ltps):
            return np.abs(c_ltps - p_ltps) / np.minimum(c_ltps, p_ltps) * 100

        data = [fetch_data(strike, expiry) for strike in strike_range]
        call_token_list, put_token_list = zip(*(tokens[1:4:2] for tokens in data))
        call_symbol_list, put_symbol_list = zip(*(symbols[0:3:2] for symbols in data))

        if websocket:
            websocket.subscribe('websocket', 1, [{'exchangeType': 2,
                                                  'tokens': list(call_token_list) + list(put_token_list)}])
            sleep(3)

        call_ltps, put_ltps = fetch_ltps(call_token_list, call_symbol_list, websocket), fetch_ltps(put_token_list,
                                                                                                   put_symbol_list,
                                                                                                   websocket)
        disparities = compute_disparities(call_ltps, put_ltps)

        if wait_for_equality:
            last_print_time = currenttime()
            print_interval = timedelta(seconds=0.0005)
            min_disparity_idx = np.argmin(disparities)
            min_disparity = disparities[min_disparity_idx]

            while min_disparity > target_disparity:
                if min_disparity < 10:
                    # Update only the minimum disparity strike data
                    call_ltp, put_ltp = fetch_ltps([call_token_list[min_disparity_idx]],
                                                   call_symbol_list[min_disparity_idx], websocket), \
                        fetch_ltps([put_token_list[min_disparity_idx]], put_symbol_list[min_disparity_idx], websocket)
                    disparities[min_disparity_idx] = compute_disparities(call_ltp, put_ltp)
                    single_check = True
                else:
                    # Update all strike data
                    call_ltps, put_ltps = fetch_ltps(call_token_list, call_symbol_list, websocket), fetch_ltps(
                        put_token_list, put_symbol_list, websocket)
                    disparities = compute_disparities(call_ltps, put_ltps)
                    single_check = False

                min_disparity_idx = np.argmin(disparities)
                min_disparity = disparities[min_disparity_idx]

                if (currenttime() - last_print_time) > print_interval:
                    print(f'Time: {currenttime().strftime("%H:%M:%S")}\n' +
                          f'Index: {self.name}\n' +
                          f'Current lowest disparity: {min_disparity:.2f}\n' +
                          f'Strike: {strike_range[min_disparity_idx]}\n' +
                          f'Single Strike: {single_check}\n')
                    last_print_time = currenttime()

                if (currenttime() + timedelta(minutes=5)).time() > time(*exit_time):
                    notifier('Equal strike tracker exited due to time limit.', self.webhook_url)
                    raise Exception('Equal strike tracker exited due to time limit.')

        idx = np.argmin(disparities)
        strike_to_trade, call_symbol, call_token, put_symbol, put_token, call_ltp, put_ltp = (strike_range[idx],
                                                                                              call_symbol_list[idx],
                                                                                              call_token_list[idx],
                                                                                              put_symbol_list[idx],
                                                                                              put_token_list[idx],
                                                                                              call_ltps[idx],
                                                                                              put_ltps[idx])

        return strike_to_trade, call_symbol, call_token, put_symbol, put_token, call_ltp, put_ltp

    @log_errors
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

        vix = yf.Ticker('^INDIAVIX')
        sleep(120)
        vix = vix.fast_info['last_price']

        order_tag = 'Overnight Short Straddle'

        if timetoexpiry(self.current_expiry, effective_time=True, in_days=True) > 4:  # far from expiry
            ltp = self.fetch_ltp()
            sell_strike = findstrike(ltp * strike_offset, self.base)
            call_ltp, put_ltp = fetch_straddle_price(self.name, self.current_expiry, sell_strike)
            call_iv, put_iv, iv = straddle_iv(call_ltp, put_ltp, ltp, sell_strike, timetoexpiry(self.current_expiry))
            if iv < vix * iv_threshold:
                notifier(f'IV is too low compared to VIX: IV {iv}, Vix {vix}.', self.webhook_url)
                return
            else:
                notifier(f'IV is fine compared to VIX: IV {iv}, Vix {vix}.', self.webhook_url)
                return
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

    @log_errors
    def intraday_straddle(self, quantity_in_lots, exit_time=(15, 28), websocket=None, wait_for_equality=False,
                          move_sl=False, shared_data=None, stoploss='dynamic', target_disparity=10,
                          catch_trend=False, trend_qty_ratio=0.5, trend_catcher_sl=0.003, safeguard_movement=0.003,
                          safeguard_iv_spike=1.3, smart_exit=False, take_profit=False, take_profit_points=np.inf,
                          take_profit_order=False):

        """ Params:
                quantity_in_lots: int
                exit_time: tuple
                websocket: websocket object
                wait_for_equality: bool
                move_sl: bool
                shared_data: class object
                stoploss: str|float
                target_disparity: float
                catch_trend: bool
                take_profit: bool
                trend_qty_ratio: float
                trend_catcher_sl: float
                take_profit_points: float
        """

        order_tag = 'Intraday straddle'
        expiry = self.next_expiry
        sleep_interval = 3 if shared_data is None and not take_profit else 0

        # Splicing orders
        spliced_orders = self.splice_orders(quantity_in_lots)

        # Finding equal strike
        equal_strike, call_symbol, \
            call_token, put_symbol, \
            put_token, call_price, put_price = self.find_equal_strike(exit_time=exit_time, websocket=websocket,
                                                                      wait_for_equality=wait_for_equality,
                                                                      target_disparity=target_disparity,
                                                                      expiry=expiry)

        notifier(f'{self.name}: Initiating intraday trade on {equal_strike} strike.', self.webhook_url)

        # Placing orders
        call_avg_price, put_avg_price = self.place_combined_order(expiry, 'SELL', quantity_in_lots, strike=equal_strike,
                                                                  return_avg_price=True, order_tag=order_tag)

        underlying_price = self.fetch_ltp()
        entry_spot = underlying_price

        # Placing stoploss orders
        if stoploss == 'fixed':
            if self.name == 'BANKNIFTY':
                sl = 1.7
            elif self.name == 'NIFTY':
                sl = 1.5
            else:
                sl = 1.6
        elif stoploss == 'dynamic':
            if self.name == 'BANKNIFTY' or timetoexpiry(expiry, in_days=True) < 1:
                sl = 1.7
            elif self.name == 'NIFTY':
                sl = 1.5
            else:
                sl = 1.6
        else:
            sl = stoploss

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

        # Recording initial iv information

        traded_call_iv, traded_put_iv, traded_avg_iv = straddle_iv(call_avg_price, put_avg_price, entry_spot,
                                                                   equal_strike, timetoexpiry(expiry))

        summary_message += f'\nTraded IV: {traded_avg_iv}'
        notifier(summary_message, self.webhook_url)
        sleep(1)

        @log_errors
        def price_tracker():

            nonlocal call_price, put_price, underlying_price, call_avg_price, put_avg_price, sl_hit_dict
            nonlocal take_profit_exit, call_exit_price, put_exit_price, mtm_price, profit_in_pts
            nonlocal call_iv, put_iv, avg_iv

            # Print settings
            last_print_time = currenttime()
            print_interval = timedelta(seconds=5)
            days_to_expiry = timetoexpiry(expiry, in_days=True)

            # Smart exit settings
            smart_exit_trg = False
            smart_exit_delta_threshold = 0.06 if days_to_expiry < 1 else 0.12
            smart_exit_notification_sent = False
            incremental_gains = 100
            average_delta = 1

            while in_trade and not error_faced:
                # Update prices
                if websocket:
                    underlying_price = websocket.price_dict.get(self.token, 0)['ltp']
                    call_price = websocket.price_dict.get(call_token, 0)['ltp']
                    put_price = websocket.price_dict.get(put_token, 0)['ltp']
                else:
                    underlying_price = self.fetch_ltp()
                    call_price = fetchltp('NFO', call_symbol, call_token)
                    put_price = fetchltp('NFO', put_symbol, put_token)

                # Fetch stop loss status
                callsl = sl_hit_dict['call']
                putsl = sl_hit_dict['put']

                # Calculate mtm price
                mtm_ce_price = call_exit_price if callsl else call_price
                mtm_pe_price = put_exit_price if putsl else put_price
                mtm_price = mtm_ce_price + mtm_pe_price

                # Calculate profit
                profit_in_pts = (call_avg_price + put_avg_price) - mtm_price
                profit_in_rs = profit_in_pts * self.lot_size * quantity_in_lots

                if take_profit and profit_in_pts > (take_profit_points + per_share_charges):
                    notifier(f'{self.name} take profit exit triggered\n'
                             f'Time: {currenttime().time()}\n'
                             f'Profit: {profit_in_pts}\n', self.webhook_url)
                    take_profit_exit = True

                call_iv, put_iv, avg_iv = straddle_iv(call_price, put_price,
                                                      underlying_price, equal_strike, timetoexpiry(expiry))

                if (callsl or putsl) and (call_iv or put_iv):
                    option_type = 'p' if callsl else 'c'
                    option_price = put_price if callsl else call_price
                    tracked_iv = put_iv if callsl and put_iv is not None else avg_iv
                    if tracked_iv is not None:
                        incremental_gains, average_delta = simulate_option_movement(underlying_price, equal_strike,
                                                                                    timetoexpiry(expiry), option_type,
                                                                                    simulated_move=0.2, r=0.06,
                                                                                    vol=tracked_iv, price=option_price)
                    else:
                        incremental_gains, average_delta = 100, 1

                    if smart_exit and average_delta < smart_exit_delta_threshold:
                        if not smart_exit_notification_sent:
                            notifier(f'{self.name} smart exit triggered\n'
                                     f'Time: {currenttime().time()}\n'
                                     f'Average delta: {average_delta}\n', self.webhook_url)
                            smart_exit_notification_sent = True
                            smart_exit_trg = True

                stoploss_message = ''
                if callsl:
                    stoploss_message += f'Call Exit Price: {mtm_ce_price}\nIncr. Gains: {incremental_gains}\n' + \
                                        f'Avg. Delta: {average_delta}\n'
                if putsl:
                    stoploss_message += f'Put Exit Price: {mtm_pe_price}\nIncr. Gains: {incremental_gains}\n' + \
                                        f'Avg. Delta: {average_delta}\n'

                if currenttime() - last_print_time > print_interval:
                    print(f'Index: {self.name}\nTime: {currenttime().time()}\nStrike: {equal_strike}\n' +
                          f'Underlying Price: {underlying_price}\nCall SL: {callsl}\nPut SL: {putsl}\n' +
                          f'Call Price: {call_price}\nPut Price: {put_price}\n' +
                          stoploss_message +
                          f'Total price: {call_price + put_price:0.2f}\nMTM Price: {mtm_price:0.2f}\n' +
                          f'Profit in points: {profit_in_pts:0.2f}\n' +
                          f'Profit Value: {profit_in_rs:0.2f}\nIV: {avg_iv}\nSmart Exit: {smart_exit_trg}\n')
                    last_print_time = currenttime()

        def process_order_statuses(order_book, order_ids, stop_loss=False, notify_url=None, context=''):

            nonlocal orderbook

            pending_text = "trigger pending" if stop_loss else "open"
            context = f'{context} ' if context else ''

            statuses = lookup_and_return(order_book, 'orderid', order_ids, 'status')

            if all(statuses == pending_text):
                return False, False

            elif all(statuses == "rejected") or all(statuses == "cancelled"):
                rejection_reasons = lookup_and_return(order_book, 'order_id', order_ids, 'text')
                if all(rejection_reasons == "17070 : The Price is out of the LPP range"):
                    return True, False
                else:
                    notifier(f'{context}Order rejected or cancelled. Reasons: {rejection_reasons[0]}', notify_url)
                    raise Exception(f"Orders rejected or cancelled.")

            elif stop_loss and all(statuses == "pending"):

                sleep(1)
                orderbook = fetch_book("orderbook")
                statuses = lookup_and_return(orderbook, 'orderid', order_ids, 'status')

                if all(statuses == "pending"):
                    try:
                        cancel_pending_orders(order_ids, 'NORMAL')
                    except Exception as e:
                        try:
                            cancel_pending_orders(order_ids, 'STOPLOSS')
                        except Exception as e:
                            notifier(f'{context}Could not cancel orders: {e}', notify_url)
                            raise Exception(f"Could not cancel orders: {e}")
                    notifier(f'{context}Orders pending and cancelled. Please check.', notify_url)
                    return True, False

                elif all(statuses == "complete"):
                    return True, True

                else:
                    raise Exception(f"Orders in unknown state.")

            elif all(statuses == "complete"):
                return True, True

            else:
                notifier(f'{context}Orders in unknown state. Statuses: {statuses}', notify_url)
                raise Exception(f"Orders in unknown state.")

        def fetch_orderbook_if_needed(data_class=shared_data, refresh_needed: bool = False):
            if data_class is None or refresh_needed:
                return fetch_book("orderbook")
            if (
                    currenttime() - data_class.updated_time < timedelta(seconds=15)
                    and data_class.orderbook_data is not None
            ):
                return data_class.orderbook_data
            return fetch_book("orderbook")

        def check_sl_orders(order_ids, side: str, data=shared_data, refresh=False):

            """ This function checks if the stop loss orders have been triggered or not. It also updates the order book
            in the nonlocal scope. This function is responsible for setting the exit prices."""

            nonlocal orderbook, call_exit_price, put_exit_price, call_price, put_price, underlying_price
            nonlocal traded_call_iv, traded_put_iv, traded_avg_iv, call_iv, put_iv, avg_iv, entry_spot

            orderbook = fetch_orderbook_if_needed(data, refresh)
            triggered, complete = process_order_statuses(
                orderbook, order_ids, stop_loss=True, notify_url=self.webhook_url, context=f'SL: {side}'
            )

            if not triggered and not complete:
                return False, False

            # Checking if there has been an unjustified trigger of stoploss without much movement in the underlying
            # We will also use IV to check if the stoploss was justified or not
            if triggered:
                movement_from_entry = abs((underlying_price/entry_spot)-1)
                present_iv = call_iv if side == 'call' else put_iv
                original_iv = traded_call_iv if side == 'call' else traded_put_iv
                iv_spike = present_iv/original_iv
                if movement_from_entry < safeguard_movement and iv_spike > safeguard_iv_spike:
                    notifier(f'{self.name} {side.capitalize()} stoploss triggered without much '
                             f'movement in the underlying or because of IV spike.\n'
                             f'Movement: {movement_from_entry*100:0.2f}. IV spike: {iv_spike}', self.webhook_url)
                else:
                    notifier(f'{self.name} {side.capitalize()} stoploss triggered. '
                             f'Movement: {movement_from_entry*100:0.2f}. '
                             f'IV spike: {iv_spike}',
                             self.webhook_url)

            if complete:
                exit_price = (
                    lookup_and_return(orderbook, "orderid", order_ids, "averageprice")
                    .astype(float)
                    .mean()
                )
            else:
                exit_price = call_price if side == "call" else put_price

            if side == "call":
                call_exit_price = exit_price
            else:
                put_exit_price = exit_price

            return True, complete

        def check_exit_conditions(time_now, time_of_exit, *bools):
            return time_now >= time_of_exit or any(bools)

        @log_errors
        def trend_catcher(sl_type, qty_ratio, trend_sl):

            nonlocal underlying_price, take_profit_exit, in_trade

            strike = findstrike(underlying_price, self.base)
            opt_type = "PE" if sl_type == "call" else "CE"
            symbol, token = fetch_symbol_token(
                f'{self.name} {strike} {expiry} {opt_type}')
            option_ltp = fetchltp('NFO', symbol, token)
            qty = max(int(quantity_in_lots * qty_ratio), 1)
            trend_spliced_orders = self.splice_orders(qty)
            for spliced_qty in trend_spliced_orders:
                placeorder(symbol, token, spliced_qty * self.lot_size, 'SELL', option_ltp * 0.9)
            sl_price = (underlying_price * (1 - trend_sl)) if sl_type == 'call' else (underlying_price * (1 + trend_sl))
            notifier(f'{self.name} {sl_type} trend catcher starting. ' +
                     f'Placed {qty} lots of {strike} {opt_type} at {option_ltp}. ' +
                     f'Stoploss price: {sl_price}, Underlying Price: {underlying_price}', self.webhook_url)
            trend_sl_hit = False
            last_print_time = currenttime()
            print_interval = timedelta(seconds=10)
            while not check_exit_conditions(currenttime().time(), time(*exit_time), trend_sl_hit,
                                            self.intraday_straddle_forced_exit, take_profit_exit, not in_trade):

                if sl_type == 'call':
                    trend_sl_hit = underlying_price < sl_price
                else:
                    trend_sl_hit = underlying_price > sl_price
                sleep(1)
                if currenttime() - last_print_time > print_interval:
                    last_print_time = currenttime()
                    print(f'{self.name} {sl_type} trend catcher running. ' +
                          f'Stoploss price: {sl_price}, Underlying Price: {underlying_price} ' +
                          f'Stoploss hit: {trend_sl_hit}')

            if trend_sl_hit:
                notifier(f'{self.name} {sl_type} trend catcher stoploss hit.', self.webhook_url)
            else:
                notifier(f'{self.name} {sl_type} trend catcher exiting.', self.webhook_url)

            for qty in trend_spliced_orders:
                placeorder(symbol, token, qty * self.lot_size, 'BUY', 'MARKET')

        @log_errors
        def take_profit_when_sl_hit(sl_type, sl_dict):

            nonlocal orderbook
            nonlocal call_exit_price, put_exit_price, call_symbol, call_token, put_symbol, put_token, break_even_price
            nonlocal take_profit_exit, take_profit_exit_triggered

            exit_price = call_exit_price if sl_type == 'call' else put_exit_price
            target_price = break_even_price - take_profit_points - exit_price
            target_price = round(target_price, 1)
            symbol, token = call_symbol, call_token if sl_type == 'put' else (put_symbol, put_token)
            other_side = 'call' if sl_type == 'put' else 'put'
            target_profit_order_ids = []
            for qty in spliced_orders:
                order_id = placeorder(symbol, token, qty * self.lot_size, 'BUY', target_price)
                target_profit_order_ids.append(order_id)
            notifier(f'{self.name} {other_side} target profit orders placed at {target_price}', self.webhook_url)

            sleep(1)
            orderbook = fetch_book("orderbook")
            while not check_exit_conditions(currenttime().time(), time(*exit_time), sl_dict[other_side],
                                            self.intraday_straddle_forced_exit):
                take_profit_exit_triggered, complete = process_order_statuses(orderbook, target_profit_order_ids,
                                                                              stop_loss=False,
                                                                              notify_url=self.webhook_url,
                                                                              context=f'Target Profit {other_side}')
                if take_profit_exit_triggered:
                    notifier(f'{self.name} {other_side} target profit order triggered.', self.webhook_url)
                    break
                if take_profit_exit:
                    notifier(f'{self.name} {other_side} target profit orders should have triggered', self.webhook_url)
                    break

            # Once out of the loop
            if not take_profit_exit_triggered:
                notifier(f'{self.name} {other_side} cancelling target profit orders', self.webhook_url)
                cancel_pending_orders(target_profit_order_ids, 'NORMAL')

        def process_sl_hit(sl_type, sl_dict, sl_orders_complete, symbol, token, other_symbol,
                           other_token, other_stoploss_order_ids, other_avg_price):

            nonlocal take_profit_exit, call_exit_price, put_exit_price

            if all(sl_dict.values()):
                # print(f'{self.name} both stoploss orders completed. Not processing {sl_type}.')
                return

            other_sl_type = 'call' if sl_type == 'put' else 'put'

            if not sl_orders_complete:
                for qty in spliced_orders:
                    placeorder(symbol, token, qty * self.lot_size, 'BUY', 'MARKET')

            if catch_trend:
                trend_thread = Thread(target=trend_catcher, args=(sl_type, trend_qty_ratio, trend_catcher_sl))
                trend_thread.start()

            if take_profit and take_profit_order:
                trend_thread = Thread(target=take_profit_when_sl_hit, args=(sl_type, sl_dict))
                trend_thread.start()

            if move_sl:
                for o_id in other_stoploss_order_ids:
                    obj.cancelOrder(o_id, 'STOPLOSS')

                other_stoploss_order_ids.clear()
                for qty in spliced_orders:
                    sl_order_id = placeSLorder(other_symbol, other_token, qty * self.lot_size,
                                               'BUY', other_avg_price, f'Stoploss {other_sl_type}')
                    other_stoploss_order_ids.append(sl_order_id)

            # notifier(f'{self.name} {sl_type} stoploss triggered and completed.', self.webhook_url)

            refresh = True
            while not check_exit_conditions(currenttime().time(), time(*exit_time), sl_dict[other_sl_type],
                                            self.intraday_straddle_forced_exit, take_profit_exit, take_profit_exit_triggered):

                sl_dict[other_sl_type], other_sl_orders_complete = check_sl_orders(other_stoploss_order_ids,
                                                                                   other_sl_type,
                                                                                   refresh=refresh)
                if sl_dict[other_sl_type]:
                    if not other_sl_orders_complete:
                        for qty in spliced_orders:
                            placeorder(other_symbol, other_token, qty * self.lot_size, 'BUY', 'MARKET')
                    break
                else:
                    sleep(sleep_interval)
                refresh = False

        def process_exit(call_stop_loss_hit, put_stop_loss_hit, already_exited=False):

            def exit_spliced_orders(orders, side):
                symbol = call_symbol if side == "call" else put_symbol
                token = call_token if side == "call" else put_token
                price = call_price if side == "call" else put_price

                for qty in orders:
                    placeorder(symbol, token, qty * self.lot_size, "BUY", price * 1.1, "Exit order")
                    sleep(0.3)

            if call_stop_loss_hit and put_stop_loss_hit:
                sl_type = "Both"
            elif call_stop_loss_hit and not already_exited:
                sl_type = "Call"
                exit_spliced_orders(spliced_orders, "put")
            elif put_stop_loss_hit and not already_exited:
                sl_type = "Put"
                exit_spliced_orders(spliced_orders, "call")
            else:
                sl_type = "None"
                self.place_combined_order(expiry, "BUY", quantity_in_lots, strike=equal_strike)

            return sl_type

        # After placing the orders and stoploss orders
        in_trade = True
        error_faced = False
        take_profit_exit = False
        take_profit_exit_triggered = False

        call_exit_price = 0
        put_exit_price = 0
        mtm_price = 0
        profit_in_pts = 0

        call_iv = 0
        put_iv = 0
        avg_iv = 0

        # Basic price information being set in outer scope as well
        total_avg_price = call_avg_price + put_avg_price
        per_share_charges = charges((call_avg_price + put_avg_price),
                                    self.lot_size, quantity_in_lots, self.lot_size)
        break_even_price = total_avg_price - per_share_charges
        notifier(f'{self.name}: Charges per share {per_share_charges} | Break even price {break_even_price}',
                 self.webhook_url)

        # Setting up stop loss dictionary and starting price thread
        sl_hit_dict = {'call': False, 'put': False}
        price_updater = Thread(target=price_tracker)
        price_updater.start()
        refresh_orderbook = True

        sleep(2)
        # Monitoring begins here
        while not check_exit_conditions(currenttime().time(), time(*exit_time), any(sl_hit_dict.values()),
                                        self.intraday_straddle_forced_exit, take_profit_exit):
            try:
                sl_hit_dict['call'], call_sl_orders_complete = check_sl_orders(call_stoploss_order_ids, 'call',
                                                                               refresh=refresh_orderbook)
                sl_hit_dict['put'], put_sl_orders_complete = check_sl_orders(put_stoploss_order_ids, 'put',
                                                                             refresh=refresh_orderbook)
            except Exception as e:
                notifier(f'{self.name} Error: {e}', self.webhook_url)
                error_faced = True
                price_updater.join()
                raise Exception(f'Error: {e}')

            if sl_hit_dict['call']:
                process_sl_hit('call', sl_hit_dict, call_sl_orders_complete, call_symbol, call_token,
                               put_symbol, put_token, put_stoploss_order_ids, put_avg_price)

            if sl_hit_dict['put']:
                process_sl_hit('put', sl_hit_dict, put_sl_orders_complete, put_symbol, put_token,
                               call_symbol, call_token, call_stoploss_order_ids, call_avg_price)
            refresh_orderbook = False
            sleep(sleep_interval)

        # After a complete exit is triggered
        sl_hit_call, sl_hit_put = sl_hit_dict["call"], sl_hit_dict["put"]

        if websocket:
            call_price = fetchltp("NFO", call_symbol, call_token)
            put_price = fetchltp("NFO", put_symbol, put_token)

        # Exiting remaining positions if any
        stop_loss_type = process_exit(sl_hit_call, sl_hit_put, take_profit_exit_triggered)

        # Cancelling pending orders
        pending_order_ids = (
            call_stoploss_order_ids + put_stoploss_order_ids
            if not (sl_hit_call or sl_hit_put)
            else put_stoploss_order_ids
            if sl_hit_call and not sl_hit_put
            else call_stoploss_order_ids
            if sl_hit_put and not sl_hit_call
            else []
        )
        cancel_pending_orders(pending_order_ids)

        # Exit price information
        c_exit_price = call_exit_price if sl_hit_call else call_price
        p_exit_price = put_exit_price if sl_hit_put else put_price

        exit_dict = {
            "Call exit price": c_exit_price,
            "Put exit price": p_exit_price,
            "Total exit price": c_exit_price + p_exit_price,
            "Points captured": profit_in_pts,
            "Stoploss": stop_loss_type,
        }

        try:
            self.order_log[order_tag][0].update(exit_dict)
        except Exception as e:
            notifier(f"{self.name}: Error updating order list with exit details. {e}", self.webhook_url)

        notifier(f'{self.name}: Exited positions\n' +
                 ''.join([f"{key}: {value}\n" for key, value in exit_dict.items()]), self.webhook_url)
        in_trade = False

    def intraday_straddle_delta_hedged(self, quantity_in_lots, exit_time=(15, 30), websocket=None,
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
                underlying_price = websocket.price_dict.get(self.token, 0)['ltp']
                position_df['ltp'] = position_df['token'].apply(lambda x: websocket.price_dict.get(x, 'None')['ltp'])
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
