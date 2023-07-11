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
from volstreet.SmartWebSocketV2 import SmartWebSocketV2
from volstreet.constants import scrips, holidays, symbol_df, logger
from volstreet import blackscholes as bs, datamodule as dm
from collections import defaultdict, deque
import yfinance as yf
from fuzzywuzzy import process
import re
import logging
import functools
import itertools
import traceback

global login_data, obj

LARGE_ORDER_THRESHOLD = 10
ERROR_NOTIFICATION_SETTINGS = {'url': None}


def set_error_notification_settings(key, value):
    global ERROR_NOTIFICATION_SETTINGS
    ERROR_NOTIFICATION_SETTINGS[key] = value


def log_errors(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            user_prefix = ERROR_NOTIFICATION_SETTINGS.get('user', '')
            logger.error(f"{user_prefix}Error in function {func.__name__}: {e}\nTraceback:{traceback.format_exc()}")
            notifier(
                f"{user_prefix}Error in function {func.__name__}: {e}\nTraceback:{traceback.format_exc()}",
                ERROR_NOTIFICATION_SETTINGS['url']
            )
            raise e
    return wrapper


class OptionChains(defaultdict):
    """An object for having option chains for multiple expiries.
    Each expiry is a dictionary with integer default values"""

    def __init__(self):
        super().__init__(lambda: defaultdict(lambda: defaultdict(int)))
        self.underlying_price = None
        self.exp_strike_pairs = []


class PriceFeed(SmartWebSocketV2):
    def __init__(self, obj, login_data, webhook_url=None, correlation_id="default"):
        auth_token = login_data["data"]["jwtToken"]
        feed_token = obj.getfeedToken()
        api_key = obj.api_key
        client_code = obj.userId
        super().__init__(auth_token, api_key, client_code, feed_token)
        self.price_dict = {}
        self.symbol_option_chains = {}
        self.last_update_time = None
        self.iv_log = defaultdict(lambda: defaultdict(dict))
        self.webhook_url = webhook_url
        self.index_option_chains_subscribed = []
        self.correlation_id = correlation_id
        self.finnifty_index = Index("FINNIFTY")  # Finnifty temp fix

    def start_websocket(self):
        def on_open(wsapp):
            self.subscribe_tokens()

        # Assign the callbacks.
        self.on_open = on_open
        self.on_data = self.on_data_handler
        self.on_error = lambda wsapp, error: print(error)
        self.on_close = lambda wsapp: print("Close")
        Thread(target=self.connect).start()

    def subscribe_tokens(self):
        tokens = ["26000", "26009"]
        mode = 1
        token_list = [{"exchangeType": 1, "tokens": tokens}]
        self.subscribe(self.correlation_id, mode, token_list)

    def on_data_handler(self, wsapp, message):
        self.price_dict[message["token"]] = {
            "ltp": message["last_traded_price"] / 100,
            "best_bid": message["best_5_sell_data"][0]["price"] / 100
            if "best_5_sell_data" in message
            else None,
            # 'best_5_sell_data' is not present in 'mode 1' messages
            "best_bid_qty": message["best_5_sell_data"][0]["quantity"]
            if "best_5_sell_data" in message
            else None,
            "best_ask": message["best_5_buy_data"][0]["price"] / 100
            if "best_5_buy_data" in message
            else None,
            "best_ask_qty": message["best_5_buy_data"][0]["quantity"]
            if "best_5_buy_data" in message
            else None,
            "timestamp": datetime.fromtimestamp(
                message["exchange_timestamp"] / 1000
            ).strftime("%H:%M:%S"),
            **message,
        }
        self.last_update_time = currenttime()

    def parse_price_dict(self):
        new_price_dict = {
            scrips.loc[scrips.token == token]["symbol"].values[0]: value
            for token, value in self.price_dict.items()
        }
        new_price_dict.update({"FINNIFTY": {"ltp": self.finnifty_index.fetch_ltp()}})  # Finnifty temp fix
        return new_price_dict

    def add_options(self, *underlyings, range_of_strikes=10, expiries=None, mode=1):
        """Adds options for the given underlyings to the symbol_option_chains dictionary.
        Params:
        underlyings is a list of underlying objects not strings
        If expiries is None, then the current, next and month expiry are added.
        If expiries is 'current', then only the current expiry is added.
        If expiries is a dictionary, then the expiries for each underlying are added.
        range_of_strikes is the number of strikes to be added on either side of the current price.
        mode is the mode of the websocket connection. 1 for LTP, 3 for full market depth.
        """

        def get_option_tokens(name, strike, expiry):
            _, c_token = fetch_symbol_token(name, expiry, strike, "CE")
            _, p_token = fetch_symbol_token(name, expiry, strike, "PE")
            return c_token, p_token

        for underlying in underlyings:
            if expiries is None:
                expiries_list = [
                    underlying.current_expiry,
                    underlying.next_expiry,
                    underlying.month_expiry,
                ]
            elif expiries == "current":
                expiries_list = [underlying.current_expiry]
            else:
                expiries_list = expiries[underlying.name]

            ltp = underlying.fetch_ltp()

            # Creating a OptionChains object for each index
            self.symbol_option_chains[underlying.name] = OptionChains()
            current_strike = findstrike(ltp, underlying.base)
            strike_range = np.arange(
                current_strike - (underlying.base * (range_of_strikes / 2)),
                current_strike + (underlying.base * (range_of_strikes / 2)),
                underlying.base,
            )
            strike_range = map(int, strike_range)
            data = []
            call_token_list, put_token_list = [], []
            for strike, expiry in list(itertools.product(strike_range, expiries_list)):
                try:
                    call_token, put_token = get_option_tokens(
                        underlying.name, strike, expiry
                    )
                    data.append((call_token, put_token))
                    call_token_list, put_token_list = zip(*data)
                    # Appending the expiry-strike pair to the container in OptionChains object
                    self.symbol_option_chains[underlying.name].exp_strike_pairs.append(
                        (expiry, strike)
                    )
                except Exception as e:
                    logger.error(
                        f"Error in fetching tokens for {strike, expiry} for {underlying.name}: {e}"
                    )
                    print(
                        f"Error in fetching tokens for {strike, expiry} for {underlying.name}: {e}"
                    )
                    call_token_list, put_token_list = ["abc"], ["abc"]
                    continue
            token_list = [
                {
                    "exchangeType": 2,
                    "tokens": list(call_token_list) + list(put_token_list),
                }
            ]
            self.subscribe(self.correlation_id, mode, token_list)
            self.index_option_chains_subscribed.append(underlying.name)
        sleep(3)

    def update_option_chain(
        self,
        sleep_time=5,
        exit_time=(15, 30),
        process_iv_log=True,
        market_depth=True,
        calculate_iv=True,
        stop_iv_calculation_hours=3,
        n_values=100,
        iv_threshold=1.1,
    ):
        while currenttime().time() < time(*exit_time):
            parsed_dict = self.parse_price_dict()
            indices = self.index_option_chains_subscribed
            for index in indices:
                expiries_subscribed = set(
                    [*zip(*self.symbol_option_chains[index].exp_strike_pairs)][0]
                )
                for expiry in expiries_subscribed:
                    self.build_option_chain(
                        index,
                        expiry,
                        parsed_dict,
                        market_depth,
                        process_iv_log,
                        calculate_iv,
                        n_values,
                        iv_threshold,
                        stop_iv_calculation_hours,
                    )

            sleep(sleep_time)

    def build_option_chain(
        self,
        index: str,
        expiry: str,
        parsed_dict: dict,
        market_depth,
        process_iv_log,
        calculate_iv,
        n_values,
        iv_threshold,
        stop_iv_calculation_hours=3,
    ):
        instrument_info = parsed_dict[index]
        spot = instrument_info["ltp"]

        for symbol, info in parsed_dict.items():
            if symbol.startswith(index) and "CE" in symbol and expiry in symbol:
                strike = float(parse_symbol(symbol)[2])
                put_symbol = symbol.replace("CE", "PE")
                put_option = parsed_dict[put_symbol]
                call_price = info["ltp"]
                put_price = put_option["ltp"]

                self.symbol_option_chains[index][expiry][strike][
                    "call_price"
                ] = call_price
                self.symbol_option_chains[index][expiry][strike][
                    "put_price"
                ] = put_price
                self.symbol_option_chains[index].underlying_price = spot

                if calculate_iv:
                    time_to_expiry = timetoexpiry(expiry)
                    if time_to_expiry < stop_iv_calculation_hours / (
                        24 * 365
                    ):  # If time to expiry is less than 3 hours stop calculating iv
                        continue
                    call_iv, put_iv, avg_iv = straddle_iv(
                        call_price, put_price, spot, strike, time_to_expiry
                    )
                    self.symbol_option_chains[index][expiry][strike][
                        "call_iv"
                    ] = call_iv
                    self.symbol_option_chains[index][expiry][strike]["put_iv"] = put_iv
                    self.symbol_option_chains[index][expiry][strike]["avg_iv"] = avg_iv

                    if process_iv_log:
                        self.process_iv_log(
                            index,
                            spot,
                            strike,
                            expiry,
                            call_iv,
                            put_iv,
                            avg_iv,
                            n_values,
                            iv_threshold,
                        )

                if market_depth:
                    self.symbol_option_chains[index][expiry][strike][
                        "call_best_bid"
                    ] = info["best_bid"]
                    self.symbol_option_chains[index][expiry][strike][
                        "call_best_ask"
                    ] = info["best_ask"]
                    self.symbol_option_chains[index][expiry][strike][
                        "put_best_bid"
                    ] = put_option["best_bid"]
                    self.symbol_option_chains[index][expiry][strike][
                        "put_best_ask"
                    ] = put_option["best_ask"]
                    self.symbol_option_chains[index][expiry][strike][
                        "call_best_bid_qty"
                    ] = info["best_bid_qty"]
                    self.symbol_option_chains[index][expiry][strike][
                        "call_best_ask_qty"
                    ] = info["best_ask_qty"]
                    self.symbol_option_chains[index][expiry][strike][
                        "put_best_bid_qty"
                    ] = put_option["best_bid_qty"]
                    self.symbol_option_chains[index][expiry][strike][
                        "put_best_ask_qty"
                    ] = put_option["best_ask_qty"]

    def process_iv_log(
        self,
        index,
        spot,
        strike,
        expiry,
        call_iv,
        put_iv,
        avg_iv,
        n_values,
        iv_threshold,
    ):
        if strike not in self.iv_log[index][expiry]:
            self.iv_log[index][expiry][strike] = {
                "call_ivs": [],
                "put_ivs": [],
                "total_ivs": [],
                "times": [],
                "count": 0,
                "last_notified_time": currenttime(),
            }

        self.iv_log[index][expiry][strike]["call_ivs"].append(call_iv)
        self.iv_log[index][expiry][strike]["put_ivs"].append(put_iv)
        self.iv_log[index][expiry][strike]["total_ivs"].append(avg_iv)
        self.iv_log[index][expiry][strike]["times"].append(currenttime().time())
        self.iv_log[index][expiry][strike]["count"] += 1

        call_ivs, put_ivs, total_ivs = self.get_recent_ivs(
            index, expiry, strike, n_values
        )

        running_avg_call_iv = sum(call_ivs) / len(call_ivs) if call_ivs else None
        running_avg_put_iv = sum(put_ivs) / len(put_ivs) if put_ivs else None
        running_avg_total_iv = sum(total_ivs) / len(total_ivs) if total_ivs else None

        self.check_and_notify_iv_spike(
            call_iv,
            running_avg_call_iv,
            "Call",
            index,
            spot,
            strike,
            expiry,
            iv_threshold,
        )
        self.check_and_notify_iv_spike(
            put_iv, running_avg_put_iv, "Put", index, spot, strike, expiry, iv_threshold
        )

        self.symbol_option_chains[index][expiry][strike].update(
            {
                "running_avg_call_iv": running_avg_call_iv,
                "running_avg_put_iv": running_avg_put_iv,
                "running_avg_total_iv": running_avg_total_iv,
            }
        )

    def get_recent_ivs(self, index, expiry, strike, n_values):
        call_ivs = self.iv_log[index][expiry][strike]["call_ivs"][-n_values:]
        put_ivs = self.iv_log[index][expiry][strike]["put_ivs"][-n_values:]
        total_ivs = self.iv_log[index][expiry][strike]["total_ivs"][-n_values:]
        call_ivs = [*filter(lambda x: x is not None, call_ivs)]
        put_ivs = [*filter(lambda x: x is not None, put_ivs)]
        total_ivs = [*filter(lambda x: x is not None, total_ivs)]
        return call_ivs, put_ivs, total_ivs

    def check_and_notify_iv_spike(
        self, iv, running_avg_iv, iv_type, idx, idx_price, K, exp, iv_hurdle
    ):
        not_in_the_money_by_100 = False

        if iv_type == "Call":
            not_in_the_money_by_100 = idx_price <= K - 100
        elif iv_type == "Put":
            not_in_the_money_by_100 = idx_price >= K + 100

        if (
            iv
            and iv > iv_hurdle * running_avg_iv
            and self.iv_log[idx][exp][K]["last_notified_time"]
            < currenttime() - timedelta(minutes=5)
            and not_in_the_money_by_100
        ):
            notifier(
                f"{iv_type} IV for {idx} {K} greater than average.\nIV: {iv}\n"
                f"Running Average: {running_avg_iv}",
                self.webhook_url,
            )
            self.iv_log[idx][exp][K]["last_notified_time"] = currenttime()


class SharedData:
    def __init__(self):
        self.position_data = None
        self.orderbook_data = None
        self.updated_time = None
        self.error_info = None
        self.force_stop = False

    def fetch_data(self):
        try:
            self.position_data = fetch_book("position")
            self.orderbook_data = fetch_book("orderbook")
            self.updated_time = currenttime()
        except Exception as e:
            self.position_data = None
            self.orderbook_data = None
            self.error_info = e

    def update_data(self, sleep_time=5, exit_time=(15, 30)):
        while currenttime().time() < time(*exit_time) and not self.force_stop:
            self.fetch_data()
            sleep(sleep_time)


class Option:
    def __init__(self, strike: int, option_type: str, underlying: str, expiry: str):
        self.strike = round(int(strike), 0)
        self.option_type = option_type
        self.underlying = underlying
        self.expiry = expiry
        self.symbol, self.token = fetch_symbol_token(
            underlying, expiry, strike, option_type
        )
        self.freeze_qty_in_shares = symbol_df[symbol_df["SYMBOL"] == self.underlying][
            "VOL_FRZ_QTY"
        ].values[0]
        self.lot_size = fetch_lot_size(self.underlying, expiry=self.expiry)
        self.freeze_qty_in_lots = int(self.freeze_qty_in_shares / self.lot_size)
        self.order_id_log = []

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(strike={self.strike}, option_type={self.option_type}, "
            f"underlying={self.underlying}, expiry={self.expiry})"
        )

    def __hash__(self):
        return hash((self.strike, self.option_type, self.underlying, self.expiry))

    def __eq__(self, other):
        if isinstance(other, (int, float, np.int32, np.int64, np.float32, np.float64)):
            return self.strike == other
        return self.strike == other.strike

    def __lt__(self, other):
        if isinstance(other, (int, float, np.int32, np.int64, np.float32, np.float64)):
            return self.strike < other
        return self.strike < other.strike

    def __gt__(self, other):
        if isinstance(other, (int, float, np.int32, np.int64, np.float32, np.float64)):
            return self.strike > other
        return self.strike > other.strike

    def __le__(self, other):
        if isinstance(other, (int, float, np.int32, np.int64, np.float32, np.float64)):
            return self.strike <= other
        return self.strike <= other.strike

    def __ge__(self, other):
        if isinstance(other, (int, float, np.int32, np.int64, np.float32, np.float64)):
            return self.strike >= other
        return self.strike >= other.strike

    def __ne__(self, other):
        if isinstance(other, (int, float, np.int32, np.int64, np.float32, np.float64)):
            return self.strike != other
        return self.strike != other.strike

    def __sub__(self, other):
        if isinstance(other, (int, float, np.int32, np.int64, np.float32, np.float64)):
            return self.strike - other
        return self.strike - other.strike

    def __add__(self, other):
        if isinstance(other, (int, float, np.int32, np.int64, np.float32, np.float64)):
            return self.strike + other
        return self.strike + other.strike

    def fetch_ltp(self):
        return fetchltp("NFO", self.symbol, self.token)

    def fetch_symbol_token(self):
        return self.symbol, self.token

    def place_order(self, transaction_type, quantity_in_lots, price="LIMIT", stop_loss_order=False, order_tag=""):
        if isinstance(price, str):
            if price.upper() == "LIMIT":
                price = self.fetch_ltp()
                modifier = 1.05 if transaction_type.upper() == "BUY" else 0.95
                price = price * modifier
        spliced_orders = splice_orders(quantity_in_lots, self.freeze_qty_in_lots)
        order_ids = []
        for qty in spliced_orders:
            order_id = place_order(
                self.symbol,
                self.token,
                qty * self.lot_size,
                transaction_type,
                price,
                stop_loss_order=stop_loss_order,
                order_tag=order_tag
            )
            order_ids.append(order_id)
        self.order_id_log.append(order_ids)
        return order_ids


class Strangle:
    def __init__(self, call_strike, put_strike, underlying, expiry):
        self.call_option = Option(call_strike, "CE", underlying, expiry)
        self.put_option = Option(put_strike, "PE", underlying, expiry)
        self.call_strike = self.call_option.strike
        self.put_strike = self.put_option.strike
        self.underlying = underlying
        self.underlying_exchange = "NFO" if self.underlying == "FINNIFTY" else "NSE"  # Finnifty temp fix
        self.expiry = expiry
        self.call_symbol, self.call_token = self.call_option.fetch_symbol_token()
        self.put_symbol, self.put_token = self.put_option.fetch_symbol_token()
        self.freeze_qty_in_shares = self.call_option.freeze_qty_in_shares
        self.freeze_qty_in_lots = self.call_option.freeze_qty_in_lots
        self.lot_size = self.call_option.lot_size

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(callstrike={self.call_option.strike}, putstrike={self.put_option.strike}, "
            f"underlying={self.underlying}, expiry={self.expiry})"
        )

    def __hash__(self):
        return hash((self.call_strike, self.put_strike, self.underlying, self.expiry))

    def fetch_ltp(self):
        return fetchltp("NFO", self.call_symbol, self.call_token), fetchltp(
            "NFO", self.put_symbol, self.put_token
        )

    def underlying_ltp(self):
        symbol, token = fetch_symbol_token(self.underlying)
        return fetchltp(self.underlying_exchange, symbol, token)

    def ivs(self):
        return strangle_iv(
            *self.fetch_ltp(),
            self.underlying_ltp(),
            self.call_strike,
            self.put_strike,
            timeleft=timetoexpiry(self.expiry),
        )

    def fetch_total_ltp(self):
        call_ltp, put_ltp = fetchltp(
            "NFO", self.call_symbol, self.call_token
        ), fetchltp("NFO", self.put_symbol, self.put_token)
        return call_ltp + put_ltp

    def price_disparity(self):
        call_ltp, put_ltp = self.fetch_ltp()
        disparity = abs(call_ltp - put_ltp)/min(call_ltp, put_ltp)
        return disparity

    def fetch_symbol_token(self):
        return self.call_symbol, self.call_token, self.put_symbol, self.put_token

    def place_order(self, transaction_type, quantity_in_lots, prices="LIMIT", stop_loss_order=False, order_tag=""):

        if stop_loss_order:
            assert isinstance(prices, (tuple, list, np.ndarray)), "Prices must be a tuple of prices for stop loss order"
            call_price, put_price = prices
        else:
            if isinstance(prices, (tuple, list, np.ndarray)):
                call_price, put_price = prices
            elif prices.upper() == "LIMIT":
                call_price, put_price = self.fetch_ltp()
                modifier = 1.05 if transaction_type.upper() == "BUY" else 0.95
                call_price, put_price = call_price * modifier, put_price * modifier
            elif prices.upper() == "MARKET":
                call_price = put_price = prices
            else:
                raise ValueError("Prices must be either 'LIMIT' or 'MARKET' or a tuple of prices")

        spliced_orders = splice_orders(quantity_in_lots, self.freeze_qty_in_lots)
        call_order_ids = []
        put_order_ids = []
        for qty in spliced_orders:
            call_order_id = place_order(
                self.call_symbol,
                self.call_token,
                qty * self.lot_size,
                transaction_type,
                call_price,
                stop_loss_order=stop_loss_order,
                order_tag=order_tag
            )
            put_order_id = place_order(
                self.put_symbol,
                self.put_token,
                qty * self.lot_size,
                transaction_type,
                put_price,
                stop_loss_order=stop_loss_order,
                order_tag=order_tag
            )
            call_order_ids.append(call_order_id)
            put_order_ids.append(put_order_id)
        return call_order_ids, put_order_ids


class Straddle(Strangle):
    def __init__(self, strike, underlying, expiry):
        super().__init__(strike, strike, underlying, expiry)


class SyntheticFuture(Strangle):
    def __init__(self, strike, underlying, expiry):
        super().__init__(strike, strike, underlying, expiry)

    def place_order(
            self, transaction_type, quantity_in_lots, prices: str | tuple = "LIMIT", stop_loss_order=False, order_tag=""
    ):
        if isinstance(prices, (tuple, list, np.ndarray)):
            call_price, put_price = prices
        elif prices.upper() == "LIMIT":
            call_price, put_price = self.fetch_ltp()
            c_modifier, p_modifier = (1.05, 0.95) if transaction_type.upper() == "BUY" else (0.95, 1.05)
            call_price, put_price = call_price * c_modifier, put_price * p_modifier
        elif prices.upper() == "MARKET":
            call_price = put_price = prices
        else:
            raise ValueError("Prices must be either 'LIMIT' or 'MARKET' or a tuple of prices")

        call_transaction_type = "BUY" if transaction_type.upper() == "BUY" else "SELL"
        put_transaction_type = "SELL" if transaction_type.upper() == "BUY" else "BUY"

        spliced_orders = splice_orders(quantity_in_lots, self.freeze_qty_in_lots)
        call_order_ids = []
        put_order_ids = []
        for qty in spliced_orders:
            call_order_id = place_order(
                self.call_symbol,
                self.call_token,
                qty * self.lot_size,
                call_transaction_type,
                call_price,
                order_tag=order_tag
            )
            put_order_id = place_order(
                self.put_symbol,
                self.put_token,
                qty * self.lot_size,
                put_transaction_type,
                put_price,
                order_tag=order_tag
            )
            call_order_ids.append(call_order_id)
            put_order_ids.append(put_order_id)
        return call_order_ids, put_order_ids


class SyntheticArbSystem:
    def __init__(self, symbol_option_chains):
        self.symbol_option_chains = symbol_option_chains
        self.index_expiry_pairs = {}
        self.successful_trades = 0
        self.unsuccessful_trades = 0

    def get_single_index_single_expiry_data(self, index, expiry):
        option_chain = self.symbol_option_chains[index][expiry]
        strikes = [strike for strike in option_chain]
        call_prices = [option_chain[strike]["call_price"] for strike in strikes]
        put_prices = [option_chain[strike]["put_price"] for strike in strikes]
        call_bids = [option_chain[strike]["call_best_bid"] for strike in strikes]
        call_asks = [option_chain[strike]["call_best_ask"] for strike in strikes]
        put_bids = [option_chain[strike]["put_best_bid"] for strike in strikes]
        put_asks = [option_chain[strike]["put_best_ask"] for strike in strikes]
        call_bid_qty = [option_chain[strike]["call_best_bid_qty"] for strike in strikes]
        call_ask_qty = [option_chain[strike]["call_best_ask_qty"] for strike in strikes]
        put_bid_qty = [option_chain[strike]["put_best_bid_qty"] for strike in strikes]
        put_ask_qty = [option_chain[strike]["put_best_ask_qty"] for strike in strikes]

        return (
            np.array(strikes),
            np.array(call_prices),
            np.array(put_prices),
            np.array(call_bids),
            np.array(call_asks),
            np.array(put_bids),
            np.array(put_asks),
            np.array(call_bid_qty),
            np.array(call_ask_qty),
            np.array(put_bid_qty),
            np.array(put_ask_qty),
        )

    def find_arbitrage_opportunities(
        self,
        index: str,
        expiry: str,
        qty_in_lots: int,
        exit_time=(15, 28),
        threshold=3,
    ):
        (
            strikes,
            call_prices,
            put_prices,
            call_bids,
            call_asks,
            put_bids,
            put_asks,
            call_bid_qty,
            call_ask_qty,
            put_bid_qty,
            put_ask_qty,
        ) = self.get_single_index_single_expiry_data(index, expiry)
        synthetic_buy_prices = strikes + call_asks - put_bids
        synthetic_sell_prices = strikes + call_bids - put_asks
        min_price_index = np.argmin(synthetic_buy_prices)
        max_price_index = np.argmax(synthetic_sell_prices)
        min_price = synthetic_buy_prices[min_price_index]
        max_price = synthetic_sell_prices[max_price_index]

        last_print_time = currenttime()
        while currenttime().time() < time(*exit_time):
            # print(strikes, call_prices, put_prices, synthetic_prices)
            if currenttime() > last_print_time + timedelta(seconds=5):
                print(
                    f"{currenttime()} - {index} - {expiry}:\n"
                    f"Minimum price: {min_price} at strike: {strikes[min_price_index]} Call Ask: {call_asks[min_price_index]} Put Bid: {put_bids[min_price_index]}\n"
                    f"Maximum price: {max_price} at strike: {strikes[max_price_index]} Call Bid: {call_bids[max_price_index]} Put Ask: {put_asks[max_price_index]}\n"
                    f"Price difference: {max_price - min_price}\n"
                )
                last_print_time = currenttime()

            if max_price - min_price > threshold:
                print(
                    f"**********Trade Identified at {currenttime()} on strike: Min {strikes[min_price_index]} "
                    f"and Max {strikes[max_price_index]}**********\n"
                    f"Minimum price: {min_price} at strike: {strikes[min_price_index]} Call Ask: {call_asks[min_price_index]} Put Bid: {put_bids[min_price_index]}\n"
                    f"Maximum price: {max_price} at strike: {strikes[max_price_index]} Call Bid: {call_bids[max_price_index]} Put Ask: {put_asks[max_price_index]}\n"
                    f"Price difference: {max_price - min_price}\n"
                )
                min_strike = strikes[min_price_index]
                max_strike = strikes[max_price_index]

                self.execute_synthetic_trade(
                    index,
                    expiry,
                    qty_in_lots,
                    min_strike,
                    max_strike,
                    sleep_interval=5,
                )

            for i, strike in enumerate(strikes):
                call_prices[i] = self.symbol_option_chains[index][expiry][strike][
                    "call_price"
                ]
                put_prices[i] = self.symbol_option_chains[index][expiry][strike][
                    "put_price"
                ]
                call_bids[i] = self.symbol_option_chains[index][expiry][strike][
                    "call_best_bid"
                ]
                call_asks[i] = self.symbol_option_chains[index][expiry][strike][
                    "call_best_ask"
                ]
                put_bids[i] = self.symbol_option_chains[index][expiry][strike][
                    "put_best_bid"
                ]
                put_asks[i] = self.symbol_option_chains[index][expiry][strike][
                    "put_best_ask"
                ]
                call_bid_qty[i] = self.symbol_option_chains[index][expiry][strike][
                    "call_best_bid_qty"
                ]
                call_ask_qty[i] = self.symbol_option_chains[index][expiry][strike][
                    "call_best_ask_qty"
                ]
                put_bid_qty[i] = self.symbol_option_chains[index][expiry][strike][
                    "put_best_bid_qty"
                ]
                put_ask_qty[i] = self.symbol_option_chains[index][expiry][strike][
                    "put_best_ask_qty"
                ]
            synthetic_buy_prices = strikes + call_asks - put_bids
            synthetic_sell_prices = strikes + call_bids - put_asks
            min_price_index = np.argmin(synthetic_buy_prices)
            max_price_index = np.argmax(synthetic_sell_prices)
            min_price = synthetic_buy_prices[min_price_index]
            max_price = synthetic_sell_prices[max_price_index]

    @staticmethod
    def execute_synthetic_trade(
        index,
        expiry,
        qty_in_lots,
        buy_strike,
        sell_strike,
        sleep_interval=1,
    ):
        ids_call_buy, ids_put_sell = place_synthetic_fut_order(
            index, buy_strike, expiry, "BUY", qty_in_lots, 'MARKET'
        )
        ids_call_sell, ids_put_buy = place_synthetic_fut_order(
            index, sell_strike, expiry, "SELL", qty_in_lots, 'MARKET'
        )
        ids = np.concatenate((ids_call_buy, ids_put_sell, ids_call_sell, ids_put_buy))

        sleep(sleep_interval)
        statuses = lookup_and_return("orderbook", "orderid", ids, "status")

        if any(statuses == "rejected"):
            logger.error(
                f"Order rejected for {index} {expiry} {qty_in_lots} Buy {buy_strike} Sell {sell_strike}"
            )


class Index:
    """Initialize an index with the name of the index in uppercase"""

    def __init__(self, name, webhook_url=None, websocket=None, spot_future_rate=0.06):
        if name not in symbol_df["SYMBOL"].values:
            closest_match, confidence = process.extractOne(
                name, symbol_df["SYMBOL"].values
            )
            if confidence > 80:
                raise Exception(
                    f"Index {name} not found. Did you mean {closest_match}?"
                )

            else:
                raise ValueError(f"Index {name} not found")

        self.name = name
        self.ltp = None
        self.previous_close = None
        self.current_expiry = None
        self.next_expiry = None
        self.month_expiry = None
        self.fut_expiry = None
        self.order_log = defaultdict(list)
        self.webhook_url = webhook_url
        self.spot_future_rate = spot_future_rate
        self.symbol, self.token = fetch_symbol_token(self.name)
        self.lot_size = fetch_lot_size(self.name)
        self.fetch_expirys(self.symbol)
        self.freeze_qty = self.fetch_freeze_limit()
        self.available_strikes = None
        self.available_straddle_strikes = None
        self.intraday_straddle_forced_exit = False

        if self.name == "BANKNIFTY":
            self.base = 100
            self.exchange_type = 1
        elif self.name == "NIFTY":
            self.base = 50
            self.exchange_type = 1
        elif self.name == "FINNIFTY":  # Finnifty temp fix
            self.base = 50
            self.exchange_type = 2
        else:
            self.base = get_base(self.name)
            self.exchange_type = 1
            logger.info(f"Base for {self.name} is {self.base}")
            # print(f"Base for {self.name} is {self.base}")

        if websocket:
            try:
                websocket.subscribe(
                    websocket.correlation_id,
                    1,
                    [{"exchangeType": self.exchange_type, "tokens": [self.token]}],
                )
                sleep(2)
                print(f"{self.name}: Subscribed underlying to the websocket")
            except Exception as e:
                print(f"{self.name}: Websocket subscription failed. {e}")

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(Name: {self.name}, Lot Size: {self.lot_size}, "
            f"Freeze Qty: {self.freeze_qty}, Current Expiry: {self.current_expiry}, Symbol: {self.symbol}, "
            f"Token: {self.token})"
        )

    def fetch_freeze_limit(self):
        try:
            freeze_qty_url = "https://archives.nseindia.com/content/fo/qtyfreeze.xls"
            response = requests.get(freeze_qty_url, timeout=10)  # Set the timeout value
            response.raise_for_status()  # Raise an exception if the response contains an HTTP error status
            df = pd.read_excel(response.content)
            df.columns = df.columns.str.strip()
            df["SYMBOL"] = df["SYMBOL"].str.strip()
            freeze_qty = df[df["SYMBOL"] == self.name]["VOL_FRZ_QTY"].values[0]
            freeze_qty_in_lots = freeze_qty / self.lot_size
            return int(freeze_qty_in_lots)
        except requests.exceptions.Timeout as e:
            notifier(
                f"Timeout error in fetching freeze limit for {self.name}: {e}",
                self.webhook_url,
            )
            freeze_qty_in_lots = 20
            return int(freeze_qty_in_lots)
        except requests.exceptions.HTTPError as e:
            notifier(
                f"HTTP error in fetching freeze limit for {self.name}: {e}",
                self.webhook_url,
            )
            freeze_qty_in_lots = 20
            return int(freeze_qty_in_lots)
        except Exception as e:
            notifier(
                f"Error in fetching freeze limit for {self.name}: {e}", self.webhook_url
            )
            freeze_qty_in_lots = 20
            return freeze_qty_in_lots

    def fetch_expirys(self, symbol: str):
        expirymask = (
            (scrips.expiry != "")
            & (scrips.exch_seg == "NFO")
            & (scrips.name == self.name)
        )
        expirydates = (
            pd.to_datetime(scrips[expirymask].expiry, format="%d%b%Y")
            .astype("datetime64[ns]")
            .sort_values()
            .unique()
        )
        allexpiries = [
            (pd.to_datetime(exps) + pd.DateOffset(minutes=930))
            .strftime("%d%b%y")
            .upper()
            for exps in expirydates
            if (pd.to_datetime(exps) + pd.DateOffset(minutes=930)) > currenttime()
        ]

        if symbol.endswith("EQ"):
            self.current_expiry = allexpiries[0]
            self.next_expiry = allexpiries[1]
            self.month_expiry = allexpiries[2]
            self.fut_expiry = allexpiries[0]
            return

        expiry_month_list = [
            int(datetime.strptime(i[2:5], "%b").strftime("%m")) for i in allexpiries
        ]
        monthmask = np.where(np.diff(expiry_month_list) == 0, 0, 1)
        monthexpiries = [b for a, b in zip(monthmask, allexpiries) if a == 1]

        currentexpiry = allexpiries[0]
        nextexpiry = allexpiries[1]
        monthexpiry = (
            monthexpiries[1] if monthexpiries[0] == allexpiries[0] else monthexpiries[0]
        )
        futexpiry = (
            allexpiries[0] if monthexpiries[0] == allexpiries[0] else monthexpiries[0]
        )

        self.current_expiry = currentexpiry
        self.next_expiry = nextexpiry
        self.month_expiry = monthexpiry
        self.fut_expiry = futexpiry

    def fetch_ltp(self):
        """Fetch LTP of the index. Uses futures for FINNIFTY"""
        if self.name == "FINNIFTY":  # Finnifty temp fix
            ltp = fetchltp("NFO", self.symbol, self.token)
            self.ltp = spot_price_from_future(
                ltp, self.spot_future_rate, timetoexpiry(self.fut_expiry)
            )
        else:
            self.ltp = fetchltp("NSE", self.symbol, self.token)
        return self.ltp

    def fetch_previous_close(self):
        self.previous_close = fetchpreviousclose("NSE", self.symbol, self.token)
        return self.previous_close

    def fetch_atm_info(self, expiry="current"):
        expiry_dict = {
            "current": self.current_expiry,
            "next": self.next_expiry,
            "month": self.month_expiry,
            "future": self.fut_expiry,
            "fut": self.fut_expiry,
        }
        expiry = expiry_dict[expiry]
        price = self.fetch_ltp()
        atm_strike = findstrike(price, self.base)
        atm_straddle = Straddle(atm_strike, self.name, expiry)
        call_price, put_price = atm_straddle.fetch_ltp()
        total_price = call_price + put_price
        call_iv, put_iv, avg_iv = atm_straddle.ivs()
        return {
            "underlying_price": price,
            "strike": atm_strike,
            "call_price": call_price,
            "put_price": put_price,
            "total_price": total_price,
            "call_iv": call_iv,
            "put_iv": put_iv,
            "avg_iv": avg_iv,
        }

    def fetch_otm_info(self, strike_offset, expiry="current"):
        expiry_dict = {
            "current": self.current_expiry,
            "next": self.next_expiry,
            "month": self.month_expiry,
            "future": self.fut_expiry,
            "fut": self.fut_expiry,
        }
        expiry = expiry_dict[expiry]
        price = self.fetch_ltp()
        call_strike = price * (1 + strike_offset)
        put_strike = price * (1 - strike_offset)
        call_strike = findstrike(call_strike, self.base)
        put_strike = findstrike(put_strike, self.base)
        otm_strangle = Strangle(call_strike, put_strike, self.name, expiry)
        call_price, put_price = otm_strangle.fetch_ltp()
        total_price = call_price + put_price
        call_iv, put_iv, avg_iv = otm_strangle.ivs()
        return {
            "underlying_price": price,
            "call_strike": call_strike,
            "put_strike": put_strike,
            "call_price": call_price,
            "put_price": put_price,
            "total_price": total_price,
            "call_iv": call_iv,
            "put_iv": put_iv,
            "avg_iv": avg_iv,
        }

    def get_available_strikes(self, both_pairs=False):
        available_strikes = get_available_strikes(self.name, both_pairs)
        if not both_pairs:
            self.available_strikes = available_strikes
        else:
            self.available_straddle_strikes = available_strikes
        return available_strikes

    def get_constituents(self, cutoff_pct=101):
        tickers, weights = get_index_constituents(self.name, cutoff_pct)
        return tickers, weights

    def log_combined_order(
            self, strike=None,
            call_strike=None,
            put_strike=None,
            expiry=None,
            buy_or_sell=None,
            call_price=None,
            put_price=None,
            order_tag=None):

        if strike is None and (call_strike is None or put_strike is None):
            raise Exception("Strike and call/put strike both not provided")

        if strike is not None and (call_strike is not None or put_strike is not None):
            raise Exception("Strike and call/put strike both provided")

        if strike is not None:
            call_strike = strike
            put_strike = strike

        dict_format = {
            "Date": currenttime().strftime("%d-%m-%Y %H:%M:%S"),
            "Index": self.name,
            "Call Strike": call_strike,
            "Put Strike": put_strike,
            "Expiry": expiry,
            "Action Type": buy_or_sell,
            "Call Price": call_price,
            "Put Price": put_price,
            "Total Price": call_price + put_price,
            "Tag": order_tag,
        }
        self.order_log[order_tag].append(dict_format)

    def splice_orders(self, quantity_in_lots):
        if quantity_in_lots > self.freeze_qty:
            loops = int(quantity_in_lots / self.freeze_qty)
            if loops > LARGE_ORDER_THRESHOLD:
                raise Exception(
                    "Order too big. This error was raised to prevent accidental large order placement."
                )

            remainder = quantity_in_lots % self.freeze_qty
            if remainder == 0:
                spliced_orders = [self.freeze_qty] * loops
            else:
                spliced_orders = [self.freeze_qty] * loops + [remainder]
        else:
            spliced_orders = [quantity_in_lots]
        return spliced_orders

    def place_combined_order(
        self,
        expiry,
        buy_or_sell,
        quantity_in_lots,
        strike=None,
        call_strike=None,
        put_strike=None,
        return_avg_price=False,
        order_tag="",
    ):
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
                raise ValueError("Strike price not specified")
            strike_info = f"{call_strike}CE {put_strike}PE"
        elif call_strike is None and put_strike is None:
            call_strike = strike
            put_strike = strike
            strike_info = f"{strike}"
        else:
            raise ValueError("Strike price specified twice")

        call_symbol, call_token = fetch_symbol_token(
            self.name, expiry, call_strike, "CE"
        )
        put_symbol, put_token = fetch_symbol_token(self.name, expiry, put_strike, "PE")
        call_price = fetchltp("NFO", call_symbol, call_token)
        put_price = fetchltp("NFO", put_symbol, put_token)

        limit_price_extender = 1.05 if buy_or_sell == "BUY" else 0.95

        spliced_orders = self.splice_orders(quantity_in_lots)

        call_order_id_list = []
        put_order_id_list = []
        for quantity in spliced_orders:
            call_order_id = place_order(
                call_symbol,
                call_token,
                quantity * self.lot_size,
                buy_or_sell,
                call_price * limit_price_extender,
                order_tag=order_tag,
            )
            put_order_id = place_order(
                put_symbol,
                put_token,
                quantity * self.lot_size,
                buy_or_sell,
                put_price * limit_price_extender,
                order_tag=order_tag,
            )
            call_order_id_list.append(call_order_id)
            put_order_id_list.append(put_order_id)
            sleep(0.3)

        orderbook = fetch_book("orderbook")

        call_order_statuses = lookup_and_return(
            orderbook, "orderid", call_order_id_list, "status"
        )
        put_order_statuses = lookup_and_return(
            orderbook, "orderid", put_order_id_list, "status"
        )

        order_prefix = f"{order_tag}: " if order_tag else ""

        if all(call_order_statuses == "complete") and all(
            put_order_statuses == "complete"
        ):
            notifier(
                f"{order_prefix}Order(s) placed successfully for {buy_or_sell} {self.name} "
                + f"{strike_info} {expiry} {quantity_in_lots} lot(s).",
                self.webhook_url,
            )
            call_order_avg_price = (
                lookup_and_return(
                    orderbook, "orderid", call_order_id_list, "averageprice"
                )
                .astype(float)
                .mean()
            )
            put_order_avg_price = (
                lookup_and_return(
                    orderbook, "orderid", put_order_id_list, "averageprice"
                )
                .astype(float)
                .mean()
            )
            if return_avg_price:
                return call_order_avg_price, put_order_avg_price
            else:
                return
        elif all(call_order_statuses == "rejected") and all(
            put_order_statuses == "rejected"
        ):
            notifier(
                f"{order_prefix}All orders rejected for {buy_or_sell} {self.name} "
                + f"{strike_info} {expiry} {quantity_in_lots} lot(s).",
                self.webhook_url,
            )
            raise Exception("Orders rejected")
        elif any(call_order_statuses == "open") or any(put_order_statuses == "open"):
            notifier(
                f"{order_prefix}Some orders pending for {buy_or_sell} {self.name} "
                + f"{strike_info} {expiry} {quantity_in_lots} lot(s). You can modify the orders.",
                self.webhook_url,
            )
        elif any(call_order_statuses == "rejected") or any(
            put_order_statuses == "rejected"
        ):
            notifier(
                f"{order_prefix}Some orders rejected for {buy_or_sell} {self.name} "
                + f"{strike_info} {expiry} {quantity_in_lots} lot(s). You can place the rejected orders again.",
                self.webhook_url,
            )
        else:
            notifier(
                f"{order_prefix}ERROR. Order statuses uncertain for {buy_or_sell} {self.name} "
                + f"{strike_info} {expiry} {quantity_in_lots} lot(s).",
                self.webhook_url,
            )
            raise Exception("Order statuses uncertain")

    def place_synthetic_fut(
            self, strike, expiry, buy_or_sell, quantity_in_lots, prices="LIMIT", stop_loss_order=False, order_tag=""
    ):
        return place_synthetic_fut_order(
            self.name, strike, expiry, buy_or_sell, quantity_in_lots, prices, stop_loss_order, order_tag
        )

    def find_equal_strike(
        self, exit_time, websocket, wait_for_equality, target_disparity, expiry=None
    ):
        expiry = expiry or self.current_expiry
        ltp = (
            self.fetch_ltp()
            if not websocket
            else websocket.price_dict.get(self.token, 0)["ltp"]
        )
        current_strike = findstrike(ltp, self.base)
        strike_range = np.arange(
            current_strike - self.base * 2, current_strike + self.base * 2, self.base
        )

        def fetch_data(strike, exp):
            c_symbol, c_token = fetch_symbol_token(self.name, exp, strike, "CE")
            p_symbol, p_token = fetch_symbol_token(self.name, exp, strike, "PE")
            return c_symbol, c_token, p_symbol, p_token

        def fetch_ltps(tokens, symbols, socket):
            if socket:
                return np.array(
                    [
                        websocket.price_dict.get(token, {"ltp": 0})["ltp"]
                        for token in tokens
                    ]
                )
            else:
                return np.array(
                    [
                        fetchltp("NFO", symbol, token)
                        for symbol, token in zip(symbols, tokens)
                    ]
                )

        def compute_disparities(c_ltps, p_ltps):
            return np.abs(c_ltps - p_ltps) / np.minimum(c_ltps, p_ltps) * 100

        data = [fetch_data(strike, expiry) for strike in strike_range]
        call_token_list, put_token_list = zip(*(tokens[1:4:2] for tokens in data))
        call_symbol_list, put_symbol_list = zip(*(symbols[0:3:2] for symbols in data))

        if websocket:
            websocket.subscribe(
                websocket.correlation_id,
                1,
                [
                    {
                        "exchangeType": 2,
                        "tokens": list(call_token_list) + list(put_token_list),
                    }
                ],
            )
            sleep(3)

        call_ltps, put_ltps = fetch_ltps(
            call_token_list, call_symbol_list, websocket
        ), fetch_ltps(put_token_list, put_symbol_list, websocket)
        disparities = compute_disparities(call_ltps, put_ltps)

        if wait_for_equality:

            last_print_time = currenttime()
            last_log_time = currenttime()
            last_notify_time = currenttime()
            print_interval = timedelta(seconds=0.05)
            log_interval = timedelta(minutes=1)
            notify_interval = timedelta(minutes=2)

            min_disparity_idx = np.argmin(disparities)
            min_disparity = disparities[min_disparity_idx]

            while min_disparity > target_disparity:
                if min_disparity < 10:
                    # Update only the minimum disparity strike data
                    call_ltp, put_ltp = fetch_ltps(
                        [call_token_list[min_disparity_idx]],
                        call_symbol_list[min_disparity_idx],
                        websocket,
                    ), fetch_ltps(
                        [put_token_list[min_disparity_idx]],
                        put_symbol_list[min_disparity_idx],
                        websocket,
                    )
                    disparities[min_disparity_idx] = compute_disparities(
                        call_ltp, put_ltp
                    )
                    single_check = True
                else:
                    # Update all strike data
                    call_ltps, put_ltps = fetch_ltps(
                        call_token_list, call_symbol_list, websocket
                    ), fetch_ltps(put_token_list, put_symbol_list, websocket)
                    disparities = compute_disparities(call_ltps, put_ltps)
                    single_check = False

                min_disparity_idx = np.argmin(disparities)
                min_disparity = disparities[min_disparity_idx]
                message = (
                    f'Time: {currenttime().strftime("%H:%M:%S")}\n'
                    + f"Index: {self.name}\n"
                    + f"Current lowest disparity: {min_disparity:.2f}\n"
                    + f"Strike: {strike_range[min_disparity_idx]}\n"
                    + f"Single Strike: {single_check}\n"
                )
                if currenttime() - last_print_time > print_interval:
                    print(message)
                    last_print_time = currenttime()
                if currenttime() - last_log_time > log_interval:
                    logger.info(message)
                    last_log_time = currenttime()
                if currenttime() - last_notify_time > notify_interval:
                    notifier(message, self.webhook_url)
                    last_notify_time = currenttime()

                if (currenttime() + timedelta(minutes=5)).time() > time(*exit_time):
                    notifier(
                        "Equal strike tracker exited due to time limit.",
                        self.webhook_url,
                    )
                    raise Exception("Equal strike tracker exited due to time limit.")

        idx = np.argmin(disparities)
        (
            strike_to_trade,
            call_symbol,
            call_token,
            put_symbol,
            put_token,
            call_ltp,
            put_ltp,
        ) = (
            strike_range[idx],
            call_symbol_list[idx],
            call_token_list[idx],
            put_symbol_list[idx],
            put_token_list[idx],
            call_ltps[idx],
            put_ltps[idx],
        )

        return (
            strike_to_trade,
            call_symbol,
            call_token,
            put_symbol,
            put_token,
            call_ltp,
            put_ltp,
        )

    @log_errors
    def rollover_overnight_short_straddle(
        self, quantity_in_lots, strike_offset=1, iv_threshold=0.95, take_avg_price=False
    ):
        """Rollover overnight short straddle to the next expiry.
        Args:
            quantity_in_lots (int): Quantity of the straddle in lots.
            strike_offset (float): Strike offset from the current strike.
            iv_threshold (float): IV threshold compared to vix.
            take_avg_price (bool): Take average price of the index over 5m timeframes.
        """

        def load_data():
            try:
                with open(f"{obj.userId}_overnight_positions.json", "r") as f:
                    data = json.load(f)
                    return data
            except FileNotFoundError:
                data = {}
                notifier(
                    "No positions found for overnight straddle. Creating new file.", self.webhook_url
                )
                with open(f"{obj.userId}_overnight_positions.json", "w") as f:
                    json.dump(data, f)
                return data
            except Exception as e:
                notifier(f"Error while reading overnight_positions.json: {e}", self.webhook_url)
                logger.error(f"Error while reading positions.json", exc_info=(type(e), e, e.__traceback__))
                raise Exception("Error while reading positions.json")

        def save_data(data):
            with open(f"{obj.userId}_overnight_positions.json", "w") as f:
                json.dump(data, f)

        avg_ltp = None
        if take_avg_price:
            if currenttime().time() < time(15, 00):
                notifier(
                    f"{self.name} Cannot take avg price before 3pm. Try running the strategy after 3pm",
                    self.webhook_url,
                )
                raise Exception(
                    "Cannot take avg price before 3pm. Try running the strategy after 3pm"
                )
            notifier(
                f"{self.name} Taking average price of the index over 5m timeframes.",
                self.webhook_url,
            )
            price_list = [self.fetch_ltp()]
            while currenttime().time() < time(15, 28):
                _ltp = self.fetch_ltp()
                price_list.append(_ltp)
                sleep(60)
            avg_ltp = np.mean(price_list)

        # Assigning vix
        vix = yf.Ticker("^INDIAVIX")
        vix = vix.fast_info["last_price"]
        if self.name in ["FINNIFTY", "BANKNIFTY"]:
            beta = dm.get_summary_ratio(self.name, 'NIFTY')  # beta of the index vs nifty since vix is of nifty
            beta = 1.3 if beta is None else beta
            vix = vix * beta

        order_tag = "Overnight Short Straddle"

        weekend_in_expiry = check_for_weekend(self.current_expiry)
        ltp = avg_ltp if avg_ltp else self.fetch_ltp()
        sell_strike = findstrike(ltp * strike_offset, self.base)
        call_ltp, put_ltp = fetch_straddle_price(
            self.name, self.current_expiry, sell_strike
        )
        call_iv, put_iv, iv = straddle_iv(
            call_ltp, put_ltp, ltp, sell_strike, timetoexpiry(self.current_expiry)
        )
        iv = iv * 100

        # This if-clause checks how far the expiry is
        if weekend_in_expiry:  # far from expiry

            if iv < vix * iv_threshold:
                notifier(
                    f"{self.name} IV is too low compared to VIX - IV: {iv}, Vix: {vix}.",
                    self.webhook_url,
                )
                return
            else:
                notifier(
                    f"{self.name} IV is fine compared to VIX - IV: {iv}, Vix: {vix}.", self.webhook_url
                )
        elif (
            timetoexpiry(self.current_expiry, effective_time=True, in_days=True) < 2
        ):  # only exit
            sell_strike = None
            notifier(f"{self.name} Only exiting current position. IV: {iv}, Vix: {vix}.", self.webhook_url)
        else:
            notifier(
                f"{self.name} Rolling over overnight straddle - IV: {iv}, Vix: {vix}.", self.webhook_url
            )

        trade_data = load_data()
        buy_strike = trade_data.get(self.name, None)

        # Checking if the buy strike is valid
        if (
            not isinstance(buy_strike, int)
            and not isinstance(buy_strike, float)
            and buy_strike is not None
        ):
            notifier(f"Invalid strike found for {self.name}.", self.webhook_url)
            raise Exception(f"Invalid strike found for {self.name}.")

        # Placing orders
        if buy_strike is None and sell_strike is None:
            notifier(f"{self.name} No trade required.", self.webhook_url)
        elif sell_strike is None:  # only exiting current position
            notifier(
                f"{self.name} Exiting current position on strike {buy_strike}.", self.webhook_url
            )
            call_buy_avg, put_buy_avg = self.place_combined_order(
                self.current_expiry,
                "BUY",
                quantity_in_lots,
                strike=buy_strike,
                return_avg_price=True,
                order_tag=order_tag,
            )
            self.log_combined_order(buy_strike, expiry=self.current_expiry, buy_or_sell="BUY", call_price=call_buy_avg,
                                    put_price=put_buy_avg, order_tag=order_tag)
        elif buy_strike is None:  # only entering new position
            notifier(
                f"{self.name} Entering new position on strike {sell_strike}.", self.webhook_url
            )
            call_sell_avg, put_sell_avg = self.place_combined_order(
                self.current_expiry,
                "SELL",
                quantity_in_lots,
                strike=sell_strike,
                return_avg_price=True,
                order_tag=order_tag,
            )
            self.log_combined_order(sell_strike, expiry=self.current_expiry, buy_or_sell="SELL",
                                    call_price=call_sell_avg, put_price=put_sell_avg, order_tag=order_tag)
        else:  # both entering and exiting positions
            if buy_strike == sell_strike:
                notifier(f"{self.name} No trade required as strike is same.", self.webhook_url)
                call_ltp, put_ltp = fetch_straddle_price(
                    self.name, self.current_expiry, sell_strike
                )
                self.log_combined_order(buy_strike, expiry=self.current_expiry, buy_or_sell="BUY", call_price=call_ltp,
                                        put_price=put_ltp, order_tag=order_tag)
                self.log_combined_order(sell_strike, expiry=self.current_expiry, buy_or_sell="SELL",
                                        call_price=call_ltp, put_price=put_ltp, order_tag=order_tag)
            else:
                notifier(
                    f"{self.name} Buying {buy_strike} and selling {sell_strike}.", self.webhook_url
                )
                call_buy_avg, put_buy_avg = self.place_combined_order(
                    self.current_expiry,
                    "BUY",
                    quantity_in_lots,
                    strike=buy_strike,
                    return_avg_price=True,
                    order_tag=order_tag,
                )
                call_sell_avg, put_sell_avg = self.place_combined_order(
                    self.current_expiry,
                    "SELL",
                    quantity_in_lots,
                    strike=sell_strike,
                    return_avg_price=True,
                    order_tag=order_tag,
                )
                self.log_combined_order(buy_strike, expiry=self.current_expiry, buy_or_sell="BUY",
                                        call_price=call_buy_avg, put_price=put_buy_avg, order_tag=order_tag)
                self.log_combined_order(sell_strike, expiry=self.current_expiry, buy_or_sell="SELL",
                                        call_price=call_sell_avg, put_price=put_sell_avg, order_tag=order_tag)

        trade_data[self.name] = sell_strike
        save_data(trade_data)

    @log_errors
    def buy_weekly_hedge(
        self,
        quantity_in_lots,
        type_of_hedge="strangle",
        strike_offset=1,
        call_offset=1,
        put_offset=1,
    ):
        ltp = self.fetch_ltp()
        if type_of_hedge == "strangle":
            call_strike = findstrike(ltp * call_offset, self.base)
            put_strike = findstrike(ltp * put_offset, self.base)
            strike = None
        elif type_of_hedge == "straddle":
            strike = findstrike(ltp * strike_offset, self.base)
            call_strike = None
            put_strike = None
        else:
            raise Exception("Invalid type of hedge.")

        call_buy_avg, put_buy_avg = self.place_combined_order(
            self.next_expiry,
            "BUY",
            quantity_in_lots,
            strike=strike,
            call_strike=call_strike,
            put_strike=put_strike,
            order_tag="Weekly Hedge",
            return_avg_price=True,
        )

        self.log_combined_order(strike=strike, call_strike=call_strike, put_strike=put_strike, expiry=self.next_expiry,
                                buy_or_sell="BUY", call_price=call_buy_avg, put_price=put_buy_avg,
                                order_tag="Weekly Hedge")

    @log_errors
    def intraday_straddle(
        self,
        quantity_in_lots,
        exit_time=(15, 28),
        websocket=None,
        wait_for_equality=False,
        move_sl=False,
        shared_data=None,
        stoploss="dynamic",
        target_disparity=10,
        catch_trend=False,
        trend_qty_ratio=0.5,
        trend_catcher_sl=0.003,
        safeguard=False,
        safeguard_movement=0.0035,
        safeguard_spike=1.2,
        smart_exit=False,
        take_profit=False,
        take_profit_points=np.inf,
        convert_to_butterfly=False,
        check_force_exit=False,
    ):
        """Params:
        quantity_in_lots: int
        exit_time: tuple
        websocket: websocket object
        wait_for_equality: bool
        move_sl: bool
        shared_data: class object
        stoploss: str
        target_disparity: float
        catch_trend: bool
        trend_qty_ratio: float
        trend_catcher_sl: float
        safeguard: bool
        safeguard_movement: float
        safeguard_spike: float
        smart_exit: bool
        take_profit: bool
        take_profit_points: float
        take_profit_order: bool
        """

        order_tag = "Intraday straddle"
        strategy_id = currenttime().strftime("%d%m%y%H%M%S%f")
        expiry = self.current_expiry
        sleep_interval = 0 if take_profit else 5

        # Splicing orders
        spliced_orders = self.splice_orders(quantity_in_lots)

        # Finding equal strike
        (
            equal_strike,
            call_symbol,
            call_token,
            put_symbol,
            put_token,
            call_price,
            put_price,
        ) = self.find_equal_strike(
            exit_time=exit_time,
            websocket=websocket,
            wait_for_equality=wait_for_equality,
            target_disparity=target_disparity,
            expiry=expiry,
        )

        notifier(
            f"{self.name}: Initiating intraday trade on {equal_strike} strike.",
            self.webhook_url,
        )

        # Placing orders
        call_avg_price, put_avg_price = self.place_combined_order(
            expiry,
            "SELL",
            quantity_in_lots,
            strike=equal_strike,
            return_avg_price=True,
            order_tag=order_tag,
        )

        underlying_price = self.fetch_ltp()
        entry_spot = underlying_price

        # Placing stoploss orders
        if stoploss == "fixed":
            if self.name == "BANKNIFTY":
                sl = 1.7
            elif self.name == "NIFTY":
                sl = 1.5
            else:
                sl = 1.6
        elif stoploss == "dynamic":
            if self.name == "BANKNIFTY" or timetoexpiry(expiry, in_days=True) < 1:
                sl = 1.7
            elif self.name == "NIFTY":
                sl = 1.5
            else:
                sl = 1.6
        else:
            sl = stoploss

        call_stoploss_order_ids = []
        put_stoploss_order_ids = []
        stoploss_tag = f"{self.name} {strategy_id} stoploss"
        for quantity in spliced_orders:
            call_sl_order_id = place_order(
                call_symbol,
                call_token,
                quantity * self.lot_size,
                "BUY",
                call_avg_price * sl,
                order_tag=stoploss_tag,
                stop_loss_order=True
            )
            put_sl_order_id = place_order(
                put_symbol,
                put_token,
                quantity * self.lot_size,
                "BUY",
                put_avg_price * sl,
                order_tag=stoploss_tag,
                stop_loss_order=True
            )
            call_stoploss_order_ids.append(call_sl_order_id)
            put_stoploss_order_ids.append(put_sl_order_id)
            sleep(0.3)

        orderbook = fetch_book("orderbook")
        call_sl_statuses = lookup_and_return(
            orderbook, "orderid", call_stoploss_order_ids, "status"
        )
        put_sl_statuses = lookup_and_return(
            orderbook, "orderid", put_stoploss_order_ids, "status"
        )

        if all(call_sl_statuses == "trigger pending") and all(
            put_sl_statuses == "trigger pending"
        ):
            notifier(
                f"{self.name} stoploss orders placed successfully", self.webhook_url
            )
        else:
            notifier(
                f"{self.name} stoploss orders not placed successfully", self.webhook_url
            )
            raise Exception("Stoploss orders not placed successfully")

        self.log_combined_order(equal_strike, expiry=expiry, buy_or_sell="SELL", call_price=call_avg_price,
                                put_price=put_avg_price, order_tag=order_tag)
        summary_message = "\n".join(
            f"{k}: {v}" for k, v in self.order_log[order_tag][0].items()
        )

        # Recording initial iv information

        traded_call_iv, traded_put_iv, traded_avg_iv = straddle_iv(
            call_avg_price,
            put_avg_price,
            entry_spot,
            equal_strike,
            timetoexpiry(expiry),
        )
        summary_iv = traded_avg_iv if traded_avg_iv is not None else 0
        summary_message += f"\nTraded IV: {summary_iv * 100:0.2f}"
        notifier(summary_message, self.webhook_url)
        sleep(1)

        def write_force_exit_status(user):
            with open(f"{user}_{self.name}_force_exit.json", "w") as file:
                json.dump(False, file)

        def read_force_exit_status(user):
            with open(f"{user}_{self.name}_force_exit.json", "r") as file:
                status = json.load(file)
                return status

        @log_errors
        def price_tracker():
            nonlocal call_price, put_price, underlying_price, call_avg_price, put_avg_price, call_exit_price
            nonlocal put_exit_price, mtm_price, profit_in_pts, call_iv, put_iv, avg_iv
            nonlocal sl_hit_dict, take_profit_exit, ctb_trg, ctb_hedge, force_exit

            # Print settings
            last_print_time = currenttime()
            print_interval = timedelta(seconds=5)

            # General settings
            days_to_expiry = timetoexpiry(expiry, in_days=True)

            # Smart exit settings
            smart_exit_trg = False
            smart_exit_delta_threshold = 0.06 if days_to_expiry < 1 else 0.12
            smart_exit_notification_sent = False
            incremental_gains = 100
            average_delta = 1

            # Hedge settings
            ctb_notification_sent = False
            ctb_message = ""
            profit_if_call_sl = put_avg_price - (call_avg_price * (sl - 1))
            profit_if_put_sl = call_avg_price - (put_avg_price * (sl - 1))
            ctb_threshold = max(profit_if_call_sl, profit_if_put_sl)

            def process_ctb(profit_threshold):
                strike_range = np.arange(
                    equal_strike - self.base * 2,
                    equal_strike + self.base * 3,
                    self.base,
                )  #
                # Hard-coding 2 strikes for now
                hedges = [*zip(strike_range, strike_range[::-1])][
                    -2:
                ]  # Hard-coding 2 hedges for now
                hedges = np.array(
                    [Strangle(pair[0], pair[1], self.name, expiry) for pair in hedges]
                )
                hedges_ltps = np.array([hedge.fetch_total_ltp() for hedge in hedges])
                distance_from_equal_strike = np.array(
                    [
                        hedge.call_option - equal_strike
                        if hedge.call_option < equal_strike
                        else hedge.put_option - equal_strike
                        for hedge in hedges
                    ]
                )
                hedge_profits = (
                    total_avg_price - hedges_ltps + distance_from_equal_strike
                )
                filtered_hedge = hedges[np.where(hedge_profits > profit_threshold)]
                print(
                    f"{self.name} CTB threshold: {profit_threshold}, Hedge working: {hedge_profits}"
                )
                if filtered_hedge.size > 0:
                    filtered_hedge = filtered_hedge[0]
                    return filtered_hedge

            while in_trade and not error_faced:
                # Update prices
                if websocket:
                    underlying_price = websocket.price_dict.get(
                        self.token, {"ltp": None}
                    )["ltp"]
                    call_price = websocket.price_dict.get(call_token, {"ltp": None})[
                        "ltp"
                    ]
                    put_price = websocket.price_dict.get(put_token, {"ltp": None})[
                        "ltp"
                    ]
                else:
                    underlying_price = self.fetch_ltp()
                    call_price = fetchltp("NFO", call_symbol, call_token)
                    put_price = fetchltp("NFO", put_symbol, put_token)

                # Fetch stop loss status
                callsl = sl_hit_dict["call"]
                putsl = sl_hit_dict["put"]

                # Calculate mtm price
                mtm_ce_price = call_exit_price if callsl else call_price
                mtm_pe_price = put_exit_price if putsl else put_price
                mtm_price = mtm_ce_price + mtm_pe_price

                # Calculate profit
                profit_in_pts = (call_avg_price + put_avg_price) - mtm_price
                profit_in_rs = profit_in_pts * self.lot_size * quantity_in_lots

                # Continuously check if profit is greater than target profit
                if take_profit and profit_in_pts > (
                    take_profit_points + per_share_charges
                ):
                    notifier(
                        f"{self.name} take profit exit triggered\n"
                        f"Time: {currenttime().time()}\n"
                        f"Profit: {profit_in_pts}\n",
                        self.webhook_url,
                    )
                    take_profit_exit = True

                # If no stop-loss is hit, and it is expiry day, then check for potential hedge purchase
                if (
                    not (callsl or putsl)
                    and days_to_expiry < 1
                    and convert_to_butterfly
                    and not ctb_notification_sent
                ):
                    try:
                        ctb_hedge = process_ctb(ctb_threshold)
                        if ctb_hedge is not None:
                            notifier(
                                f"{self.name} Convert to butterfly triggered\n",
                                self.webhook_url,
                            )
                            ctb_trg = True
                            ctb_message = f"Hedged with: {ctb_hedge}\n"
                            ctb_notification_sent = True
                    except Exception as e:
                        print(f"Error in process_ctb: {e}")

                # Continuously calculate IV
                call_iv, put_iv, avg_iv = straddle_iv(
                    call_price,
                    put_price,
                    underlying_price,
                    equal_strike,
                    timetoexpiry(expiry),
                )

                # If one of the stop-losses is hit then checking for smart exit
                if (
                    smart_exit
                    and (callsl or putsl)
                    and not (callsl and putsl)
                    and (call_iv or put_iv)
                    and not smart_exit_notification_sent
                ):
                    option_type = "p" if callsl else "c"
                    option_price = put_price if callsl else call_price
                    tracked_iv = put_iv if callsl and put_iv is not None else avg_iv
                    if tracked_iv is not None:
                        incremental_gains, average_delta = simulate_option_movement(
                            underlying_price,
                            equal_strike,
                            timetoexpiry(expiry),
                            option_type,
                            simulated_move=0.002,
                            r=0.06,
                            vol=tracked_iv,
                            price=option_price,
                        )
                    else:
                        incremental_gains, average_delta = 100, 1

                    if average_delta < smart_exit_delta_threshold:
                        if not smart_exit_notification_sent:
                            notifier(
                                f"{self.name} smart exit triggered\n"
                                f"Time: {currenttime().time()}\n"
                                f"Average delta: {average_delta}\n"
                                f"Incremental gains: {incremental_gains}\n",
                                self.webhook_url,
                            )
                            smart_exit_notification_sent = True
                            smart_exit_trg = True

                # Check for force exit
                if check_force_exit:
                    force_exit = read_force_exit_status(obj.userId)

                stoploss_message = ""
                if callsl:
                    stoploss_message += (
                        f"Call Exit Price: {mtm_ce_price}\nIncr. Gains: {incremental_gains}\n"
                        + f"Avg. Delta: {average_delta}\n"
                    )
                if putsl:
                    stoploss_message += (
                        f"Put Exit Price: {mtm_pe_price}\nIncr. Gains: {incremental_gains}\n"
                        + f"Avg. Delta: {average_delta}\n"
                    )
                print_iv = avg_iv if avg_iv is not None else 0
                if currenttime() - last_print_time > print_interval:
                    print(
                        f"Index: {self.name}\nTime: {currenttime().time()}\nStrike: {equal_strike}\n"
                        + f"Underlying Price: {underlying_price}\nCall SL: {callsl}\nPut SL: {putsl}\n"
                        + f"Call Price: {call_price}\nPut Price: {put_price}\n"
                        + stoploss_message
                        + f"Total price: {call_price + put_price:0.2f}\nMTM Price: {mtm_price:0.2f}\n"
                        + f"Profit in points: {profit_in_pts:0.2f}\n"
                        + f"Profit Value: {profit_in_rs:0.2f}\n"
                        + f"IV: {print_iv * 100:0.2f}\nSmart Exit: {smart_exit_trg}\n"
                        + ctb_message
                    )
                    last_print_time = currenttime()

        def process_order_statuses(
            order_book, order_ids, stop_loss=False, notify_url=None, context=""
        ):
            nonlocal orderbook

            pending_text = "trigger pending" if stop_loss else "open"
            context = f"{context} " if context else ""

            statuses = lookup_and_return(order_book, "orderid", order_ids, "status")

            if isinstance(statuses, (int, np.int32, np.int64)):
                logger.error(f"Statuses is {statuses} for orderid(s) {order_ids}")

            if all(statuses == pending_text):
                return False, False

            elif all(statuses == "rejected") or all(statuses == "cancelled"):
                rejection_reasons = lookup_and_return(
                    order_book, "orderid", order_ids, "text"
                )
                if all(
                    rejection_reasons == "17070 : The Price is out of the LPP range"
                ):
                    return True, False
                else:
                    notifier(
                        f"{context}Order rejected or cancelled. Reasons: {rejection_reasons[0]}",
                        notify_url,
                    )
                    raise Exception(f"Orders rejected or cancelled.")

            elif stop_loss and all(statuses == "pending"):
                sleep(1)
                orderbook = fetch_book("orderbook")
                statuses = lookup_and_return(orderbook, "orderid", order_ids, "status")

                if all(statuses == "pending"):
                    try:
                        cancel_pending_orders(order_ids, "NORMAL")
                    except Exception as e:
                        try:
                            cancel_pending_orders(order_ids, "STOPLOSS")
                        except Exception as e:
                            notifier(
                                f"{context}Could not cancel orders: {e}", notify_url
                            )
                            raise Exception(f"Could not cancel orders: {e}")
                    notifier(
                        f"{context}Orders pending and cancelled. Please check.",
                        notify_url,
                    )
                    return True, False

                elif all(statuses == "complete"):
                    return True, True

                else:
                    raise Exception(f"Orders in unknown state.")

            elif all(statuses == "complete"):
                return True, True

            else:
                notifier(
                    f"{context}Orders in unknown state. Statuses: {statuses}",
                    notify_url,
                )
                raise Exception(f"Orders in unknown state.")

        def fetch_orderbook_if_needed(
            data_class=shared_data, refresh_needed: bool = False
        ):
            if data_class is None or refresh_needed:
                return fetch_book("orderbook")
            if (
                currenttime() - data_class.updated_time < timedelta(seconds=15)
                and data_class.orderbook_data is not None
            ):
                return data_class.orderbook_data
            return fetch_book("orderbook")

        def check_sl_orders(order_ids, side: str, data=shared_data, refresh=False):
            """This function checks if the stop loss orders have been triggered or not. It also updates the order book
            in the nonlocal scope. This function is responsible for setting the exit prices.
            """

            nonlocal orderbook, call_exit_price, put_exit_price, call_price, put_price, underlying_price
            nonlocal traded_call_iv, traded_put_iv, traded_avg_iv, call_iv, put_iv, avg_iv, entry_spot

            orderbook = fetch_orderbook_if_needed(data, refresh)
            triggered, complete = process_order_statuses(
                orderbook,
                order_ids,
                stop_loss=True,
                notify_url=self.webhook_url,
                context=f"SL: {side}",
            )

            if not triggered and not complete:
                return False, False

            # Checking if there has been an unjustified trigger of stoploss without much movement in the underlying
            # We will also use IV to check if the stoploss was justified or not
            if triggered and safeguard:
                movement_from_entry = abs((underlying_price / entry_spot) - 1)
                present_iv = (
                    call_iv if side == "call" and call_iv is not None else avg_iv
                )
                present_price = call_price if side == "call" else put_price
                original_iv = (
                    traded_call_iv
                    if side == "call" and traded_call_iv is not None
                    else traded_avg_iv
                )

                if present_iv is None or original_iv is None:
                    notifier(
                        f"{self.name} {side.capitalize()} stoploss triggered. "
                        f"Unable to calculate IV spike due to missing IV data.",
                        self.webhook_url,
                    )
                else:
                    price_function = bs.call if side == "call" else bs.put
                    iv_spike = present_iv / original_iv
                    ideal_price = price_function(
                        underlying_price,
                        equal_strike,
                        timetoexpiry(expiry),
                        0.06,
                        original_iv,
                    )
                    price_spike = present_price / ideal_price

                    if movement_from_entry < safeguard_movement and (
                        iv_spike > safeguard_spike or price_spike > safeguard_spike
                    ):
                        notifier(
                            f"{self.name} {side.capitalize()} stoploss triggered without much "
                            f"movement in the underlying or because of IV/Price spike.\n"
                            f"Movement: {movement_from_entry * 100:0.2f}\nPresent IV: {present_iv}\n"
                            f"IV spike: {iv_spike}\nIdeal Price: {ideal_price}\nPresent Price: {present_price}\n"
                            f"Price Spike: {price_spike}",
                            self.webhook_url,
                        )
                    else:
                        notifier(
                            f"{self.name} {side.capitalize()} stoploss triggered. "
                            f"Movement: {movement_from_entry * 100:0.2f}\nPresent IV: {present_iv}\n"
                            f"IV spike: {iv_spike}\nIdeal Price: {ideal_price}\nPresent Price: {present_price}\n"
                            f"Price Spike: {price_spike}",
                            self.webhook_url,
                        )
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
            nonlocal underlying_price, take_profit_exit, in_trade, force_exit

            strike = findstrike(underlying_price, self.base)
            opt_type = "PE" if sl_type == "call" else "CE"
            symbol, token = fetch_symbol_token(self.name, expiry, strike, opt_type)
            option_ltp = fetchltp("NFO", symbol, token)
            qty = max(int(quantity_in_lots * qty_ratio), 1)
            trend_spliced_orders = self.splice_orders(qty)
            for spliced_qty in trend_spliced_orders:
                place_order(
                    symbol, token, spliced_qty * self.lot_size, "SELL", option_ltp * 0.9
                )
            sl_price = (
                (underlying_price * (1 - trend_sl))
                if sl_type == "call"
                else (underlying_price * (1 + trend_sl))
            )
            notifier(
                f"{self.name} {sl_type} trend catcher starting. "
                + f"Placed {qty} lots of {strike} {opt_type} at {option_ltp}. "
                + f"Stoploss price: {sl_price}, Underlying Price: {underlying_price}",
                self.webhook_url,
            )
            trend_sl_hit = False
            last_print_time = currenttime()
            print_interval = timedelta(seconds=10)
            while not check_exit_conditions(
                currenttime().time(),
                time(*exit_time),
                trend_sl_hit,
                self.intraday_straddle_forced_exit,
                take_profit_exit,
                not in_trade,
                force_exit,
            ):
                if sl_type == "call":
                    trend_sl_hit = underlying_price < sl_price
                else:
                    trend_sl_hit = underlying_price > sl_price
                sleep(1)
                if currenttime() - last_print_time > print_interval:
                    last_print_time = currenttime()
                    print(
                        f"{self.name} {sl_type} trend catcher running. "
                        + f"Stoploss price: {sl_price}, Underlying Price: {underlying_price} "
                        + f"Stoploss hit: {trend_sl_hit}"
                    )

            if trend_sl_hit:
                notifier(
                    f"{self.name} {sl_type} trend catcher stoploss hit.",
                    self.webhook_url,
                )
            else:
                notifier(
                    f"{self.name} {sl_type} trend catcher exiting.", self.webhook_url
                )

            for qty in trend_spliced_orders:
                place_order(symbol, token, qty * self.lot_size, "BUY", "MARKET")

        def process_sl_hit(
            sl_type,
            sl_dict,
            sl_orders_complete,
            symbol,
            token,
            other_symbol,
            other_token,
            other_stoploss_order_ids,
            other_avg_price,
        ):
            nonlocal take_profit_exit, call_exit_price, put_exit_price

            if all(sl_dict.values()):
                # print(f'{self.name} both stoploss orders completed. Not processing {sl_type}.')
                return

            other_sl_type = "call" if sl_type == "put" else "put"

            if not sl_orders_complete:
                for qty in spliced_orders:
                    place_order(symbol, token, qty * self.lot_size, "BUY", "MARKET")

            if catch_trend:
                trend_thread = Thread(
                    target=trend_catcher,
                    args=(sl_type, trend_qty_ratio, trend_catcher_sl),
                )
                trend_thread.start()

            if move_sl:
                for o_id in other_stoploss_order_ids:
                    obj.cancelOrder(o_id, "STOPLOSS")

                other_stoploss_order_ids.clear()
                for qty in spliced_orders:
                    sl_order_id = place_order(
                        other_symbol,
                        other_token,
                        qty * self.lot_size,
                        "BUY",
                        other_avg_price,
                        order_tag=stoploss_tag,
                        stop_loss_order=True

                    )
                    other_stoploss_order_ids.append(sl_order_id)

            # notifier(f'{self.name} {sl_type} stoploss triggered and completed.', self.webhook_url)

            refresh = True
            sleep(5)
            while not check_exit_conditions(
                currenttime().time(),
                time(*exit_time),
                sl_dict[other_sl_type],
                self.intraday_straddle_forced_exit,
                take_profit_exit,
                force_exit,
            ):
                sl_dict[other_sl_type], other_sl_orders_complete = check_sl_orders(
                    other_stoploss_order_ids, other_sl_type, refresh=refresh
                )
                if sl_dict[other_sl_type]:
                    if not other_sl_orders_complete:
                        for qty in spliced_orders:
                            place_order(
                                other_symbol,
                                other_token,
                                qty * self.lot_size,
                                "BUY",
                                "MARKET",
                            )
                    break
                else:
                    sleep(sleep_interval)
                refresh = False

        def process_exit(call_stop_loss_hit, put_stop_loss_hit, hedged=False):
            def exit_spliced_orders(orders, side):
                symbol = call_symbol if side == "call" else put_symbol
                token = call_token if side == "call" else put_token
                price = call_price if side == "call" else put_price

                for qty in orders:
                    place_order(
                        symbol,
                        token,
                        qty * self.lot_size,
                        "BUY",
                        price * 1.1,
                        "Exit order",
                    )
                    sleep(0.3)

            if call_stop_loss_hit and put_stop_loss_hit:
                sl_type = "Both"
            elif call_stop_loss_hit:
                sl_type = "Call"
                exit_spliced_orders(spliced_orders, "put")
            elif put_stop_loss_hit:
                sl_type = "Put"
                exit_spliced_orders(spliced_orders, "call")
            else:
                sl_type = "None"
                self.place_combined_order(
                    expiry, "BUY", quantity_in_lots, strike=equal_strike
                )
                if hedged:
                    # noinspection PyUnresolvedReferences
                    self.place_combined_order(
                        expiry,
                        "SELL",
                        quantity_in_lots,
                        call_strike=ctb_hedge.call_strike,
                        put_strike=ctb_hedge.put_strike,
                    )

            return sl_type

        # After placing the orders and stoploss orders setting up nonlocal variables
        in_trade = True
        error_faced = False

        # Take profit settings
        take_profit_exit = False

        # Convert to butterfly settings
        ctb_trg = False
        ctb_hedge = None

        # Writing force exit status
        force_exit = False
        if check_force_exit:
            write_force_exit_status(obj.userId)

        # Price information
        call_exit_price = 0
        put_exit_price = 0
        mtm_price = 0
        profit_in_pts = 0

        # IV information
        call_iv = 0
        put_iv = 0
        avg_iv = 0

        # Basic price information being set in outer scope as well
        total_avg_price = call_avg_price + put_avg_price
        if take_profit:
            per_share_charges = charges(
                (call_avg_price + put_avg_price),
                self.lot_size,
                quantity_in_lots,
                self.lot_size,
            )
            break_even_price = total_avg_price - per_share_charges
            notifier(
                f"{self.name}: Charges per share {per_share_charges} | Break even price {break_even_price}",
                self.webhook_url,
            )

        # Setting up stop loss dictionary and starting price thread
        sl_hit_dict = {"call": False, "put": False}
        price_updater = Thread(target=price_tracker)

        sleep(5)
        refresh_orderbook = True
        price_updater.start()
        # Monitoring begins here
        while not check_exit_conditions(
            currenttime().time(),
            time(*exit_time),
            any(sl_hit_dict.values()),
            self.intraday_straddle_forced_exit,
            take_profit_exit,
            ctb_trg,
            force_exit,
        ):
            try:
                sl_hit_dict["call"], call_sl_orders_complete = check_sl_orders(
                    call_stoploss_order_ids, "call", refresh=refresh_orderbook
                )
                sl_hit_dict["put"], put_sl_orders_complete = check_sl_orders(
                    put_stoploss_order_ids, "put", refresh=refresh_orderbook
                )
            except Exception as e:
                notifier(f"{self.name} Error: {e}", self.webhook_url)
                error_faced = True
                price_updater.join()
                raise Exception(f"Error: {e}")

            if sl_hit_dict["call"]:
                process_sl_hit(
                    "call",
                    sl_hit_dict,
                    call_sl_orders_complete,
                    call_symbol,
                    call_token,
                    put_symbol,
                    put_token,
                    put_stoploss_order_ids,
                    put_avg_price,
                )

            if sl_hit_dict["put"]:
                process_sl_hit(
                    "put",
                    sl_hit_dict,
                    put_sl_orders_complete,
                    put_symbol,
                    put_token,
                    call_symbol,
                    call_token,
                    call_stoploss_order_ids,
                    call_avg_price,
                )
            refresh_orderbook = False
            sleep(sleep_interval)

        # Out of main loop

        # If we are hedged then wait till exit time
        if ctb_trg:
            # noinspection PyUnresolvedReferences
            self.place_combined_order(
                expiry,
                "BUY",
                quantity_in_lots,
                call_strike=ctb_hedge.call_strike,
                put_strike=ctb_hedge.put_strike,
                order_tag=order_tag + " CTB",
            )
            cancel_pending_orders(call_stoploss_order_ids + put_stoploss_order_ids)
            notifier(f"{self.name}: Converted to butterfly", self.webhook_url)
            while not check_exit_conditions(currenttime().time(), time(*exit_time)):
                sleep(3)

        # After a complete exit is triggered
        sl_hit_call, sl_hit_put = sl_hit_dict["call"], sl_hit_dict["put"]

        # Fetching prices as a backup incase websocket has failed
        if websocket:
            call_price = fetchltp("NFO", call_symbol, call_token)
            put_price = fetchltp("NFO", put_symbol, put_token)

        # Exiting remaining positions if any
        stop_loss_type = process_exit(sl_hit_call, sl_hit_put, hedged=ctb_trg)

        # New code for cancelling pending orders
        pending_order_ids = lookup_and_return(
            "orderbook",
            ["ordertag", "status"],
            [stoploss_tag, "trigger pending"],
            "orderid",
        )

        if isinstance(pending_order_ids, (str, np.ndarray)):
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
            notifier(
                f"{self.name}: Error updating order list with exit details. {e}",
                self.webhook_url,
            )

        notifier(
            f"{self.name}: Exited positions\n"
            + "".join([f"{key}: {value}\n" for key, value in exit_dict.items()]),
            self.webhook_url,
        )
        in_trade = False

    @log_errors
    def intraday_strangle(
        self,
        quantity_in_lots,
        call_strike_offset=0,
        put_strike_offset=0,
        stop_loss=1.6,
        call_stop_loss=None,
        put_stop_loss=None,
        exit_time=(15, 29),
        sleep_time=5,
        catch_trend=False,
        trend_qty_ratio=1,
        trend_strike_offset=0,
        trend_sl=0.003,
        place_sl_orders=False
    ):

        """Intraday strangle strategy. Trades strangle with  stop loss. All offsets are in percentage terms.
        Parameters
        ----------
        quantity_in_lots : int
            Quantity in lots
        call_strike_offset : float, optional
            Call strike offset in percentage terms, by default 0
        put_strike_offset : float, optional
            Put strike offset in percentage terms, by default 0
        stop_loss : float, optional
            Stop loss percentage, by default 1.6
        call_stop_loss : float, optional
            Call stop loss percentage, by default None. If None then stop loss is same as stop_loss.
        put_stop_loss : float, optional
            Put stop loss percentage, by default None. If None then stop loss is same as stop_loss.
        exit_time : tuple, optional
            Exit time, by default (15, 29)
        sleep_time : int, optional
            Sleep time in seconds for updating prices, by default 5
        catch_trend : bool, optional
            Catch trend or not, by default False
        trend_qty_ratio : int, optional
            Ratio of trend quantity to strangle quantity, by default 1
        trend_strike_offset : float, optional
            Strike offset for trend order in percentage terms, by default 0
        trend_sl : float, optional
            Stop loss for trend order, by default 0.003
        place_sl_orders : bool, optional
            Place stop loss orders or not, by default False
        """

        @log_errors
        def position_monitor(info_dict):

            c_avg_price = info_dict["call_avg_price"]
            p_avg_price = info_dict["put_avg_price"]
            traded_strangle = info_dict["traded_strangle"]

            # Price deque
            n_prices = max(int(30/sleep_time), 1)  # Hard coded 30-second price window for now
            last_n_prices = {
                "call": deque(maxlen=n_prices), "put": deque(maxlen=n_prices), "underlying": deque(maxlen=n_prices)
            }

            last_print_time = currenttime()
            last_log_time = currenttime()
            last_notify_time = currenttime()
            print_interval = timedelta(seconds=5)
            log_interval = timedelta(minutes=60)
            notify_interval = timedelta(minutes=180)

            while not info_dict["exit_triggers"]["trade_complete"]:

                # Fetching prices
                spot_price = self.fetch_ltp()
                c_ltp, p_ltp = traded_strangle.fetch_ltp()
                info_dict["underlying_ltp"] = spot_price
                info_dict["call_ltp"] = c_ltp
                info_dict["put_ltp"] = p_ltp
                last_n_prices["call"].append(c_ltp)
                last_n_prices["put"].append(p_ltp)
                last_n_prices["underlying"].append(spot_price)
                c_ltp_avg = sum(last_n_prices["call"])/len(last_n_prices["call"]) if last_n_prices["call"] else c_ltp
                p_ltp_avg = sum(last_n_prices["put"])/len(last_n_prices["put"]) if last_n_prices["put"] else p_ltp
                spot_price_avg = sum(last_n_prices["underlying"])/len(last_n_prices["underlying"]) \
                    if last_n_prices["underlying"] else spot_price
                info_dict["call_ltp_avg"] = c_ltp_avg
                info_dict["put_ltp_avg"] = p_ltp_avg
                info_dict["underlying_ltp_avg"] = spot_price_avg

                # Calculate IV
                call_iv, put_iv, avg_iv = strangle_iv(
                    callprice=c_ltp,
                    putprice=p_ltp,
                    callstrike=traded_strangle.call_strike,
                    putstrike=traded_strangle.put_strike,
                    spot=spot_price,
                    timeleft=timetoexpiry(expiry)
                )
                info_dict["call_iv"] = call_iv
                info_dict["put_iv"] = put_iv
                info_dict["avg_iv"] = avg_iv

                # Calculate mtm price
                call_exit_price = info_dict.get('call_exit_price', c_ltp)
                put_exit_price = info_dict.get('put_exit_price', p_ltp)
                mtm_price = call_exit_price + put_exit_price

                # Calculate profit
                profit_in_pts = (c_avg_price + p_avg_price) - mtm_price
                profit_in_rs = profit_in_pts * self.lot_size * quantity_in_lots
                info_dict["profit_in_pts"] = profit_in_pts
                info_dict["profit_in_rs"] = profit_in_rs

                message = (
                    f"\nUnderlying: {self.name}\n"
                    f"Time: {currenttime():%d-%m-%Y %H:%M:%S}\n"
                    f"Underlying LTP: {spot_price}\n"
                    f"Call Strike: {traded_strangle.call_strike}\n"
                    f"Put Strike: {traded_strangle.put_strike}\n"
                    f"Call Price: {c_ltp}\n"
                    f"Put Price: {p_ltp}\n"
                    f"MTM Price: {mtm_price}\n"
                    f"Call last n avg: {c_ltp_avg}\n"
                    f"Put last n avg: {p_ltp_avg}\n"
                    f"IVs: {call_iv}, {put_iv}, {avg_iv}\n"
                    f"Call SL: {info_dict['call_sl']}\n"
                    f"Put SL: {info_dict['put_sl']}\n"
                    f"Profit Pts: {info_dict['profit_in_pts']:.2f}\n"
                    f"Profit: {info_dict['profit_in_rs']:.2f}\n"
                )
                if currenttime() - last_print_time > print_interval:
                    print(message)
                    last_print_time = currenttime()
                if currenttime() - last_log_time > log_interval:
                    logger.info(message)
                    last_log_time = currenttime()
                if currenttime() - last_notify_time > notify_interval:
                    notifier(message, self.webhook_url)
                    last_notify_time = currenttime()
                sleep(sleep_time)

        def get_range_of_strangles(c_strike, p_strike, exp, range_of_strikes=4):
            if range_of_strikes % 2 != 0:
                range_of_strikes += 1
            c_strike_range = np.arange(
                c_strike - (range_of_strikes / 2) * self.base,
                c_strike + (range_of_strikes / 2) * self.base + self.base,
                self.base
            )
            if c_strike == p_strike:
                return [Straddle(strike, self.name, exp) for strike in c_strike_range]
            else:
                p_strike_ranges = np.arange(
                        p_strike - (range_of_strikes/2)*self.base,
                        p_strike + (range_of_strikes/2)*self.base + self.base,
                        self.base
                )
                pairs = itertools.product(c_strike_range, p_strike_ranges)
                return [Strangle(pair[0], pair[1], self.name, exp) for pair in pairs]

        @log_errors
        def trend_catcher(info_dict, sl_type, qty_ratio, sl, strike_offset):

            offset = 1-strike_offset if sl_type == "call" else 1+strike_offset

            spot_price = info_dict["underlying_ltp"]

            # Setting up the trend option
            strike = spot_price * offset
            strike = findstrike(strike, self.base)
            opt_type = "PE" if sl_type == "call" else "CE"
            qty_in_lots = max(int(quantity_in_lots * qty_ratio), 1)
            trend_option = Option(strike, opt_type, self.name, expiry)

            # Placing the trend option order
            place_option_order_and_notify(
                trend_option, "SELL", qty_in_lots, "LIMIT", "Intraday Strangle Trend Catcher", self.webhook_url
            )

            # Setting up the stop loss
            sl_multiplier = 1 - sl if sl_type == "call" else 1 + sl
            sl_price = spot_price * sl_multiplier
            trend_sl_hit = False

            notifier(
                f"{self.name} strangle {sl_type} trend catcher starting. "
                + f"Placed {qty_in_lots} lots of {strike} {opt_type} at {trend_option.fetch_ltp()}. "
                + f"Stoploss price: {sl_price}, Underlying Price: {spot_price}",
                self.webhook_url,
            )

            last_print_time = currenttime()
            print_interval = timedelta(seconds=10)
            while all([currenttime().time() < time(*exit_time), not info_dict["exit_triggers"]["trade_complete"]]):
                spot_price = info_dict["underlying_ltp"]
                spot_price_avg = info_dict["underlying_ltp_avg"]
                trend_sl_hit = spot_price_avg < sl_price if sl_type == "call" else spot_price_avg > sl_price
                if trend_sl_hit:
                    break
                sleep(sleep_time)
                if currenttime() - last_print_time > print_interval:
                    last_print_time = currenttime()
                    print(
                        f"{self.name} {sl_type} trend catcher running\n"
                        + f"Stoploss price: {sl_price}, Underlying price: {spot_price}\n"
                        + f"Underlying price avg: {spot_price_avg}, Stoploss hit: {trend_sl_hit}\n"
                    )

            if trend_sl_hit:
                notifier(
                    f"{self.name} strangle {sl_type} trend catcher stoploss hit.", self.webhook_url
                )
            else:
                notifier(
                    f"{self.name} strangle {sl_type} trend catcher exiting.", self.webhook_url
                )

            # Buying the trend option back
            place_option_order_and_notify(
                trend_option, "BUY", qty_in_lots, "LIMIT", "Intraday Strangle Trend Catcher", self.webhook_url
            )

        def _stop_loss_triggered(info, side, stop_loss_order_ids):
            if stop_loss_order_ids is not None:
                return  # Should return a boolean once implemented

            avg_price = info.get(f"{side}_ltp_avg")
            stop_loss_price = info.get(f"{side}_stop_loss_price")
            return avg_price > stop_loss_price

        def justify_stop_loss(info, side):

            entry_spot = info.get("spot_at_entry")
            current_spot = info.get("underlying_ltp")

            # If the spot has moved in the direction of stop loss
            time_left_day_start = info.get("time_left_day_start")
            time_left_now = timetoexpiry(expiry)
            time_delta = (time_left_day_start - time_left_now)*525600
            time_delta = int(time_delta)
            estimated_movement = bs.target_movement(
                side,
                info.get(f"{side}_avg_price"),
                info.get(f"{side}_stop_loss_price"),
                entry_spot,
                info.get("traded_strangle").call_strike if side == "call" else info.get("traded_strangle").put_strike,
                time_left_day_start,
                time_delta
            )
            actual_movement = (current_spot - entry_spot) / entry_spot
            difference_in_sign = np.sign(estimated_movement) != np.sign(actual_movement)
            lack_of_movement = abs(actual_movement) < 0.8 * abs(estimated_movement)
            # 0.8 above is a magic number TODO: Remove magic number and find a better way to check for lack of movement
            if difference_in_sign or lack_of_movement:
                if not info.get(f"{side}_sl_check_notification_sent"):
                    message = f'{self.name} strangle {side} stop loss appears to be unjustified. ' \
                              f'Estimated movement: {estimated_movement}, Actual movement: {actual_movement}'
                    notifier(message, self.webhook_url)
                    info[f"{side}_sl_check_notification_sent"] = True
                return
            else:
                message = f"{self.name} strangle {side} stop loss triggered. " \
                          f"Estimated movement: {estimated_movement}, Actual movement: {actual_movement}"
                notifier(message, self.webhook_url)
                info[f"{side}_sl"] = True

        def check_for_stop_loss(info, side, stop_loss_order_ids=None):
            """Check for stop loss."""
            stop_loss_triggered = _stop_loss_triggered(info, side, stop_loss_order_ids)
            if stop_loss_triggered is None:
                return  # TODO: Implement fetching order status and checking for stop loss
            if stop_loss_triggered:
                justify_stop_loss(info, side)

        def process_stop_loss(info_dict, sl_type):

            if info_dict["call_sl"] and info_dict["put_sl"]:  # Check to avoid double processing
                return

            traded_strangle = info_dict["traded_strangle"]
            # Buying the stop loss option back
            option_to_buy = traded_strangle.call_option if sl_type == "call" else traded_strangle.put_option
            exit_price = place_option_order_and_notify(
                option_to_buy, "BUY", quantity_in_lots, "LIMIT", order_tag, self.webhook_url
            )
            info_dict[f'{sl_type}_exit_price'] = exit_price

            # Starting the trend catcher
            if catch_trend:
                trend_thread = Thread(
                    target=trend_catcher,
                    args=(info_dict, sl_type, trend_qty_ratio, trend_sl, trend_strike_offset),
                )
                trend_thread.start()

            # Wait for exit or other stop loss to hit
            other_sl_type = "put" if sl_type == "call" else "call"
            while all([currenttime().time() < time(*exit_time)]):
                check_for_stop_loss(info_dict, other_sl_type)
                if info_dict[f"{other_sl_type}_sl"]:
                    other_sl_option = traded_strangle.put_option if sl_type == "call" else traded_strangle.call_option
                    notifier(f'{self.name} strangle {other_sl_type} stop loss hit.', self.webhook_url)
                    other_exit_price = place_option_order_and_notify(
                        other_sl_option, "BUY", quantity_in_lots, "LIMIT", order_tag, self.webhook_url
                    )
                    info_dict[f'{other_sl_type}_exit_price'] = other_exit_price
                    break
                sleep(1)

        # Setting strikes and expiry
        order_tag = "Intraday Strangle"
        underlying_ltp = self.fetch_ltp()
        temp_call_strike = underlying_ltp * (1 + call_strike_offset)
        temp_put_strike = underlying_ltp * (1 - put_strike_offset)
        temp_call_strike = findstrike(temp_call_strike, self.base)
        temp_put_strike = findstrike(temp_put_strike, self.base)
        expiry = self.current_expiry

        prospective_strangles = get_range_of_strangles(temp_call_strike, temp_put_strike, expiry, range_of_strikes=4)

        # Placing the main order
        strangle = most_equal_strangle(*prospective_strangles)
        call_ltp, put_ltp = strangle.fetch_ltp()
        call_avg_price, put_avg_price = place_option_order_and_notify(
            strangle, "SELL", quantity_in_lots, "LIMIT", order_tag, self.webhook_url, return_avg_price=True
        )
        total_avg_price = call_avg_price + put_avg_price

        call_stop_loss_price = call_avg_price * call_stop_loss if call_stop_loss else call_avg_price * stop_loss
        put_stop_loss_price = put_avg_price * put_stop_loss if put_stop_loss else put_avg_price * stop_loss

        # Logging information and sending notification
        self.log_combined_order(
            call_strike=strangle.call_strike,
            put_strike=strangle.put_strike,
            expiry=expiry,
            buy_or_sell="SELL",
            call_price=call_avg_price,
            put_price=put_avg_price,
            order_tag=order_tag
        )

        summary_message = "\n".join(
            f"{k}: {v}" for k, v in self.order_log[order_tag][-1].items()
        )

        traded_call_iv, traded_put_iv, traded_avg_iv = strangle_iv(
            callprice=call_avg_price,
            putprice=put_avg_price,
            callstrike=strangle.call_strike,
            putstrike=strangle.put_strike,
            spot=underlying_ltp,
            timeleft=timetoexpiry(expiry)
        )

        time_left_at_trade = timetoexpiry(expiry)
        summary_message += f"\nTraded IVs: {traded_call_iv}, {traded_put_iv}, {traded_avg_iv}"
        summary_message += f"\nCall SL: {call_stop_loss_price}, Put SL: {put_stop_loss_price}"
        notifier(summary_message, self.webhook_url)

        if place_sl_orders:
            call_stop_loss_order_ids = place_option_order_and_notify(
                instrument=strangle.call_option,
                action="BUY",
                qty_in_lots=quantity_in_lots,
                prices=call_stop_loss_price,
                order_tag="Call SL Strangle",
                webhook_url=self.webhook_url,
                stop_loss_order=True,
                target_status="trigger pending",
                return_avg_price=False
            )
            put_stop_loss_order_ids = place_option_order_and_notify(
                instrument=strangle.put_option,
                action="BUY",
                qty_in_lots=quantity_in_lots,
                prices=put_stop_loss_price,
                order_tag="Put SL Strangle",
                webhook_url=self.webhook_url,
                stop_loss_order=True,
                target_status="trigger pending",
                return_avg_price=False
            )
        else:
            call_stop_loss_order_ids = None
            put_stop_loss_order_ids = None

        # Setting up shared info dict
        shared_info_dict = {
            "traded_strangle": strangle,
            "spot_at_entry": underlying_ltp,
            "call_avg_price": call_avg_price,
            "put_avg_price": put_avg_price,
            "call_iv_at_entry": traded_call_iv,
            "put_iv_at_entry": traded_put_iv,
            "avg_iv_at_entry": traded_avg_iv,
            "call_stop_loss_price": call_stop_loss_price,
            "put_stop_loss_price": put_stop_loss_price,
            "call_stop_loss_order_ids": call_stop_loss_order_ids,
            "put_stop_loss_order_ids": put_stop_loss_order_ids,
            "time_left_day_start": time_left_at_trade,
            "call_ltp": call_ltp,
            "put_ltp": put_ltp,
            "underlying_ltp": underlying_ltp,
            "call_iv": traded_call_iv,
            "put_iv": traded_put_iv,
            "avg_iv": traded_avg_iv,
            "call_sl": False,
            "put_sl": False,
            "exit_triggers": {"trade_complete": False},
            "call_sl_check_notification_sent": False,
            "put_sl_check_notification_sent": False,
        }

        position_monitor_thread = Thread(target=position_monitor, args=(shared_info_dict,))
        position_monitor_thread.start()
        sleep(3)  # To ensure that the position monitor thread has started

        # Wait for exit time or both stop losses to hit (Main Loop)
        while all([currenttime().time() < time(*exit_time)]):
            check_for_stop_loss(shared_info_dict, 'call')
            if shared_info_dict["call_sl"]:
                process_stop_loss(shared_info_dict, "call")
                break
            check_for_stop_loss(shared_info_dict, 'put')
            if shared_info_dict["put_sl"]:
                process_stop_loss(shared_info_dict, "put")
                break
            sleep(1)

        # Out of the while loop, so exit time reached or both stop losses hit

        call_sl = shared_info_dict["call_sl"]
        put_sl = shared_info_dict["put_sl"]

        if not call_sl and not put_sl:
            # Both stop losses not hit
            call_exit_avg_price, put_exit_avg_price = place_option_order_and_notify(
                strangle, "BUY", quantity_in_lots, "LIMIT", order_tag, self.webhook_url, return_avg_price=True
            )

            shared_info_dict['call_exit_price'] = call_exit_avg_price
            shared_info_dict['put_exit_price'] = put_exit_avg_price

        elif (call_sl or put_sl) and not (call_sl and put_sl):  # Only one stop loss hit
            exit_option = strangle.put_option if call_sl else strangle.call_option
            non_sl_exit_price = place_option_order_and_notify(
                exit_option, "BUY", quantity_in_lots, "LIMIT", order_tag, self.webhook_url
            )
            exit_option = "put" if call_sl else "call"
            shared_info_dict[f"{exit_option}_exit_price"] = non_sl_exit_price

        else:  # Both stop losses hit
            pass

        # Calculate profit
        total_exit_price = shared_info_dict["call_exit_price"] + shared_info_dict["put_exit_price"]
        exit_message = (
            f"{self.name} strangle exited.\n"
            f"Time: {currenttime():%d-%m-%Y %H:%M:%S}\n"
            f"Underlying LTP: {shared_info_dict['underlying_ltp']}\n"
            f"Call Price: {shared_info_dict['call_ltp']}\n"
            f"Put Price: {shared_info_dict['put_ltp']}\n"
            f"Call SL: {shared_info_dict['call_sl']}\n"
            f"Put SL: {shared_info_dict['put_sl']}\n"
            f"Call Exit Price: {shared_info_dict['call_exit_price']}\n"
            f"Put Exit Price: {shared_info_dict['put_exit_price']}\n"
            f"Total Exit Price: {total_exit_price}\n"
            f"Total Entry Price: {total_avg_price}\n"
            f"Profit Pts: {total_avg_price - total_exit_price}\n"
        )
        exit_dict = {
            "Call exit price": shared_info_dict["call_exit_price"],
            "Put exit price": shared_info_dict["put_exit_price"],
            "Total exit price": total_exit_price,
            "Points captured": total_avg_price - total_exit_price,
            "Call SL": shared_info_dict["call_sl"],
            "Put SL": shared_info_dict["put_sl"],
        }
        try:
            self.order_log[order_tag][0].update(exit_dict)
        except Exception as e:
            notifier(
                f"{self.name}: Error updating order list with exit details. {e}",
                self.webhook_url,
            )
        notifier(exit_message, self.webhook_url)
        shared_info_dict["exit_triggers"] = {"trade_complete": True}
        position_monitor_thread.join()
        return shared_info_dict

    @log_errors
    def intraday_trend(
            self,
            quantity_in_lots,
            start_time=(9, 15, 55),
            exit_time=(15, 27),
            sleep_time=5,
            threshold_movement=None,
            minutes_to_avg=45,
    ):

        while currenttime().time() < time(*start_time):
            print(f"{self.name} trender sleeping till {start_time}")
            sleep(1)

        open_price = self.fetch_ltp()
        movement = 0

        if threshold_movement is None:
            vix = get_current_vix()
            beta = dm.get_summary_ratio(self.name, 'NIFTY')
            beta = 1.3 if beta is None else beta
            vix = vix * beta
            threshold_movement = vix / 48

        exit_time = time(*exit_time)
        scan_end_time = datetime.combine(currenttime().date(), exit_time)
        scan_end_time = scan_end_time - timedelta(minutes=10)
        scan_end_time = scan_end_time.time()
        upper_limit = open_price * (1 + threshold_movement / 100)
        lower_limit = open_price * (1 - threshold_movement / 100)

        # Price deque
        n_prices = max(int(minutes_to_avg / sleep_time), 1)
        price_deque = deque(maxlen=n_prices)

        notifier(
            f"{self.name} trender starting with {threshold_movement:0.2f} threshold movement\n"
            f"Current Price: {open_price}\nUpper limit: {upper_limit:0.2f}\n"
            f"Lower limit: {lower_limit:0.2f}.",
            self.webhook_url,
        )
        last_print_time = currenttime()
        while (
                abs(movement) < threshold_movement and currenttime().time() < scan_end_time
        ):
            ltp = self.fetch_ltp()
            price_deque.append(ltp)
            ltp_avg = sum(price_deque) / len(price_deque) if price_deque else ltp
            movement = ((ltp_avg / open_price) - 1) * 100
            if currenttime() > last_print_time + timedelta(minutes=1):
                print(f"{self.name} trender: {movement:0.2f} movement.")
                last_print_time = currenttime()
            sleep(sleep_time)

        if currenttime().time() > scan_end_time:
            notifier(f"{self.name} trender exiting due to time.", self.webhook_url)
            return

        price = self.fetch_ltp()
        atm_strike = findstrike(price, self.base)
        position = "BUY" if movement > 0 else "SELL"
        atm_synthetic_fut = SyntheticFuture(atm_strike, self.name, self.current_expiry)
        stop_loss_multiplier = 1.0032 if position == "SELL" else 0.9968
        stop_loss_price = price * stop_loss_multiplier
        stop_loss_hit = False
        notifier(
            f"{self.name} {position} trender triggered with {movement:0.2f} movement. {self.name} at {price}. "
            f"Stop loss at {stop_loss_price}.",
            self.webhook_url,
        )
        place_option_order_and_notify(
            atm_synthetic_fut, position, quantity_in_lots, "LIMIT", f"{self.name} trender", self.webhook_url
        )

        while currenttime().time() < exit_time and not stop_loss_hit:

            ltp = self.fetch_ltp()
            price_deque.append(ltp)
            ltp_avg = sum(price_deque) / len(price_deque) if price_deque else ltp

            if position == "BUY":
                stop_loss_hit = ltp_avg < stop_loss_price
            else:
                stop_loss_hit = ltp_avg > stop_loss_price
            sleep(sleep_time)

        stop_loss_message = "Trender stop loss hit. " if stop_loss_hit else ""
        notifier(
            f"{stop_loss_message}{self.name} trender exiting. {self.name} at {self.fetch_ltp()}.",
            self.webhook_url,
        )
        place_option_order_and_notify(
            atm_synthetic_fut,
            "BUY" if position == "SELL" else "SELL",
            quantity_in_lots,
            "LIMIT",
            f"{self.name} trender",
            self.webhook_url,
        )

    def intraday_straddle_delta_hedged(
        self,
        quantity_in_lots,
        exit_time=(15, 30),
        websocket=None,
        wait_for_equality=False,
        delta_threshold=1,
        **kwargs,
    ):
        # Finding equal strike
        (
            equal_strike,
            call_symbol,
            put_symbol,
            call_token,
            put_token,
            call_price,
            put_price,
        ) = self.find_equal_strike(exit_time, websocket, wait_for_equality, **kwargs)
        expiry = self.current_expiry
        print(
            f"Index: {self.name}, Strike: {equal_strike}, Call: {call_price}, Put: {put_price}"
        )
        notifier(
            f"{self.name}: Initiating intraday trade on {equal_strike} strike.",
            self.webhook_url,
        )

        # Placing orders
        self.place_combined_order(
            expiry,
            "SELL",
            quantity_in_lots,
            strike=equal_strike,
            return_avg_price=True,
            order_tag="Intraday straddle with delta",
        )

        positions = {
            f"{self.name} {equal_strike} {expiry} CE": {
                "token": call_token,
                "quantity": -1 * quantity_in_lots * self.lot_size,
                "delta_quantity": 0,
            },
            f"{self.name} {equal_strike} {expiry} PE": {
                "token": put_token,
                "quantity": -1 * quantity_in_lots * self.lot_size,
                "delta_quantity": 0,
            },
        }

        synthetic_fut_call = f"{self.name} {equal_strike} {expiry} CE"
        synthetic_fut_put = f"{self.name} {equal_strike} {expiry} PE"
        delta_threshold = delta_threshold * self.lot_size

        while currenttime().time() < time(*exit_time):
            position_df = pd.DataFrame(positions).T
            if websocket:
                underlying_price = websocket.price_dict.get(self.token, 0)["ltp"]
                position_df["ltp"] = position_df["token"].apply(
                    lambda x: websocket.price_dict.get(x, "None")["ltp"]
                )
            else:
                underlying_price = self.fetch_ltp()
                position_df["ltp"] = position_df.index.map(
                    lambda x: fetchltp("NFO", *fetch_symbol_token(x))
                )

            position_df[["iv", "delta", "gamma"]] = position_df.apply(
                lambda row: calc_greeks(row.name, row.ltp, underlying_price), axis=1
            ).tolist()

            position_df["total_quantity"] = (
                position_df["quantity"] + position_df["delta_quantity"]
            )
            position_df["delta"] = position_df.delta * position_df.total_quantity
            position_df["gamma"] = position_df.gamma * position_df.total_quantity
            position_df.loc["Total"] = position_df.agg(
                {"delta": "sum", "gamma": "sum", "iv": "mean", "ltp": "mean"}
            )
            current_delta = position_df.loc["Total", "delta"]
            current_gamma = position_df.loc["Total", "gamma"]

            print(
                f'\n**** Starting Loop ****\n{position_df.drop(["token"], axis=1).to_string()}\n'
                + f"\nCurrent delta: {current_delta}\n"
            )

            if abs(current_delta) > delta_threshold:
                if current_delta > 0:  # We are long
                    lots_to_sell = round(abs(current_delta) / self.lot_size, 0)
                    notifier(
                        f"Delta greater than {delta_threshold}. Selling {lots_to_sell} "
                        + f"synthetic futures to reduce delta.\n",
                        self.webhook_url,
                    )
                    place_synthetic_fut_order(
                        self.name,
                        equal_strike,
                        expiry,
                        "SELL",
                        lots_to_sell,
                    )
                    positions[synthetic_fut_call]["delta_quantity"] -= (
                        lots_to_sell * self.lot_size
                    )
                    positions[synthetic_fut_put]["delta_quantity"] += (
                        lots_to_sell * self.lot_size
                    )

                else:  # We are short
                    lots_to_buy = round(abs(current_delta) / self.lot_size, 0)
                    notifier(
                        f"Delta less than -{delta_threshold}. Buying {lots_to_buy} "
                        + f"synthetic futures to reduce delta.\n",
                        self.webhook_url,
                    )
                    place_synthetic_fut_order(
                        self.name,
                        equal_strike,
                        expiry,
                        "BUY",
                        lots_to_buy,
                    )
                    positions[synthetic_fut_call]["delta_quantity"] += (
                        lots_to_buy * self.lot_size
                    )
                    positions[synthetic_fut_put]["delta_quantity"] -= (
                        lots_to_buy * self.lot_size
                    )

            sleep(2)

        # Closing the main positions along with the delta positions if any are open
        notifier(f"Intraday straddle with delta: Closing positions.", self.webhook_url)
        self.place_combined_order(
            expiry,
            "BUY",
            quantity_in_lots,
            strike=equal_strike,
            order_tag="Intraday straddle with delta",
        )

        # Squaring off the delta positions
        call_delta_quantity = positions[synthetic_fut_call]["delta_quantity"]
        put_delta_quantity = positions[synthetic_fut_put]["delta_quantity"]

        if call_delta_quantity != 0 and put_delta_quantity != 0:
            assert call_delta_quantity == -1 * put_delta_quantity
            quantity_to_square_up = abs(call_delta_quantity)
            quantity_to_square_up_in_lots = quantity_to_square_up / self.lot_size

            if call_delta_quantity > 0:
                action = "SELL"
            else:
                action = "BUY"

            self.place_synthetic_fut(
                equal_strike, expiry, action, quantity_to_square_up_in_lots
            )
            notifier(
                f"Intraday Straddle with delta: Squared off delta positions. "
                + f"{action} {quantity_to_square_up} synthetic futures.",
                self.webhook_url,
            )
        elif call_delta_quantity == 0 and put_delta_quantity == 0:
            notifier("No delta positions to square off.", self.webhook_url)
        else:
            raise AssertionError("Delta positions are not balanced.")


class Stock(Index):
    def __init__(
        self, name, webhook_url=None, websocket=None, spot_future_difference=0.06
    ):
        super().__init__(name, webhook_url, websocket, spot_future_difference)


def convert_to_serializable(data):
    if isinstance(data, dict):
        return {k: convert_to_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_serializable(item) for item in data]
    elif hasattr(data, "tolist"):  # Check for numpy arrays
        return data.tolist()
    elif hasattr(data, "item"):  # Check for numpy scalar types, e.g., numpy.int32
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


def word_to_num(s):
    word = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
        'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
        'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
        'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90
    }
    multiplier = {'thousand': 1000, 'hundred': 100, 'million': 1000000, 'billion': 1000000000}

    words = s.lower().split()
    if words[0] == 'a':
        words[0] = 'one'
    total = 0
    current = 0
    for w in words:
        if w in word:
            current += word[w]
        if w in multiplier:
            current *= multiplier[w]
        if w == 'and':
            continue
        if w == 'thousand' or w == 'million' or w == 'billion':
            total += current
            current = 0
    total += current
    return total


def login(user, pin, apikey, authkey, webhook_url=None):
    global obj, login_data
    authkey = pyotp.TOTP(authkey)
    obj = SmartConnect(api_key=apikey)
    login_data = obj.generateSession(user, pin, authkey.now())
    if login_data["message"] != "SUCCESS":
        for attempt in range(2, 7):
            sleep(10)
            notifier(f"Login attempt {attempt}.", webhook_url)
            login_data = obj.generateSession(user, pin, authkey.now())
            if login_data["message"] == "SUCCESS":
                break
            if attempt == 6:
                notifier("Login failed.", webhook_url)
                raise Exception("Login failed.")
    else:
        today = datetime.now().strftime("%Y-%m-%d")
        info_log_filename = f"{obj.userId}-info-{today}.log"
        error_log_filename = f"{obj.userId}-error-{today}.log"
        formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")

        for handler in logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)

        info_handler = logging.FileHandler(info_log_filename)
        info_handler.setLevel(logging.INFO)
        info_handler.setFormatter(formatter)
        logger.addHandler(info_handler)

        error_handler = logging.FileHandler(error_log_filename)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)

        logger.info("Logged in successfully.")

        set_error_notification_settings("user", f"{obj.userId} - ")
        notifier(
            f'Date: {currenttime().strftime("%d %b %Y %H:%M:%S")}\nLogged in successfully.',
            webhook_url,
        )


def parse_symbol(symbol):
    match = re.match(r"([A-Za-z]+)(\d{2}[A-Za-z]{3}\d{2})(\d+)(\w+)", symbol)
    if match:
        return match.groups()
    return None


def fetch_book(book):
    def fetch_data(fetch_func, description, max_attempts=6, sleep_duration=2):
        for attempt in range(1, max_attempts + 1):
            try:
                data = fetch_func()["data"]
                return data
            except DataException:
                if attempt == max_attempts:
                    raise Exception(
                        f"Failed to fetch {description} due to DataException."
                    )
                else:
                    sleep(sleep_duration)
            except Exception as e:
                if attempt == max_attempts:
                    raise Exception(f"Failed to fetch {description}: {e}")
                else:
                    print(f"Error {attempt} in fetching {description}: {e}")
                    sleep(sleep_duration)

    if book == "orderbook":
        return fetch_data(obj.orderBook, "orderbook")
    elif book in {"positions", "position"}:
        return fetch_data(obj.position, "positions")
    else:
        raise ValueError(f"Invalid book type '{book}'.")


def lookup_and_return(book, field_to_lookup, value_to_lookup, field_to_return):
    def filter_and_return(data: list):
        if not isinstance(field_to_lookup, list):
            field_to_lookup_ = [field_to_lookup]
            value_to_lookup_ = [value_to_lookup]
        else:
            field_to_lookup_ = field_to_lookup
            value_to_lookup_ = value_to_lookup

        bucket = [
            entry[field_to_return]
            for entry in data
            if all(
                (
                    entry[field] == value
                    if not isinstance(value, list)
                    else entry[field] in value
                )
                for field, value in zip(field_to_lookup_, value_to_lookup_)
            )
            and all(entry[field] != "" for field in field_to_lookup_)
        ]

        if len(bucket) == 0:
            return 0
        else:
            return np.array(bucket)

    if not (
        isinstance(field_to_lookup, (str, list, tuple, np.ndarray))
        and isinstance(value_to_lookup, (str, list, tuple, np.ndarray))
    ):
        raise ValueError(
            "Both 'field_to_lookup' and 'value_to_lookup' must be strings or lists."
        )

    if isinstance(field_to_lookup, list) and isinstance(value_to_lookup, str):
        raise ValueError(
            "Unsupported input: 'field_to_lookup' is a list and 'value_to_lookup' is a string."
        )

    if isinstance(book, list):
        return filter_and_return(book)
    elif isinstance(book, str) and book in {"orderbook", "positions"}:
        book_data = fetch_book(book)
        return filter_and_return(book_data)
    else:
        raise ValueError("Invalid input")


# Discord messenger
def notifier(message, webhook_url=None):
    if webhook_url is None or webhook_url is False:
        print(message)
        return
    else:
        notification_url = webhook_url
        data = {"content": message}
        try:
            requests.post(
                notification_url,
                data=json.dumps(data),
                headers={"Content-Type": "application/json"},
            )
            print(message)
        except requests.exceptions.SSLError as e:
            print(f"Error in sending notification: {e}")


def check_and_notify_order_statuses(statuses, webhook_url=None, target_status="complete", **kwargs):

    order_prefix = f"{kwargs['order_tag']}: " if "order_tag" in kwargs and kwargs["order_tag"] != "" else ""
    order_message = [f"{k}-{v}" for k, v in kwargs.items() if k != "order_tag"]
    order_message = ", ".join(order_message)

    if all(statuses == target_status):
        notifier(
            f"{order_prefix}Order(s) placed successfully for {order_message}",
            webhook_url
        )
    elif all(statuses == "rejected"):
        notifier(
            f"{order_prefix}All orders rejected for {order_message}",
            webhook_url
        )
        raise Exception("Orders rejected")
    elif any(statuses == "open"):
        notifier(
            f"{order_prefix}Some orders pending for {order_message}. You can modify the orders.",
            webhook_url
        )
    elif any(statuses == "rejected"):
        notifier(
            f"{order_prefix}Some orders rejected for {order_message}.\nYou can place the rejected orders again.",
            webhook_url
        )
    else:
        notifier(
            f"{order_prefix}ERROR. Order statuses uncertain for {order_message}",
            webhook_url
        )
        raise Exception("Order statuses uncertain")


def place_option_order_and_notify(
    instrument: Option | Strangle | Straddle | SyntheticFuture,
    action: str,
    qty_in_lots: int,
    prices: str | int | float | tuple | list | np.ndarray = "LIMIT",
    order_tag: str = "",
    webhook_url=None,
    stop_loss_order: bool = False,
    target_status: str = "complete",
    return_avg_price: bool = True,
    **kwargs
):

    notify_dict = {
        "target_status": target_status,
        "order_tag": order_tag,
        "Underlying": instrument.underlying,
        "Action": action,
        "Expiry": instrument.expiry,
        "Qty": qty_in_lots
    }

    order_params = {
        "transaction_type": action,
        "quantity_in_lots": qty_in_lots,
        "stop_loss_order": stop_loss_order,
        "order_tag": order_tag
    }

    if isinstance(instrument, (Strangle, Straddle, SyntheticFuture)):
        notify_dict.update({
            "Strikes": [instrument.call_strike, instrument.put_strike]})
        order_params.update({
            "prices": prices
        })
    elif isinstance(instrument, Option):
        notify_dict.update({
            "Strike": instrument.strike,
            "OptionType": instrument.option_type
        })
        order_params.update({
            "price": prices
        })
    else:
        raise ValueError("Invalid instrument type")

    notify_dict.update(kwargs)

    if stop_loss_order:
        assert isinstance(prices, (int, float, tuple, list, np.ndarray)), "Stop loss order requires a price"

    # Placing the order
    order_ids = instrument.place_order(**order_params)

    if isinstance(order_ids, tuple):  # Strangle/Straddle/SyntheticFuture
        call_order_ids, put_order_ids = order_ids[0], order_ids[1]
        order_ids = list(itertools.chain(call_order_ids, put_order_ids))
    else:  # Option
        call_order_ids, put_order_ids = False, False

    order_book = fetch_book('orderbook')
    order_statuses_ = lookup_and_return(order_book, 'orderid', order_ids, 'status')
    check_and_notify_order_statuses(order_statuses_, webhook_url, **notify_dict)

    if return_avg_price:
        if call_order_ids and put_order_ids:
            call_avg_price = lookup_and_return(
                order_book, 'orderid', call_order_ids, 'averageprice'
            ).astype(float).mean()
            put_avg_price = lookup_and_return(
                order_book, 'orderid', put_order_ids, 'averageprice'
            ).astype(float).mean()
            return call_avg_price, put_avg_price
        else:
            avg_price = lookup_and_return(order_book, 'orderid', order_ids, 'averageprice').astype(float).mean()
            return avg_price

    return order_ids


# Market Hours
def markethours():
    if time(9, 15) <= currenttime().time() <= time(15, 30):
        return True
    else:
        return False


def last_market_close_time():

    if currenttime().time() < time(9, 15):
        wip_time = currenttime() - timedelta(days=1)
        wip_time = wip_time.replace(hour=15, minute=30, second=0, microsecond=0)
    elif currenttime().time() > time(15, 30):
        wip_time = currenttime().replace(hour=15, minute=30, second=0, microsecond=0)
    else:
        wip_time = currenttime()

    if wip_time.weekday() not in [5, 6] and wip_time.date() not in holidays:
        return wip_time
    else:
        # Handling weekends and holidays
        while wip_time.weekday() in [5, 6] or wip_time.date() in holidays:
            wip_time = wip_time - timedelta(days=1)

    last_close_day_time = wip_time.replace(hour=15, minute=30, second=0, microsecond=0)
    return last_close_day_time


# Defining current time
def currenttime():
    # Adjusting for timezones
    ist = timezone(timedelta(hours=5, minutes=30))
    return datetime.now(ist).replace(tzinfo=None)


def simulate_option_movement(
    spot,
    strike,
    time_to_expiry,
    flag,
    direction="away",
    simulated_move=0.002,
    r=0.06,
    vol=None,
    price=None,
    print_results=False,
):
    """
    :param spot:
    :param strike:
    :param time_to_expiry:
    :param flag: option type for which price is to be simulated
    :param direction: the direction of the move, "away" or "towards"
    :param simulated_move:
    :param r:
    :param vol:
    :param price:
    :param print_results:
    :return:
    """

    if price is None and vol is None:
        raise ValueError("Either price or vol must be specified.")
    flag = flag.lower()[0]
    price_func = bs.put if flag == 'p' else bs.call

    if direction == "away":
        if flag == "c":
            simulated_move = -simulated_move
        else:
            simulated_move = simulated_move
    elif direction == "towards" or direction == "toward":
        if flag == "c":
            simulated_move = simulated_move
        else:
            simulated_move = -simulated_move
    else:
        raise ValueError("Invalid direction.")

    simulated_spot = spot * (1 + simulated_move)
    if vol is None:
        try:
            vol = bs.implied_volatility(price, spot, strike, time_to_expiry, r, flag)
        except ValueError:
            return None
    if price is None:
        price = price_func(spot, strike, time_to_expiry, r, vol)
    current_delta = bs.delta(spot, strike, time_to_expiry, r, vol, flag)

    new_vol = bs.iv_curve_adjustor(simulated_move, time_to_expiry, vol, spot, strike)
    new_delta = bs.delta(simulated_spot, strike, time_to_expiry, r, new_vol, flag)
    delta_change = new_delta - current_delta
    average_delta = abs((new_delta + current_delta) / 2)
    new_price = price_func(simulated_spot, strike, time_to_expiry, r, new_vol)
    price_gain = price - new_price

    if print_results:
        print(
            f"Current Delta: {current_delta:.2f}\nNew Delta: {new_delta:.2f}\n"
            f"Delta Change: {delta_change:.2f}\nAverage Delta: {average_delta:.2f}\n"
            f"New Price: {new_price:.2f}\nPrice Change: {price_gain:.2f}\n"
            f"Volatility: {vol:.2f}\nSimulated Spot: {simulated_spot:.2f}\n"
            f"Simulated Move: {simulated_move}%\n"
        )

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


def fetch_lot_size(name, expiry=None):
    if expiry is None:
        expiry_mask = scrips.expiry_formatted == scrips.expiry_formatted
    else:
        expiry_mask = scrips.expiry_formatted == expiry

    filtered_df = scrips.loc[
        (scrips.name == name) & (scrips.exch_seg == "NFO") & expiry_mask
    ]
    lot_sizes = filtered_df.lotsize.values

    if len(set(lot_sizes)) > 1:
        print(
            f"Multiple lot sizes found for {name}. Using the closest expiry lot size."
        )
        filtered_df = filtered_df.sort_values("expiry_dt")
        return filtered_df.lotsize.iloc[0]

    else:
        return lot_sizes[0]


def get_base(name):
    strike_array = scrips.loc[
        (scrips.name == name) & (scrips.exch_seg == "NFO")
    ].sort_values("expiry_dt")
    closest_expiry = strike_array.expiry_dt.iloc[0]
    strike_array = (
        strike_array.loc[strike_array.expiry_dt == closest_expiry]["strike"] / 100
    )
    strike_differences = np.diff(strike_array.sort_values().unique())
    values, counts = np.unique(strike_differences, return_counts=True)
    mode = values[np.argmax(counts)]
    return mode


def fetch_symbol_token(
    name=None, expiry=None, strike=None, option_type=None, tokens=None
):
    """Fetches symbol & token for a given scrip name. Provide just a single world if
    you want to fetch the symbol & token for the cash segment. If you want to fetch the
    symbol & token for the options segment, provide name, strike, expiry, option_type.
    Expiry should be in the DDMMMYY format. Optiontype should be CE or PE. Optionally, provide
    a list of tokens to fetch the corresponding symbols."""

    if tokens is None and name is None:
        raise ValueError("Either name or tokens must be specified.")

    if tokens is not None:
        token_df = scrips.loc[scrips["token"].isin(tokens)]
        symbol_token_pairs = [
            (token_df.loc[token_df["token"] == token, "symbol"].values[0], token)
            for token in tokens
        ]
        return symbol_token_pairs

    if expiry is None and strike is None and option_type is None:  # Cash segment

        if name in ["NIFTY", "BANKNIFTY"]:  # Index scrips
            filtered_scrips = scrips.loc[
                (scrips.name == name) & (scrips.exch_seg == "NSE") & (scrips.instrumenttype != "AMXIDX")
            ]   # Temp fix for AMXIDX
            # print(f'Length of filtered scrips: {len(filtered_scrips)}')
            assert len(filtered_scrips) == 1, "More than one index scrip found for name."
            symbol, token = filtered_scrips[["symbol", "token"]].values[0]

        elif name == "FINNIFTY":  # Finnifty temp fix
            futures = scrips.loc[
                (scrips.name == name) & (scrips.instrumenttype == "FUTIDX"),
                ["expiry", "symbol", "token"],
            ]
            futures["expiry"] = pd.to_datetime(futures["expiry"], format="%d%b%Y")
            futures = futures.sort_values(by="expiry")
            symbol, token = futures.iloc[0][["symbol", "token"]].values

        else:  # For all other equity scrips
            filtered_scrips = scrips.loc[
                (scrips.name == name)
                & (scrips.exch_seg == "NSE")
                & (scrips.symbol.str.endswith("EQ"))
            ]
            assert len(filtered_scrips) == 1, "More than one equity scrip found for name."
            symbol, token = filtered_scrips[["symbol", "token"]].values[0]

    elif expiry is not None and strike is not None and option_type is not None:  # Options segment
        strike = str(int(strike))  # Handle float strikes, convert to integer first
        symbol = name + expiry + strike + option_type
        token = scrips[scrips.symbol == symbol]["token"].tolist()[0]

    else:
        raise ValueError("Invalid arguments")

    return symbol, token


def get_straddle_symbol_tokens(name, strike, expiry):
    c_symbol, c_token = fetch_symbol_token(name, expiry, strike, "CE")
    p_symbol, p_token = fetch_symbol_token(name, expiry, strike, "PE")
    return c_symbol, c_token, p_symbol, p_token


def get_available_strikes(name, both_pairs=False):
    mask = (
        (scrips.name == name)
        & (scrips.exch_seg == "NFO")
        & (scrips.instrumenttype.str.startswith("OPT"))
    )
    filtered = scrips.loc[mask].copy()
    filtered["strike"] = filtered["strike"] / 100
    filtered_dict = (
        filtered.groupby("expiry")["strike"]
        .unique()
        .apply(list)
        .apply(sorted)
        .to_dict()
    )
    new_keys = map(
        lambda x: datetime.strptime(x, "%d%b%Y").strftime("%d%b%y").upper(),
        filtered_dict.keys(),
    )
    filtered_dict = {k: v for k, v in zip(new_keys, filtered_dict.values())}
    sorted_dict = {
        k: filtered_dict[k]
        for k in sorted(filtered_dict, key=lambda x: datetime.strptime(x, "%d%b%y"))
    }
    if not both_pairs:
        return sorted_dict

    def filter_dictionary(dictionary):
        pair_filtered_dict = {
            expiry: [strike for strike in strikes if check_strike(expiry, strike)]
            for expiry, strikes in dictionary.items()
        }
        return {key: values for key, values in pair_filtered_dict.items() if values}

    def check_strike(expiry, stk):
        try:
            return get_straddle_symbol_tokens(name, stk, expiry)
        except IndexError:
            # print(f"No straddle available for {name} {expiry} {stk}")
            return False

    return filter_dictionary(sorted_dict)


# LTP function
def fetchltp(exchange_seg, symbol, token):
    for attempt in range(1, 6):
        try:
            price = obj.ltpData(exchange_seg, symbol, token)["data"]["ltp"]
            return price
        except DataException:
            if attempt == 5:
                raise DataException("Failed to fetch LTP due to DataException")
            else:
                sleep(1)
                continue
        except Exception as e:
            if attempt == 5:
                raise Exception(f"Error in fetching LTP: {e}")
            else:
                print(f"Error {attempt} in fetching LTP: {e}")
                sleep(1)
                continue


def get_historical_prices(
        interval,
        last_n_intervals=None,
        from_date=None,
        to_date=None,
        token=None,
        name=None,
        expiry=None,
        strike=None,
        option_type=None
):

    """ Available intervals:

        ONE_MINUTE	1 Minute
        THREE_MINUTE 3 Minute
        FIVE_MINUTE	5 Minute
        TEN_MINUTE	10 Minute
        FIFTEEN_MINUTE	15 Minute
        THIRTY_MINUTE	30 Minute
        ONE_HOUR	1 Hour
        ONE_DAY	1 Day

        """

    if token is None and name is None:
        raise ValueError("Either name or token must be specified.")

    if last_n_intervals is None and from_date is None:
        raise ValueError("Either last_n_intervals or from_date must be specified.")

    if last_n_intervals is not None and from_date is not None:
        raise ValueError("Only one of last_n_intervals or from_date must be specified.")

    if to_date is None:
        to_date = last_market_close_time()
    else:
        to_date = pd.to_datetime(to_date)

    if from_date is None and last_n_intervals is not None:
        interval_digit, interval_unit = interval.lower().split("_")
        interval_unit = interval_unit + "s" if interval_unit[-1] != "s" else interval_unit
        interval_digit = word_to_num(interval_digit)
        time_delta = interval_digit*last_n_intervals
        from_date = to_date - timedelta(**{interval_unit: time_delta})
    else:
        from_date = pd.to_datetime(from_date)

    to_date = to_date.strftime("%Y-%m-%d %H:%M")
    from_date = from_date.strftime("%Y-%m-%d %H:%M")

    if token is None:
        _, token = fetch_symbol_token(name, expiry, strike, option_type)

    exchange_seg = scrips.loc[scrips.token == token, "exch_seg"].values[0]

    historic_param = {
        "exchange": exchange_seg,
        "symboltoken": token,
        "interval": interval,
        "fromdate": from_date,
        "todate": to_date,
    }
    data = obj.getCandleData(historic_param)
    data = pd.DataFrame(data["data"])
    data.set_index(pd.Series(data.iloc[:, 0], name='date'), inplace=True)
    data.index = pd.to_datetime(data.index)
    data.index = data.index.tz_localize(None)
    data.drop(data.columns[0], axis=1, inplace=True)
    data.columns = ["open", "high", "low", "close", "volume"]
    return data


def fetchpreviousclose(exchange_seg, symbol, token):
    for attempt in range(3):
        try:
            previousclose = obj.ltpData(exchange_seg, symbol, token)["data"]["close"]
            return previousclose
        except Exception as e:
            if attempt == 2:
                print(f"Error in fetchpreviousclose: {e}")
            else:
                print(
                    f"Error {attempt} in fetchpreviousclose: {e}\nRetrying again in 1 second"
                )
                sleep(1)


def fetch_straddle_price(name, expiry, strike, return_total_price=False):
    """Fetches the price of the straddle for a given name, expiry and strike. Expiry should be in the DDMMMYY format.
    If return_total_price is True, then the total price of the straddle is returned. If return_total_price is False,
    then the price of the call and put is returned as a tuple."""

    call_symbol, call_token = fetch_symbol_token(name, expiry, strike, "CE")
    put_symbol, put_token = fetch_symbol_token(name, expiry, strike, "PE")
    call_ltp = fetchltp("NFO", call_symbol, call_token)
    put_ltp = fetchltp("NFO", put_symbol, put_token)
    if return_total_price:
        return call_ltp + put_ltp
    else:
        return call_ltp, put_ltp


def fetch_strangle_price(
    name, expiry, call_strike, put_strike, return_total_price=False
):
    """Fetches the price of the strangle for a given name, expiry and strike. Expiry should be in the DDMMMYY format.
    If return_total_price is True, then the total price of the strangle is returned. If return_total_price is False,
    then the price of the call and put is returned as a tuple."""

    call_symbol, call_token = fetch_symbol_token(name, expiry, call_strike, "CE")
    put_symbol, put_token = fetch_symbol_token(name, expiry, put_strike, "PE")
    call_ltp = fetchltp("NFO", call_symbol, call_token)
    put_ltp = fetchltp("NFO", put_symbol, put_token)
    if return_total_price:
        return call_ltp + put_ltp
    else:
        return call_ltp, put_ltp


# Finding ATM strike
def findstrike(x, base):
    number = base * round(x / base)
    return int(number)


def custom_round(x, base=0.05):

    if x == 0:
        return 0

    num = base * round(x / base)
    if num == 0:
        num = base
    return round(num, 2)


def splice_orders(quantity_in_lots, freeze_qty):
    if quantity_in_lots > freeze_qty:
        loops = int(quantity_in_lots / freeze_qty)
        if loops > LARGE_ORDER_THRESHOLD:
            raise Exception(
                "Order too big. This error was raised to prevent accidental large order placement."
            )

        remainder = quantity_in_lots % freeze_qty
        if remainder == 0:
            spliced_orders = [freeze_qty] * loops
        else:
            spliced_orders = [freeze_qty] * loops + [remainder]
    else:
        spliced_orders = [quantity_in_lots]
    return spliced_orders


def check_for_weekend(expiry):
    expiry = datetime.strptime(expiry, "%d%b%y")
    expiry = expiry + pd.DateOffset(minutes=930)
    date_range = pd.date_range(currenttime().date(), expiry - timedelta(days=1))
    return date_range.weekday.isin([5, 6]).any()


def indices_to_trade(nifty, bnf, finnifty, multi_before_weekend=False):
    fin_exp_closer = timetoexpiry(
        finnifty.current_expiry, effective_time=True, in_days=True
    ) < timetoexpiry(nifty.current_expiry, effective_time=True, in_days=True)
    weekend_in_range = check_for_weekend(finnifty.current_expiry)
    if fin_exp_closer:
        if weekend_in_range and multi_before_weekend:
            return [nifty, finnifty]
        else:
            return [finnifty]
    return [nifty, bnf]


def timetoexpiry(expiry, effective_time=False, in_days=False):
    """Return time left to expiry"""
    if in_days:
        multiplier = 365
    else:
        multiplier = 1

    expiry = datetime.strptime(expiry, "%d%b%y")
    time_to_expiry = (
        (expiry + pd.DateOffset(minutes=930)) - currenttime()
    ) / timedelta(days=365)

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
        return bs.implied_volatility(opt_price, spot, strike, tte, 0.06, opt_type)
    except ValueError:
        return None


def straddle_iv(callprice, putprice, spot, strike, timeleft):
    call_iv = calculate_iv(callprice, spot, strike, timeleft, "CE")
    put_iv = calculate_iv(putprice, spot, strike, timeleft, "PE")

    if call_iv is not None and put_iv is not None:
        avg_iv = (call_iv + put_iv) / 2
    else:
        avg_iv = call_iv if put_iv is None else put_iv

    return call_iv, put_iv, avg_iv


def strangle_iv(callprice, putprice, spot, callstrike, putstrike, timeleft):
    call_iv = calculate_iv(callprice, spot, callstrike, timeleft, "CE")
    put_iv = calculate_iv(putprice, spot, putstrike, timeleft, "PE")

    if call_iv is not None and put_iv is not None:
        avg_iv = (call_iv + put_iv) / 2
    else:
        avg_iv = call_iv if put_iv is None else put_iv

    return call_iv, put_iv, avg_iv


def calc_combined_premium(
    spot, iv, time_left, strike=None, callstrike=None, putstrike=None
):
    if strike is None and (callstrike is None or putstrike is None):
        raise Exception("Either strike or callstrike and putstrike must be provided")

    if strike is not None and (callstrike is not None or putstrike is not None):
        raise Exception(
            "Strike Provided as well as callstrike and putstrike. Please provide only one of them"
        )

    if strike is not None:
        callstrike = strike
        putstrike = strike

    if time_left > 0:
        callprice = bs.call(spot, callstrike, time_left, 0.05, iv)
        putprice = bs.put(spot, putstrike, time_left, 0.05, iv)
        return callprice + putprice
    else:
        callpayoff = max(0, spot - callstrike)
        putpayoff = max(0, putstrike - spot)
        return callpayoff + putpayoff


def calc_greeks(position_string, position_price, underlying_price):
    """Fetches the price, iv and delta of a stock"""

    name, strike, expiry, option_type = position_string.split()
    strike = int(strike)
    time_left = timetoexpiry(expiry)

    iv = (
        bs.implied_volatility(
            position_price, underlying_price, strike, time_left, 0.05, option_type
        )
        * 100
    )
    delta = bs.delta(underlying_price, strike, time_left, 0.05, iv, option_type)
    gamma = bs.gamma(underlying_price, strike, time_left, 0.05, iv)

    return iv, delta, gamma


def most_equal_strangle(*strangles: Strangle):

    ltp_cache = {}
    call_set = set(strangle.call_option for strangle in strangles)
    put_set = set(strangle.put_option for strangle in strangles)
    union_set = call_set.union(put_set)
    for option in union_set:
        ltp_cache[option] = option.fetch_ltp()

    # Use the LTPs from the cache when calculating the price disparity
    def price_disparity(_strangle):
        call_ltp = ltp_cache[_strangle.call_option]
        put_ltp = ltp_cache[_strangle.put_option]
        disparity = abs(call_ltp - put_ltp) / min(call_ltp, put_ltp)
        return disparity
    return min(strangles, key=price_disparity)


def get_current_vix():
    vix = yf.Ticker("^INDIAVIX")
    vix = vix.fast_info["last_price"]
    return vix


def get_index_constituents(index_symbol, cutoff_pct=101):
    # Fetch and filter constituents
    constituents = (
        pd.read_csv(f"data/{index_symbol}_constituents.csv")
        .sort_values("Index weight", ascending=False)
        .assign(cum_weight=lambda df: df["Index weight"].cumsum())
        .loc[lambda df: df.cum_weight < cutoff_pct]
    )

    constituent_tickers, constituent_weights = (
        constituents.Ticker.to_list(),
        constituents["Index weight"].to_list(),
    )

    return constituent_tickers, constituent_weights


def convert_option_chains_to_df(option_chains, return_all=False, for_surface=False):

    def add_columns_for_surface(data_frame):

        data_frame = data_frame.copy()
        data_frame['atm_strike'] = data_frame.apply(
            lambda row: findstrike(row.spot, 50) if row.symbol == 'NIFTY' else findstrike(row.spot, 100),
            axis=1)
        data_frame['strike_iv'] = np.where(data_frame.strike > data_frame.atm_strike, data_frame.call_iv,
                                           np.where(data_frame.strike < data_frame.atm_strike,
                                                    data_frame.put_iv, data_frame.avg_iv))
        data_frame['atm_iv'] = data_frame.apply(
            lambda row: data_frame[(data_frame.strike == row.atm_strike)
                                   & (data_frame.expiry == row.expiry)].strike_iv.values[0], axis=1)
        data_frame.sort_values(['symbol', 'expiry', 'strike'], inplace=True)
        data_frame['distance'] = (data_frame['strike'] / data_frame['spot'] - 1)
        data_frame['iv_multiple'] = data_frame['strike_iv'] / data_frame['atm_iv']
        data_frame['distance_squared'] = data_frame['distance'] ** 2

        return data_frame

    symbol_dfs = []
    for symbol in option_chains:
        spot_price = option_chains[symbol].underlying_price
        expiry_dfs = []
        for expiry in option_chains[symbol]:
            df = pd.DataFrame(option_chains[symbol][expiry]).T
            df.index = df.index.set_names('strike')
            df = df.reset_index()
            df['spot'] = spot_price
            df['expiry'] = expiry
            df['symbol'] = symbol
            df['time_to_expiry'] = timetoexpiry(expiry)
            expiry_dfs.append(df)
        symbol_oc = pd.concat(expiry_dfs)
        if for_surface:
            symbol_oc = add_columns_for_surface(symbol_oc)
        symbol_dfs.append(symbol_oc)

    if return_all:
        return pd.concat(symbol_dfs)
    else:
        return symbol_dfs


def charges(buy_premium, contract_size, num_contracts, freeze_quantity=None):
    if freeze_quantity:
        number_of_orders = np.ceil(num_contracts / freeze_quantity)
    else:
        number_of_orders = 1

    buy_brokerage = 40 * number_of_orders
    sell_brokerage = 40 * number_of_orders
    transaction_charge_rate = 0.05 / 100
    stt_ctt_rate = 0.0625 / 100
    gst_rate = 18 / 100

    buy_transaction_charges = (
        buy_premium * contract_size * num_contracts * transaction_charge_rate
    )
    sell_transaction_charges = (
        buy_premium * contract_size * num_contracts * transaction_charge_rate
    )
    stt_ctt = buy_premium * contract_size * num_contracts * stt_ctt_rate

    buy_gst = (buy_brokerage + buy_transaction_charges) * gst_rate
    sell_gst = (sell_brokerage + sell_transaction_charges) * gst_rate

    total_charges = (
        buy_brokerage
        + sell_brokerage
        + buy_transaction_charges
        + sell_transaction_charges
        + stt_ctt
        + buy_gst
        + sell_gst
    )
    charges_per_share = total_charges / (num_contracts * contract_size)

    return round(charges_per_share, 1)


# ORDER FUNCTIONS BELOW #

def place_order(symbol, token, qty, action, price, order_tag="", stop_loss_order=False):

    action = action.upper()
    if isinstance(price, str):
        price = price.upper()

    params = {
        "tradingsymbol": symbol,
        "symboltoken": token,
        "transactiontype": action,
        "exchange": "NFO",
        "producttype": "CARRYFORWARD",
        "duration": "DAY",
        "quantity": int(qty),
        "ordertag": order_tag,
    }

    if stop_loss_order:
        execution_price = price * 1.1
        params.update({
            "variety": "STOPLOSS",
            "ordertype": "STOPLOSS_LIMIT",
            "triggerprice": round(price, 1),
            "price": round(execution_price, 1),
        })
    else:
        order_type, execution_price = ("MARKET", 0) if price == "MARKET" else ("LIMIT", price)
        if order_type == "LIMIT":
            if execution_price < 10 and qty < 6000:
                execution_price = np.ceil(price) if action == "BUY" else max(np.floor(price), 0.05)

        params.update({
            "variety": "NORMAL",
            "ordertype": order_type,
            "price": custom_round(execution_price)
        })

    for attempt in range(1, 4):
        try:
            return obj.placeOrder(params)
        except Exception as e:
            if attempt == 3:
                raise e
            print(f"Error {attempt} in placing {'stop-loss ' if stop_loss_order else ''}order for {symbol}: {e}")
            sleep(2)


def place_synthetic_fut_order(
    name,
    strike,
    expiry,
    buy_or_sell,
    quantity_in_lots,
    prices: str | tuple = "MARKET",
    stop_loss_order=False,
    order_tag="",
):
    """Places a synthetic future order. Quantity is in number of shares."""

    syn_fut = SyntheticFuture(strike, name, expiry)
    call_order_ids, put_order_ids = syn_fut.place_order(
        buy_or_sell, quantity_in_lots, prices, stop_loss_order, order_tag
    )
    return call_order_ids, put_order_ids


def cancel_pending_orders(order_ids, variety="STOPLOSS"):
    if isinstance(order_ids, (list, np.ndarray)):
        for order_id in order_ids:
            obj.cancelOrder(order_id, variety)
    else:
        obj.cancelOrder(order_ids, variety)
