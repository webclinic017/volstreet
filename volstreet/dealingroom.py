import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta, timezone
from time import sleep
import requests
import json
from SmartApi import SmartConnect
from SmartApi.smartExceptions import DataException
import pyotp
from threading import Thread
from volstreet.SmartWebSocketV2 import SmartWebSocketV2
from volstreet.constants import scrips, holidays, symbol_df, logger, token_symbol_dict
from volstreet import blackscholes as bs, datamodule as dm
from volstreet.exceptions import OptionModelInputError
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
ERROR_NOTIFICATION_SETTINGS = {"url": None}


def set_error_notification_settings(key, value):
    global ERROR_NOTIFICATION_SETTINGS
    ERROR_NOTIFICATION_SETTINGS[key] = value


def time_the_function(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = datetime.now()
        result = func(*args, **kwargs)
        end = (datetime.now() - start).total_seconds()
        logger.info(f"Time taken for {func.__name__}: {end:.2f} seconds")
        return result

    return wrapper


def log_errors(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            user_prefix = ERROR_NOTIFICATION_SETTINGS.get("user", "")
            logger.error(
                f"{user_prefix}Error in function {func.__name__}: {e}\nTraceback:{traceback.format_exc()}"
            )
            notifier(
                f"{user_prefix}Error in function {func.__name__}: {e}\nTraceback:{traceback.format_exc()}",
                ERROR_NOTIFICATION_SETTINGS["url"],
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
    def __init__(self, webhook_url=None, correlation_id="default"):
        global login_data, obj
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

    def start_websocket(self, tokens=None, mode=None, exchange_type=None):
        def _subscribe_tokens():
            self.subscribe_tokens(tokens, mode, exchange_type)

        def on_open(wsapp):
            _subscribe_tokens()

        # Assign the callbacks.
        self.on_open = on_open
        self.on_data = self.on_data_handler
        self.on_error = lambda wsapp, error: print(error)
        self.on_close = lambda wsapp: print("Close")
        Thread(target=self.connect).start()

    def subscribe_tokens(self, tokens=None, mode=None, exchange_type=None):
        if tokens is None:
            tokens = ["26000", "26009"]
        if mode is None:
            mode = 1
        if exchange_type is None:
            exchange_type = 1
        token_list = [{"exchangeType": exchange_type, "tokens": tokens}]
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
            token_symbol_dict[token]: value for token, value in self.price_dict.items()
        }
        new_price_dict.update(
            {"FINNIFTY": {"ltp": self.finnifty_index.fetch_ltp()}}
        )  # Finnifty temp fix
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

    @log_errors
    def update_option_chain(
        self,
        exit_time=(15, 30),
        process_iv_log=True,
        market_depth=True,
        calc_iv=True,
        stop_iv_calculation_hours=3,
        n_values=100,
    ):
        while currenttime().time() < time(*exit_time):
            self.build_all_option_chains(
                market_depth=market_depth,
                process_iv_log=process_iv_log,
                calc_iv=calc_iv,
                stop_iv_calculation_hours=stop_iv_calculation_hours,
                n_values=n_values,
            )

    def build_option_chain(
        self,
        index: str,
        expiry: str,
        market_depth: bool = False,
        process_iv_log: bool = False,
        calc_iv: bool = False,
        n_values: int = 100,
        stop_iv_calculation_hours: int = 3,
    ):
        parsed_dict = self.parse_price_dict()
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

                if calc_iv:
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
                            strike,
                            expiry,
                            call_iv,
                            put_iv,
                            avg_iv,
                            n_values,
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

    def build_all_option_chains(
        self,
        indices: list[str] | str | None = None,
        expiries: list[list[str]] | list[str] | str | None = None,
        market_depth: bool = False,
        process_iv_log: bool = False,
        calc_iv: bool = False,
        n_values: int = 100,
        stop_iv_calculation_hours: int = 3,
    ):
        if indices is None:
            indices = self.index_option_chains_subscribed
        elif isinstance(indices, str):
            indices = [indices]
        else:
            indices = indices
        if expiries is None:
            expiries = [
                set([*zip(*self.symbol_option_chains[index].exp_strike_pairs)][0])
                for index in indices
            ]
        elif isinstance(expiries, str):
            expiries = [[expiries]]
        elif all([isinstance(expiry, str) for expiry in expiries]):
            expiries = [expiries]
        else:
            expiries = expiries

        for index, exps in zip(indices, expiries):
            for expiry in exps:
                self.build_option_chain(
                    index,
                    expiry,
                    market_depth,
                    process_iv_log,
                    calc_iv,
                    n_values,
                    stop_iv_calculation_hours,
                )

    def process_iv_log(
        self,
        index,
        strike,
        expiry,
        call_iv,
        put_iv,
        avg_iv,
        n_values,
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

        option_type = option_type.upper()
        self.option_type = "CE" if option_type.startswith("C") else "PE"
        self.underlying = underlying.upper()
        self.expiry = expiry.upper()
        self.symbol, self.token = fetch_symbol_token(
            self.underlying, self.expiry, strike, self.option_type
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

    def place_order(
        self,
        transaction_type,
        quantity_in_lots,
        price="LIMIT",
        stop_loss_order=False,
        order_tag="",
    ):
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
                order_tag=order_tag,
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
        self.underlying = underlying.upper()
        self.underlying_exchange = (
            "NFO" if self.underlying in ["FINNIFTY", "MIDCPNIFTY"] else "NSE"
        )  # Fin/Mid temp fix
        self.expiry = expiry.upper()
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
        disparity = abs(call_ltp - put_ltp) / min(call_ltp, put_ltp)
        return disparity

    def fetch_symbol_token(self):
        return self.call_symbol, self.call_token, self.put_symbol, self.put_token

    def place_order(
        self,
        transaction_type,
        quantity_in_lots,
        prices="LIMIT",
        stop_loss_order=False,
        order_tag="",
    ):
        if stop_loss_order:
            assert isinstance(
                prices, (tuple, list, np.ndarray)
            ), "Prices must be a tuple of prices for stop loss order"
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
                raise ValueError(
                    "Prices must be either 'LIMIT' or 'MARKET' or a tuple of prices"
                )

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
                order_tag=order_tag,
            )
            put_order_id = place_order(
                self.put_symbol,
                self.put_token,
                qty * self.lot_size,
                transaction_type,
                put_price,
                stop_loss_order=stop_loss_order,
                order_tag=order_tag,
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
        self,
        transaction_type,
        quantity_in_lots,
        prices: str | tuple = "LIMIT",
        stop_loss_order=False,
        order_tag="",
    ):
        if isinstance(prices, (tuple, list, np.ndarray)):
            call_price, put_price = prices
        elif prices.upper() == "LIMIT":
            call_price, put_price = self.fetch_ltp()
            c_modifier, p_modifier = (
                (1.05, 0.95) if transaction_type.upper() == "BUY" else (0.95, 1.05)
            )
            call_price, put_price = call_price * c_modifier, put_price * p_modifier
        elif prices.upper() == "MARKET":
            call_price = put_price = prices
        else:
            raise ValueError(
                "Prices must be either 'LIMIT' or 'MARKET' or a tuple of prices"
            )

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
                order_tag=order_tag,
            )
            put_order_id = place_order(
                self.put_symbol,
                self.put_token,
                qty * self.lot_size,
                put_transaction_type,
                put_price,
                order_tag=order_tag,
            )
            call_order_ids.append(call_order_id)
            put_order_ids.append(put_order_id)
        return call_order_ids, put_order_ids


class SyntheticArbSystem:
    def __init__(self, symbol_option_chains):
        self.symbol_option_chains = symbol_option_chains

    def find_arbitrage_opportunities(
        self,
        index: str,
        expiry: str,
        qty_in_lots: int,
        exit_time=(15, 28),
        threshold=3,  # in points
    ):
        def get_single_index_single_expiry_data(_index, _expiry):
            option_chain = self.symbol_option_chains[_index][_expiry]
            _strikes = [_s for _s in option_chain]
            _call_prices = [option_chain[_s]["call_price"] for _s in _strikes]
            _put_prices = [option_chain[_s]["put_price"] for _s in _strikes]
            _call_bids = [option_chain[_s]["call_best_bid"] for _s in _strikes]
            _call_asks = [option_chain[_s]["call_best_ask"] for _s in _strikes]
            _put_bids = [option_chain[_s]["put_best_bid"] for _s in _strikes]
            _put_asks = [option_chain[_s]["put_best_ask"] for _s in _strikes]
            _call_bid_qty = [option_chain[_s]["call_best_bid_qty"] for _s in _strikes]
            _call_ask_qty = [option_chain[_s]["call_best_ask_qty"] for _s in _strikes]
            _put_bid_qty = [option_chain[_s]["put_best_bid_qty"] for _s in _strikes]
            _put_ask_qty = [option_chain[_s]["put_best_ask_qty"] for _s in _strikes]

            return (
                np.array(_strikes),
                np.array(_call_prices),
                np.array(_put_prices),
                np.array(_call_bids),
                np.array(_call_asks),
                np.array(_put_bids),
                np.array(_put_asks),
                np.array(_call_bid_qty),
                np.array(_call_ask_qty),
                np.array(_put_bid_qty),
                np.array(_put_ask_qty),
            )

        def return_both_side_synthetic_prices(
            _strikes, _call_asks, _put_bids, _call_bids, _put_asks
        ):
            return (_strikes + _call_asks - _put_bids), (
                _strikes + _call_bids - _put_asks
            )

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
        ) = get_single_index_single_expiry_data(index, expiry)
        synthetic_buy_prices, synthetic_sell_prices = return_both_side_synthetic_prices(
            strikes, call_asks, put_bids, call_bids, put_asks
        )
        min_price_index = np.argmin(synthetic_buy_prices)
        max_price_index = np.argmax(synthetic_sell_prices)
        min_price = synthetic_buy_prices[min_price_index]
        max_price = synthetic_sell_prices[max_price_index]

        last_print_time = currenttime()
        while currenttime().time() < time(*exit_time):
            if currenttime() > last_print_time + timedelta(seconds=5):
                print(
                    f"{currenttime()} - {index} - {expiry}:\n"
                    f"Minimum price: {min_price} at strike: {strikes[min_price_index]} "
                    f"Call Ask: {call_asks[min_price_index]} Put Bid: {put_bids[min_price_index]}\n"
                    f"Maximum price: {max_price} at strike: {strikes[max_price_index]} "
                    f"Call Bid: {call_bids[max_price_index]} Put Ask: {put_asks[max_price_index]}\n"
                    f"Price difference: {max_price - min_price}\n"
                )
                last_print_time = currenttime()

            if max_price - min_price > threshold:
                print(
                    f"**********Trade Identified at {currenttime()} on strike: Min {strikes[min_price_index]} "
                    f"and Max {strikes[max_price_index]}**********\n"
                    f"Minimum price: {min_price} at strike: {strikes[min_price_index]} "
                    f"Call Ask: {call_asks[min_price_index]} Put Bid: {put_bids[min_price_index]}\n"
                    f"Maximum price: {max_price} at strike: {strikes[max_price_index]} "
                    f"Call Bid: {call_bids[max_price_index]} Put Ask: {put_asks[max_price_index]}\n"
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
            (
                synthetic_buy_prices,
                synthetic_sell_prices,
            ) = return_both_side_synthetic_prices(
                strikes, call_asks, put_bids, call_bids, put_asks
            )
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
    ):
        ids_call_buy, ids_put_sell = place_synthetic_fut_order(
            index, buy_strike, expiry, "BUY", qty_in_lots, "MARKET"
        )
        ids_call_sell, ids_put_buy = place_synthetic_fut_order(
            index, sell_strike, expiry, "SELL", qty_in_lots, "MARKET"
        )
        ids = np.concatenate((ids_call_buy, ids_put_sell, ids_call_sell, ids_put_buy))

        sleep(1)
        statuses = lookup_and_return("orderbook", "orderid", ids, "status")

        if any(statuses == "rejected"):
            logger.error(
                f"Order rejected for {index} {expiry} {qty_in_lots} Buy {buy_strike} Sell {sell_strike}"
            )


class IvArbitrageScanner:
    def __init__(self, symbol_option_chains, iv_log):
        self.symbol_option_chains = symbol_option_chains
        self.iv_log = iv_log
        self.trade_log = []

    @log_errors
    def scan_for_iv_arbitrage(
        self, iv_hurdle=1.5, exit_time=(15, 25), notification_url=None
    ):
        while currenttime().time() < time(*exit_time):
            for index in self.symbol_option_chains:
                spot = self.symbol_option_chains[index].underlying_price
                for expiry in self.symbol_option_chains[index]:
                    for strike in self.symbol_option_chains[index][expiry]:
                        option_to_check = "avg"

                        # Check for IV spike
                        if spot < strike + 100:
                            option_to_check = "call"

                        if spot > strike - 100:
                            option_to_check = "put"

                        try:
                            opt_iv = self.symbol_option_chains[index][expiry][strike][
                                f"{option_to_check}_iv"
                            ]
                            running_avg_opt_iv = self.symbol_option_chains[index][
                                expiry
                            ][strike][f"running_avg_{option_to_check}_iv"]
                        except KeyError as e:
                            print(f"KeyError {e} for {index} {expiry} {strike}")
                            raise e

                        self.check_iv_spike(
                            opt_iv,
                            running_avg_opt_iv,
                            option_to_check.capitalize(),
                            index,
                            strike,
                            expiry,
                            iv_hurdle,
                            notification_url,
                        )

    def check_iv_spike(
        self,
        iv,
        running_avg_iv,
        opt_type,
        underlying,
        strike,
        expiry,
        iv_hurdle,
        notification_url,
    ):
        if opt_type == "Avg":
            return

        iv_hurdle = 1 + iv_hurdle
        upper_iv_threshold = running_avg_iv * iv_hurdle
        lower_iv_threshold = running_avg_iv / iv_hurdle

        # print(
        #    f"Checking {opt_type} IV for {underlying} {strike} {expiry}\nIV: {iv}\n"
        #    f"Running Average: {running_avg_iv}\nUpper Threshold: {upper_iv_threshold}\n"
        #    f"Lower Threshold: {lower_iv_threshold}"
        # )

        if iv and (iv > upper_iv_threshold or iv < lower_iv_threshold):
            # Execute trade
            # signal = "BUY" if iv > upper_iv_threshold else "SELL"
            # self.execute_iv_arbitrage_trade(
            #     signal, underlying, strike, expiry, opt_type
            # )

            # Notify
            if self.iv_log[underlying][expiry][strike][
                "last_notified_time"
            ] < currenttime() - timedelta(minutes=5):
                notifier(
                    f"{opt_type} IV for {underlying} {strike} {expiry} different from average.\nIV: {iv}\n"
                    f"Running Average: {running_avg_iv}",
                    notification_url,
                )
                self.iv_log[underlying][expiry][strike][
                    "last_notified_time"
                ] = currenttime()

    def execute_iv_arbitrage_trade(
        self, signal, underlying, strike, expiry, option_type
    ):
        qty_in_lots = 1
        option_to_trade = Option(strike, option_type, underlying, expiry)
        order_ids = option_to_trade.place_order(signal, qty_in_lots, "MARKET")
        self.trade_log.append(
            {
                "traded_option": option_to_trade,
                "order_ids": order_ids,
                "signal": signal,
                "qty": qty_in_lots,
                "order_type": "MARKET",
                "time": currenttime(),
            }
        )


class Index:
    """Initialize an index with the name of the index in uppercase"""

    def __init__(self, name, spot_future_rate=0.06):
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
        self.spot_future_rate = spot_future_rate
        self.symbol, self.token = fetch_symbol_token(self.name)
        self.lot_size = fetch_lot_size(self.name)
        self.fetch_expirys(self.symbol)
        self.freeze_qty = self.fetch_freeze_limit()
        self.available_strikes = None
        self.available_straddle_strikes = None
        self.intraday_straddle_forced_exit = False
        self.base = get_base(self.name)
        self.strategy_log = defaultdict(list)

        if self.name == "BANKNIFTY":
            self.exchange_type = 1
        elif self.name == "NIFTY":
            self.exchange_type = 1
        elif self.name in [
            "FINNIFTY",
            "MIDCPNIFTY",
        ]:  # Finnifty and Midcpnifty temp fix
            self.exchange_type = 2
        else:
            self.exchange_type = 1

        logger.info(
            f"Initialized {self.name} with lot size {self.lot_size}, base {self.base} and freeze qty {self.freeze_qty}"
        )

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
            logger.error(f"Timeout error in fetching freeze limit for {self.name}: {e}")
            freeze_qty_in_lots = 20
            return int(freeze_qty_in_lots)
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error in fetching freeze limit for {self.name}: {e}")
            freeze_qty_in_lots = 20
            return int(freeze_qty_in_lots)
        except Exception as e:
            logger.error(f"Error in fetching freeze limit for {self.name}: {e}")
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
        """Fetch LTP of the index. Uses futures for FINNIFTY and MIDCPNIFTY"""
        if self.name in ["FINNIFTY", "MIDCPNIFTY"]:  # Finnifty & Midcpnifty temp fix
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
        self,
        strike=None,
        call_strike=None,
        put_strike=None,
        expiry=None,
        buy_or_sell=None,
        call_price=None,
        put_price=None,
        order_tag=None,
    ):
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

    def place_synthetic_fut(
        self,
        strike,
        expiry,
        buy_or_sell,
        quantity_in_lots,
        prices="LIMIT",
        stop_loss_order=False,
        order_tag="",
    ):
        return place_synthetic_fut_order(
            self.name,
            strike,
            expiry,
            buy_or_sell,
            quantity_in_lots,
            prices,
            stop_loss_order,
            order_tag,
        )

    @time_the_function
    def find_equal_strike(
        self,
        exit_time,
        websocket,
        wait_for_equality,
        target_disparity,
        expiry=None,
        notification_url=None,
    ):
        expiry = expiry or self.current_expiry
        ltp = (
            self.fetch_ltp()
            if not websocket
            else websocket.price_dict.get(self.token, 0)["ltp"]
        )
        current_strike = findstrike(ltp, self.base)
        strike_range = np.arange(
            current_strike - self.base * 2, current_strike + self.base * 3, self.base
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
                    notifier(message, notification_url)
                    last_notify_time = currenttime()

                if (currenttime() + timedelta(minutes=5)).time() > time(*exit_time):
                    notifier(
                        "Equal strike tracker exited due to time limit.",
                        notification_url,
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

    @time_the_function
    def most_equal_strangle(
        self,
        call_strike_offset=0,
        put_strike_offset=0,
        disparity_threshold=np.inf,
        exit_time=time(15, 25),
        range_of_strikes=4,
        expiry=None,
    ) -> Strangle | None:
        def get_range_of_strangles(c_strike, p_strike, exp, strike_range):
            if strike_range % 2 != 0:
                strike_range += 1
            c_strike_range = np.arange(
                c_strike - (strike_range / 2) * self.base,
                c_strike + (strike_range / 2) * self.base + self.base,
                self.base,
            )
            if c_strike == p_strike:
                return [Straddle(strike, self.name, exp) for strike in c_strike_range]
            else:
                p_strike_ranges = np.arange(
                    p_strike - (strike_range / 2) * self.base,
                    p_strike + (strike_range / 2) * self.base + self.base,
                    self.base,
                )
                pairs = itertools.product(c_strike_range, p_strike_ranges)
                return [Strangle(pair[0], pair[1], self.name, exp) for pair in pairs]

        if expiry is None:
            expiry = self.current_expiry

        underlying_ltp = self.fetch_ltp()
        temp_call_strike = underlying_ltp * (1 + call_strike_offset)
        temp_put_strike = underlying_ltp * (1 - put_strike_offset)
        temp_call_strike = findstrike(temp_call_strike, self.base)
        temp_put_strike = findstrike(temp_put_strike, self.base)

        strangles = get_range_of_strangles(
            temp_call_strike, temp_put_strike, expiry, range_of_strikes
        )
        logger.info(f"{self.name} prospective strangles: {strangles}")

        # Create a set of all distinct options
        options = set(
            option
            for strangle in strangles
            for option in (strangle.call_option, strangle.put_option)
        )

        # Define the price disparity function
        def price_disparity(strangle):
            call_ltp = ltp_cache[strangle.call_option]
            put_ltp = ltp_cache[strangle.put_option]
            return abs(call_ltp - put_ltp) / min(call_ltp, put_ltp)

        tracked_strangle = None

        while currenttime().time() < exit_time:
            # If there's no tracked strangle update all prices and find the most equal strangle
            if tracked_strangle is None:
                ltp_cache = {option: option.fetch_ltp() for option in options}
                most_equal, min_disparity = min(
                    ((s, price_disparity(s)) for s in strangles), key=lambda x: x[1]
                )
                if min_disparity < 0.10:
                    tracked_strangle = most_equal

            # If there's a tracked strangle, check its disparity
            else:
                ltp_cache = {
                    tracked_strangle.call_option: tracked_strangle.call_option.fetch_ltp(),
                    tracked_strangle.put_option: tracked_strangle.put_option.fetch_ltp(),
                }
                most_equal = tracked_strangle
                min_disparity = price_disparity(tracked_strangle)
                if min_disparity >= 0.10:
                    tracked_strangle = None

            logger.info(
                f"Most equal strangle: {most_equal} with disparity {min_disparity} "
                f"and prices {ltp_cache[most_equal.call_option]} and {ltp_cache[most_equal.put_option]}"
            )
            logger.info(f"Most equal ltp cache: {ltp_cache}")
            # If the lowest disparity is below the threshold, return the most equal strangle
            if min_disparity < disparity_threshold:
                return most_equal
            else:
                pass

        else:
            return None

    @time_the_function
    def most_resilient_strangle(
        self,
        strike_range=40,
        stop_loss=1.5,
        time_delta_minutes=60,
        expiry=None,
        extra_buffer=1.07,
    ) -> Strangle:
        def expected_movement(option: Option):
            option_ltp = ltp_cache[option]
            time_to_expiry = timetoexpiry(expiry)
            stop_loss_price = option_ltp * stop_loss
            return bs.target_movement(
                flag=option.option_type,
                starting_price=option_ltp,
                target_price=stop_loss_price,
                starting_spot=spot_price,
                strike=option.strike,
                time_left=time_to_expiry,
                time_delta_minutes=time_delta_minutes,
                symbol=self.name,
            )

        def find_favorite_strike(expected_moves, options, benchmark_movement):
            for i in range(1, len(expected_moves)):
                if (  # Remove hardcoded 20% buffer
                    expected_moves[i] > benchmark_movement * extra_buffer
                    and expected_moves[i] > expected_moves[i - 1]
                ):
                    return options[i]
            return None

        if expiry is None:
            expiry = self.current_expiry

        spot_price = self.fetch_ltp()
        atm_strike = findstrike(spot_price, self.base)

        half_range = int(strike_range / 2)
        strike_range = np.arange(
            atm_strike - (self.base * half_range),
            atm_strike + (self.base * (half_range + 1)),
            self.base,
        )

        options_by_type = {
            "CE": [
                Option(
                    strike=strike, option_type="CE", underlying=self.name, expiry=expiry
                )
                for strike in strike_range
                if strike >= atm_strike
            ],
            "PE": [
                Option(
                    strike=strike, option_type="PE", underlying=self.name, expiry=expiry
                )
                for strike in strike_range[::-1]
                if strike <= atm_strike
            ],
        }

        ltp_cache = {
            option: option.fetch_ltp()
            for option_type in options_by_type
            for option in options_by_type[option_type]
        }

        expected_movements = {
            option_type: [expected_movement(option) for option in options]
            for option_type, options in options_by_type.items()
        }

        expected_movements_ce = np.array(expected_movements["CE"])
        expected_movements_pe = np.array(expected_movements["PE"])
        expected_movements_pe = expected_movements_pe * -1

        benchmark_movement_ce = expected_movements_ce[0]
        benchmark_movement_pe = expected_movements_pe[0]

        logger.info(
            f"{self.name} - Call options' expected movements: {list(zip(options_by_type['CE'], expected_movements_ce))}"
        )
        logger.info(
            f"{self.name} - Put options' expected movements:{list(zip(options_by_type['PE'], expected_movements_pe))}"
        )

        favorite_strike_ce = (
            find_favorite_strike(
                expected_movements_ce,
                options_by_type["CE"],
                benchmark_movement_ce,
            )
            or options_by_type["CE"][0]
        )  # If no favorite strike, use ATM strike
        favorite_strike_pe = (
            find_favorite_strike(
                expected_movements_pe,
                options_by_type["PE"],
                benchmark_movement_pe,
            )
            or options_by_type["PE"][0]
        )  # If no favorite strike, use ATM strike

        ce_strike = favorite_strike_ce.strike
        pe_strike = favorite_strike_pe.strike
        strangle = Strangle(ce_strike, pe_strike, self.name, expiry)

        return strangle

    @log_errors
    def rollover_overnight_short_straddle(
        self,
        quantity_in_lots,
        strike_offset=1,
        iv_threshold=0.95,
        take_avg_price=False,
        notification_url=None,
    ):
        """Rollover overnight short straddle to the next expiry.
        Args:
            quantity_in_lots (int): Quantity of the straddle in lots.
            strike_offset (float): Strike offset from the current strike.
            iv_threshold (float): IV threshold compared to vix.
            take_avg_price (bool): Take average price of the index over 5m timeframes.
            notification_url (str): Webhook URL to send notifications.
        """

        def load_data():
            try:
                with open(f"{obj.userId}_overnight_positions.json", "r") as f:
                    data = json.load(f)
                    return data
            except FileNotFoundError:
                data = {}
                notifier(
                    "No positions found for overnight straddle. Creating new file.",
                    notification_url,
                )
                with open(f"{obj.userId}_overnight_positions.json", "w") as f:
                    json.dump(data, f)
                return data
            except Exception as e:
                notifier(
                    f"Error while reading overnight_positions.json: {e}",
                    notification_url,
                )
                logger.error(
                    f"Error while reading positions.json",
                    exc_info=(type(e), e, e.__traceback__),
                )
                raise Exception("Error while reading positions.json")

        def save_data(data):
            with open(f"{obj.userId}_overnight_positions.json", "w") as f:
                json.dump(data, f)

        avg_ltp = None
        if take_avg_price:
            if currenttime().time() < time(15, 00):
                notifier(
                    f"{self.name} Cannot take avg price before 3pm. Try running the strategy after 3pm",
                    notification_url,
                )
                raise Exception(
                    "Cannot take avg price before 3pm. Try running the strategy after 3pm"
                )
            notifier(
                f"{self.name} Taking average price of the index over 5m timeframes.",
                notification_url,
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
            beta = dm.get_summary_ratio(
                self.name, "NIFTY"
            )  # beta of the index vs nifty since vix is of nifty
            beta = 1.3 if beta is None else beta
            vix = vix * beta

        order_tag = "Overnight short straddle"

        weekend_in_expiry = check_for_weekend(self.current_expiry)
        ltp = avg_ltp if avg_ltp else self.fetch_ltp()
        sell_strike = findstrike(ltp * strike_offset, self.base)
        sell_straddle = Straddle(
            strike=sell_strike, underlying=self.name, expiry=self.current_expiry
        )
        call_iv, put_iv, iv = sell_straddle.ivs()
        iv = iv * 100 if iv is not None else None
        # This if-clause checks how far the expiry is
        if weekend_in_expiry:  # far from expiry
            if iv is not None:
                if iv < vix * iv_threshold:
                    notifier(
                        f"{self.name} IV is too low compared to VIX - IV: {iv}, Vix: {vix}.",
                        notification_url,
                    )
                    return
                else:
                    notifier(
                        f"{self.name} IV is fine compared to VIX - IV: {iv}, Vix: {vix}.",
                        notification_url,
                    )
            else:
                notifier(
                    f"{self.name} IV is None and weekend before expiry. Exiting. Vix: {vix}.",
                )
                return
        elif (
            timetoexpiry(self.current_expiry, effective_time=True, in_days=True) < 1.5
        ):  # only exit as expiry next day
            sell_strike = None
            notifier(
                f"{self.name} Only exiting current position. IV: {iv}, Vix: {vix}.",
                notification_url,
            )
        else:
            notifier(
                f"{self.name} Deploying overnight straddle - IV: {iv}, Vix: {vix}.",
                notification_url,
            )

        trade_data = load_data()
        buy_strike = trade_data.get(self.name, None)
        buy_straddle = (
            Straddle(
                strike=buy_strike, underlying=self.name, expiry=self.current_expiry
            )
            if buy_strike
            else None
        )

        # Checking if the buy strike is valid
        if (
            not isinstance(buy_strike, int)
            and not isinstance(buy_strike, float)
            and buy_strike is not None
        ):
            notifier(f"Invalid strike found for {self.name}.", notification_url)
            raise Exception(f"Invalid strike found for {self.name}.")

        trade_info_dict = {
            "Date": currenttime().strftime("%d-%m-%Y %H:%M:%S"),
            "Underlying": self.name,
            "Expiry": self.current_expiry,
        }

        call_buy_avg, put_buy_avg = np.nan, np.nan
        call_sell_avg, put_sell_avg = np.nan, np.nan

        # Placing orders
        if buy_strike is None and sell_strike is None:
            notifier(f"{self.name} No trade required.", notification_url)
        elif sell_strike is None:  # only exiting current position
            notifier(
                f"{self.name} Exiting current position on strike {buy_strike}.",
                notification_url,
            )
            call_buy_avg, put_buy_avg = place_option_order_and_notify(
                buy_straddle,
                "BUY",
                quantity_in_lots,
                "LIMIT",
                order_tag=order_tag,
                webhook_url=notification_url,
                return_avg_price=True,
            )

        elif buy_strike is None:  # only entering new position
            notifier(
                f"{self.name} Entering new position on strike {sell_strike}.",
                notification_url,
            )
            call_sell_avg, put_sell_avg = place_option_order_and_notify(
                sell_straddle,
                "SELL",
                quantity_in_lots,
                "LIMIT",
                order_tag=order_tag,
                webhook_url=notification_url,
                return_avg_price=True,
            )

        else:  # both entering and exiting positions
            if buy_strike == sell_strike:
                notifier(
                    f"{self.name} No trade required as strike is same.",
                    notification_url,
                )
                call_ltp, put_ltp = sell_straddle.fetch_ltp()
                call_buy_avg, put_buy_avg, call_sell_avg, put_sell_avg = (
                    call_ltp,
                    put_ltp,
                    call_ltp,
                    put_ltp,
                )
            else:
                notifier(
                    f"{self.name} Buying {buy_strike} and selling {sell_strike}.",
                    notification_url,
                )
                call_buy_avg, put_buy_avg = place_option_order_and_notify(
                    buy_straddle,
                    "BUY",
                    quantity_in_lots,
                    "LIMIT",
                    order_tag=order_tag,
                    webhook_url=notification_url,
                    return_avg_price=True,
                )
                call_sell_avg, put_sell_avg = place_option_order_and_notify(
                    sell_straddle,
                    "SELL",
                    quantity_in_lots,
                    "LIMIT",
                    order_tag=order_tag,
                    webhook_url=notification_url,
                    return_avg_price=True,
                )

        trade_info_dict.update(
            {
                "Buy Strike": buy_strike,
                "Buy Call Price": call_buy_avg,
                "Buy Put Price": put_buy_avg,
                "Buy Total Price": call_buy_avg + put_buy_avg,
                "Sell Strike": sell_strike,
                "Sell Call Price": call_sell_avg,
                "Sell Put Price": put_sell_avg,
                "Sell Total Price": call_sell_avg + put_sell_avg,
            }
        )

        trade_data[self.name] = sell_strike
        save_data(trade_data)

        self.strategy_log[order_tag].append(trade_info_dict)

    @log_errors
    def buy_weekly_hedge(
        self,
        quantity_in_lots,
        type_of_hedge="strangle",
        strike_offset=1,
        call_offset=1,
        put_offset=1,
        notification_url=None,
    ):
        order_tag = "Weekly hedge"

        ltp = self.fetch_ltp()
        if type_of_hedge == "strangle":
            call_strike = findstrike(ltp * call_offset, self.base)
            put_strike = findstrike(ltp * put_offset, self.base)
            strike = None
            instrument = Strangle(call_strike, put_strike, self.name, self.next_expiry)
        elif type_of_hedge == "straddle":
            strike = findstrike(ltp * strike_offset, self.base)
            call_strike = None
            put_strike = None
            instrument = Straddle(strike, self.name, self.next_expiry)
        else:
            raise Exception("Invalid type of hedge.")

        call_buy_avg, put_buy_avg = place_option_order_and_notify(
            instrument,
            "BUY",
            quantity_in_lots,
            "LIMIT",
            order_tag=order_tag,
            webhook_url=notification_url,
            return_avg_price=True,
        )

        trade_info_dict = {
            "Date": currenttime().strftime("%d-%m-%Y %H:%M:%S"),
            "Underlying": self.name,
            "Expiry": self.next_expiry,
            "Buy Strike(s)": strike
            if strike is not None
            else (call_strike, put_strike),
            "Buy Call Price": call_buy_avg,
            "Buy Put Price": put_buy_avg,
            "Buy Total Price": call_buy_avg + put_buy_avg,
        }

        self.strategy_log[order_tag].append(trade_info_dict)

    @log_errors
    def intraday_strangle(
        self,
        quantity_in_lots,
        call_strike_offset=0,
        put_strike_offset=0,
        strike_selection="equal",
        stop_loss="dynamic",
        call_stop_loss=None,
        put_stop_loss=None,
        exit_time=(15, 29),
        sleep_time=5,
        catch_trend=False,
        trend_qty_ratio=1,
        trend_strike_offset=0,
        trend_sl=0.003,
        disparity_threshold=np.inf,
        place_sl_orders=False,
        move_sl_to_cost=False,
        convert_to_butterfly=False,
        conversion_method="breakeven",
        conversion_threshold_pct=0.175,
        shared_data=None,
        notification_url=None,
    ):
        """Intraday strangle strategy. Trades strangle with stop loss. All offsets are in percentage terms.
        Parameters
        ----------
        quantity_in_lots : int
            Quantity in lots
        strike_selection : str, optional {'equal', 'resilient'}
            Mode for finding the strangle, by default 'equal'
        call_strike_offset : float, optional
            Call strike offset in percentage terms, by default 0
        put_strike_offset : float, optional
            Put strike offset in percentage terms, by default 0
        stop_loss : float or string, optional
            Stop loss percentage, by default 'dynamic'
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
        disparity_threshold : float, optional
            Disparity threshold for equality of strikes, by default np.inf
        place_sl_orders : bool, optional
            Place stop loss orders or not, by default False
        move_sl_to_cost : bool, optional
            Move other stop loss to cost or not, by default False
        convert_to_butterfly : bool, optional
            Convert to butterfly or not, by default False
        conversion_method : str, optional
            Conversion method for butterfly, by default 'breakeven'
        conversion_threshold_pct : float, optional
            Conversion threshold for butterfly if conversion method is 'pct', by default 0.175
        shared_data : SharedData class, optional
            shared data about client level orderbook and positions, by default None
        notification_url : str, optional
            URL for sending notifications, by default None
        """

        @log_errors
        def position_monitor(info_dict):
            c_avg_price = info_dict["call_avg_price"]
            p_avg_price = info_dict["put_avg_price"]
            traded_strangle = info_dict["traded_strangle"]

            # Price deque
            n_prices = max(
                int(30 / sleep_time), 1
            )  # Hard coded 30-second price window for now
            last_n_prices = {
                "call": deque(maxlen=n_prices),
                "put": deque(maxlen=n_prices),
                "underlying": deque(maxlen=n_prices),
            }

            # Conversion to butterfly
            ctb_notification_sent = False
            ctb_message = ""
            ctb_hedge = None
            conversion_threshold_break_even = None

            def process_ctb(
                h_strangle: Strangle,
                method: str,
                threshold_break_even: float | None,
                threshold_pct: float,
                total_price: float,
            ) -> bool:
                hedge_total_ltp = h_strangle.fetch_total_ltp()

                if method == "breakeven":
                    hedge_profit = (
                        info_dict["total_avg_price"] - hedge_total_ltp - self.base
                    )
                    print(
                        f"Checking breakeven method: {hedge_profit} >= {threshold_break_even}"
                    )
                    return hedge_profit >= threshold_break_even

                elif method == "pct":
                    print(
                        f"Checking pct method: {hedge_total_ltp} <= {total_price*threshold_pct}"
                    )
                    return hedge_total_ltp <= total_price * threshold_pct

                else:
                    raise ValueError(
                        f"Invalid conversion method: {method}. Valid methods are 'breakeven' and 'pct'."
                    )

            if convert_to_butterfly:
                ctb_call_strike = traded_strangle.call_strike + self.base
                ctb_put_strike = traded_strangle.put_strike - self.base
                ctb_hedge = Strangle(ctb_call_strike, ctb_put_strike, self.name, expiry)
                if conversion_method == "breakeven":
                    c_sl = call_stop_loss if call_stop_loss is not None else stop_loss
                    p_sl = put_stop_loss if put_stop_loss is not None else stop_loss
                    profit_if_call_sl = p_avg_price - (c_avg_price * (c_sl - 1))
                    profit_if_put_sl = c_avg_price - (p_avg_price * (p_sl - 1))

                    conversion_threshold_break_even = max(
                        profit_if_call_sl, profit_if_put_sl
                    )
                elif conversion_method == "pct":
                    conversion_threshold_break_even = None
                else:
                    raise ValueError(
                        f"Invalid conversion method: {conversion_method}. Valid methods are 'breakeven' and 'pct'."
                    )

            last_print_time = currenttime()
            last_log_time = currenttime()
            last_notify_time = currenttime()
            print_interval = timedelta(seconds=10)
            log_interval = timedelta(minutes=25)
            notify_interval = timedelta(minutes=180)

            while not info_dict["trade_complete"]:
                # Fetching prices
                spot_price = self.fetch_ltp()
                c_ltp, p_ltp = traded_strangle.fetch_ltp()
                info_dict["underlying_ltp"] = spot_price
                info_dict["call_ltp"] = c_ltp
                info_dict["put_ltp"] = p_ltp
                last_n_prices["call"].append(c_ltp)
                last_n_prices["put"].append(p_ltp)
                last_n_prices["underlying"].append(spot_price)
                c_ltp_avg = (
                    sum(last_n_prices["call"]) / len(last_n_prices["call"])
                    if last_n_prices["call"]
                    else c_ltp
                )
                p_ltp_avg = (
                    sum(last_n_prices["put"]) / len(last_n_prices["put"])
                    if last_n_prices["put"]
                    else p_ltp
                )
                spot_price_avg = (
                    sum(last_n_prices["underlying"]) / len(last_n_prices["underlying"])
                    if last_n_prices["underlying"]
                    else spot_price
                )
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
                    timeleft=timetoexpiry(expiry),
                )
                info_dict["call_iv"] = call_iv
                info_dict["put_iv"] = put_iv
                info_dict["avg_iv"] = avg_iv

                # Calculate mtm price
                call_exit_price = info_dict.get("call_exit_price", c_ltp)
                put_exit_price = info_dict.get("put_exit_price", p_ltp)
                mtm_price = call_exit_price + put_exit_price

                # Calculate profit
                profit_in_pts = (c_avg_price + p_avg_price) - mtm_price
                profit_in_rs = profit_in_pts * self.lot_size * quantity_in_lots
                info_dict["profit_in_pts"] = profit_in_pts
                info_dict["profit_in_rs"] = profit_in_rs

                # Conversion to butterfly working
                if (
                    not (info_dict["call_sl"] or info_dict["put_sl"])
                    and info_dict["time_left_day_start"] * 365 < 1
                    and convert_to_butterfly
                    and not ctb_notification_sent
                ):
                    try:
                        ctb_trigger = process_ctb(
                            ctb_hedge,
                            conversion_method,
                            conversion_threshold_break_even,
                            conversion_threshold_pct,
                            info_dict["total_avg_price"],
                        )
                        if ctb_trigger:
                            notifier(
                                f"{self.name} Convert to butterfly triggered\n",
                                notification_url,
                            )
                            info_dict["exit_triggers"].update(
                                {"convert_to_butterfly": True}
                            )
                            ctb_message = f"Hedged with: {ctb_hedge}\n"
                            info_dict["ctb_hedge"] = ctb_hedge
                            ctb_notification_sent = True
                    except Exception as _e:
                        logger.error(f"Error in process_ctb: {_e}")

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
                    f"Profit: {info_dict['profit_in_rs']:.2f}\n" + ctb_message
                )
                if currenttime() - last_print_time > print_interval:
                    print(message)
                    last_print_time = currenttime()
                if currenttime() - last_log_time > log_interval:
                    logger.info(message)
                    last_log_time = currenttime()
                if currenttime() - last_notify_time > notify_interval:
                    notifier(message, notification_url)
                    last_notify_time = currenttime()
                sleep(sleep_time)

        @log_errors
        def trend_catcher(info_dict, sl_type, qty_ratio, sl, strike_offset):
            offset = 1 - strike_offset if sl_type == "call" else 1 + strike_offset

            spot_price = info_dict["underlying_ltp"]

            # Setting up the trend option
            strike = spot_price * offset
            strike = findstrike(strike, self.base)
            opt_type = "PE" if sl_type == "call" else "CE"
            qty_in_lots = max(int(quantity_in_lots * qty_ratio), 1)
            trend_option = Option(strike, opt_type, self.name, expiry)

            # Placing the trend option order
            place_option_order_and_notify(
                trend_option,
                "SELL",
                qty_in_lots,
                "LIMIT",
                "Intraday Strangle Trend Catcher",
                notification_url,
            )

            # Setting up the stop loss
            sl_multiplier = 1 - sl if sl_type == "call" else 1 + sl
            sl_price = spot_price * sl_multiplier
            trend_sl_hit = False

            notifier(
                f"{self.name} strangle {sl_type} trend catcher starting. "
                + f"Placed {qty_in_lots} lots of {strike} {opt_type} at {trend_option.fetch_ltp()}. "
                + f"Stoploss price: {sl_price}, Underlying Price: {spot_price}",
                notification_url,
            )

            last_print_time = currenttime()
            print_interval = timedelta(seconds=10)
            while all(
                [
                    currenttime().time() < time(*exit_time),
                    not info_dict["trade_complete"],
                ]
            ):
                spot_price = info_dict["underlying_ltp"]
                spot_price_avg = info_dict["underlying_ltp_avg"]
                trend_sl_hit = (
                    spot_price_avg < sl_price
                    if sl_type == "call"
                    else spot_price_avg > sl_price
                )
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
                    f"{self.name} strangle {sl_type} trend catcher stoploss hit.",
                    notification_url,
                )
            else:
                notifier(
                    f"{self.name} strangle {sl_type} trend catcher exiting.",
                    notification_url,
                )

            # Buying the trend option back
            place_option_order_and_notify(
                trend_option,
                "BUY",
                qty_in_lots,
                "LIMIT",
                "Intraday Strangle Trend Catcher",
                notification_url,
            )

        def justify_stop_loss(info_dict, side):
            entry_spot = info_dict.get("spot_at_entry")
            current_spot = info_dict.get("underlying_ltp")

            # If the spot has moved in the direction of stop loss
            time_left_day_start = info_dict.get("time_left_day_start")
            time_left_now = timetoexpiry(expiry)
            time_delta = (time_left_day_start - time_left_now) * 525600
            time_delta = int(time_delta)
            try:
                estimated_movement = bs.target_movement(
                    flag=side,
                    starting_price=info_dict.get(f"{side}_avg_price"),
                    target_price=info_dict.get(f"{side}_stop_loss_price"),
                    starting_spot=entry_spot,
                    strike=info_dict.get("traded_strangle").call_strike
                    if side == "call"
                    else info_dict.get("traded_strangle").put_strike,
                    time_left=time_left_day_start,
                    time_delta_minutes=time_delta,
                )
            except OptionModelInputError:
                estimated_movement = (
                    0.0022 if side == "call" else -0.0022  # Remove hard coded number
                )
                input_error_message = (
                    f"OptionModelInputError in justify_stop_loss for {self.name} {side} strangle\n"
                    f"Setting estimated_movement to {estimated_movement}\n"
                    f"Input flag: {side}, "
                    f"Input stop loss price: {info_dict.get(f'{side}_stop_loss_price')}, "
                    f"Input avg price: {info_dict.get(f'{side}_avg_price')}, "
                    f"Input entry spot: {entry_spot}"
                )
                logger.error(input_error_message)
                notifier(input_error_message, notification_url)
            except Exception as e:
                estimated_movement = (
                    0.0022 if side == "call" else -0.0022  # Remove hard coded number
                )
                error_message = (
                    f"Error in justify_stop_loss for {self.name} {side} strangle\n"
                    f"Setting estimated_movement to {estimated_movement}\n"
                    f"Error: {e}"
                )
                logger.error(error_message)
                notifier(error_message, notification_url)

            actual_movement = (current_spot - entry_spot) / entry_spot
            difference_in_sign = np.sign(estimated_movement) != np.sign(actual_movement)
            lack_of_movement = abs(actual_movement) < 0.8 * abs(estimated_movement)
            # Remove hard coded number.
            if difference_in_sign or lack_of_movement:
                if not info_dict.get(f"{side}_sl_check_notification_sent"):
                    message = (
                        f"{self.name} strangle {side} stop loss appears to be unjustified. "
                        f"Estimated movement: {estimated_movement}, Actual movement: {actual_movement}"
                    )
                    notifier(message, notification_url)
                    info_dict[f"{side}_sl_check_notification_sent"] = True
                return False
            else:
                message = (
                    f"{self.name} strangle {side} stop loss triggered. "
                    f"Estimated movement: {estimated_movement}, Actual movement: {actual_movement}"
                )
                notifier(message, notification_url)
                return True

        def check_for_stop_loss(info_dict, side, refresh_orderbook=False):
            """Check for stop loss."""

            stop_loss_order_ids = info_dict.get(f"{side}_stop_loss_order_ids")

            if stop_loss_order_ids is None:  # If stop loss order ids are not provided
                avg_price = info_dict.get(f"{side}_ltp_avg")
                stop_loss_price = info_dict.get(f"{side}_stop_loss_price")
                stop_loss_triggered = avg_price > stop_loss_price
                if stop_loss_triggered:
                    stop_loss_justified = justify_stop_loss(info_dict, side)
                    price_increase = avg_price / stop_loss_price
                    if (
                        stop_loss_justified or price_increase > 1.5
                    ):  # Remove hard coded safety number
                        info_dict[f"{side}_sl"] = True

            else:  # If stop loss order ids are provided
                orderbook = fetch_orderbook_if_needed(
                    shared_data, refresh_needed=refresh_orderbook
                )
                if shared_data is None:
                    sleep(3)
                orders_triggered, orders_complete = process_stop_loss_order_statuses(
                    orderbook,
                    stop_loss_order_ids,
                    context=side,
                    notify_url=notification_url,
                )
                if orders_triggered:
                    justify_stop_loss(info_dict, side)
                    info_dict[f"{side}_sl"] = True
                    if not orders_complete:
                        info_dict[f"{side}_stop_loss_order_ids"] = None

        def process_stop_loss(info_dict, sl_type):
            if (
                info_dict["call_sl"] and info_dict["put_sl"]
            ):  # Check to avoid double processing
                return

            traded_strangle = info_dict["traded_strangle"]
            other_side: str = "call" if sl_type == "put" else "put"

            # Buying the stop loss option back if it is not already bought
            if info_dict[f"{sl_type}_stop_loss_order_ids"] is None:
                option_to_buy = (
                    traded_strangle.call_option
                    if sl_type == "call"
                    else traded_strangle.put_option
                )
                exit_price = place_option_order_and_notify(
                    option_to_buy,
                    "BUY",
                    quantity_in_lots,
                    "LIMIT",
                    order_tag,
                    notification_url,
                )
            else:
                orderbook = fetch_book("orderbook")
                exit_price = (
                    lookup_and_return(
                        orderbook,
                        "orderid",
                        info_dict[f"{sl_type}_stop_loss_order_ids"],
                        "averageprice",
                    )
                    .astype(float)
                    .mean()
                )
            info_dict[f"{sl_type}_exit_price"] = exit_price

            if move_sl_to_cost:
                info_dict[f"{other_side}_stop_loss_price"] = info_dict[
                    f"{other_side}_avg_price"
                ]
                if info_dict[f"{other_side}_stop_loss_order_ids"] is not None:
                    cancel_pending_orders(
                        info_dict[f"{other_side}_stop_loss_order_ids"], "STOPLOSS"
                    )
                    option_to_repair = (
                        traded_strangle.call_option
                        if other_side == "call"
                        else traded_strangle.put_option
                    )
                    info_dict[
                        f"{other_side}_stop_loss_order_ids"
                    ] = place_option_order_and_notify(
                        instrument=option_to_repair,
                        action="BUY",
                        qty_in_lots=quantity_in_lots,
                        prices=info_dict[f"{other_side}_stop_loss_price"],
                        order_tag=f"{other_side} SL Strangle",
                        webhook_url=notification_url,
                        stop_loss_order=True,
                        target_status="trigger pending",
                        return_avg_price=False,
                    )

            # Starting the trend catcher
            if catch_trend:
                trend_thread = Thread(
                    target=trend_catcher,
                    args=(
                        info_dict,
                        sl_type,
                        trend_qty_ratio,
                        trend_sl,
                        trend_strike_offset,
                    ),
                )
                trend_thread.start()

            refresh_orderbook = True
            # Wait for exit or other stop loss to hit
            while all([currenttime().time() < time(*exit_time)]):
                check_for_stop_loss(
                    info_dict, other_side, refresh_orderbook=refresh_orderbook
                )
                if info_dict[f"{other_side}_sl"]:
                    if info_dict[f"{other_side}_stop_loss_order_ids"] is None:
                        other_sl_option = (
                            traded_strangle.call_option
                            if other_side == "call"
                            else traded_strangle.put_option
                        )
                        notifier(
                            f"{self.name} strangle {other_side} stop loss hit.",
                            notification_url,
                        )
                        other_exit_price = place_option_order_and_notify(
                            other_sl_option,
                            "BUY",
                            quantity_in_lots,
                            "LIMIT",
                            order_tag,
                            notification_url,
                        )
                    else:
                        orderbook = fetch_book("orderbook")
                        other_exit_price = (
                            lookup_and_return(
                                orderbook,
                                "orderid",
                                info_dict[f"{other_side}_stop_loss_order_ids"],
                                "averageprice",
                            )
                            .astype(float)
                            .mean()
                        )
                    info_dict[f"{other_side}_exit_price"] = other_exit_price
                    break
                refresh_orderbook = False
                sleep(1)

        # Entering the main function

        # Setting strikes and expiry
        order_tag = "Intraday strangle"
        underlying_ltp = self.fetch_ltp()

        expiry = self.current_expiry

        # Setting stop loss
        stop_loss_dict = {
            "fixed": {"BANKNIFTY": 1.7, "NIFTY": 1.5},
            "dynamic": {"BANKNIFTY": 1.7, "NIFTY": 1.5},
        }

        if isinstance(stop_loss, str):
            if stop_loss == "dynamic" and timetoexpiry(expiry, in_days=True) < 1:
                stop_loss = 1.7
            else:
                stop_loss = stop_loss_dict[stop_loss].get(self.name, 1.6)
        else:
            stop_loss = stop_loss

        if strike_selection == "equal":
            strangle = self.most_equal_strangle(
                call_strike_offset=call_strike_offset,
                put_strike_offset=put_strike_offset,
                disparity_threshold=disparity_threshold,
                exit_time=(
                    datetime.combine(datetime.now().date(), time(*exit_time))
                    - timedelta(minutes=5)
                ).time(),
                expiry=expiry,
            )
            if strangle is None:
                notifier(
                    f"{self.name} no strangle found within disparity threshold {disparity_threshold}",
                    notification_url,
                )
                return
        elif strike_selection == "resilient":
            strangle = self.most_resilient_strangle(stop_loss=stop_loss, expiry=expiry)
        else:
            raise ValueError(f"Invalid find mode: {strike_selection}")

        call_ltp, put_ltp = strangle.fetch_ltp()

        # Placing the main order
        call_avg_price, put_avg_price = place_option_order_and_notify(
            strangle,
            "SELL",
            quantity_in_lots,
            "LIMIT",
            order_tag,
            notification_url,
            return_avg_price=True,
        )
        call_avg_price = (
            call_ltp
            if np.isnan(call_avg_price) or call_avg_price == 0
            else call_avg_price
        )
        put_avg_price = (
            put_ltp if np.isnan(put_avg_price) or put_avg_price == 0 else put_avg_price
        )
        total_avg_price = call_avg_price + put_avg_price

        call_stop_loss_price = (
            call_avg_price * call_stop_loss
            if call_stop_loss
            else call_avg_price * stop_loss
        )
        put_stop_loss_price = (
            put_avg_price * put_stop_loss
            if put_stop_loss
            else put_avg_price * stop_loss
        )

        # Logging information and sending notification
        trade_log = {
            "Time": currenttime().strftime("%d-%m-%Y %H:%M:%S"),
            "Index": self.name,
            "Call strike": strangle.call_strike,
            "Put strike": strangle.put_strike,
            "Expiry": expiry,
            "Action": "SELL",
            "Call price": call_avg_price,
            "Put price": put_avg_price,
            "Total price": total_avg_price,
            "Order tag": order_tag,
        }

        summary_message = "\n".join(f"{k}: {v}" for k, v in trade_log.items())

        traded_call_iv, traded_put_iv, traded_avg_iv = strangle_iv(
            callprice=call_avg_price,
            putprice=put_avg_price,
            callstrike=strangle.call_strike,
            putstrike=strangle.put_strike,
            spot=underlying_ltp,
            timeleft=timetoexpiry(expiry),
        )

        time_left_at_trade = timetoexpiry(expiry)
        summary_message += (
            f"\nTraded IVs: {traded_call_iv}, {traded_put_iv}, {traded_avg_iv}"
        )
        summary_message += (
            f"\nCall SL: {call_stop_loss_price}, Put SL: {put_stop_loss_price}"
        )
        notifier(summary_message, notification_url)

        if place_sl_orders:
            call_stop_loss_order_ids = place_option_order_and_notify(
                instrument=strangle.call_option,
                action="BUY",
                qty_in_lots=quantity_in_lots,
                prices=call_stop_loss_price,
                order_tag="Call SL Strangle",
                webhook_url=notification_url,
                stop_loss_order=True,
                target_status="trigger pending",
                return_avg_price=False,
            )
            put_stop_loss_order_ids = place_option_order_and_notify(
                instrument=strangle.put_option,
                action="BUY",
                qty_in_lots=quantity_in_lots,
                prices=put_stop_loss_price,
                order_tag="Put SL Strangle",
                webhook_url=notification_url,
                stop_loss_order=True,
                target_status="trigger pending",
                return_avg_price=False,
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
            "total_avg_price": total_avg_price,
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
            "exit_triggers": {"convert_to_butterfly": False},
            "trade_complete": False,
            "call_sl_check_notification_sent": False,
            "put_sl_check_notification_sent": False,
        }

        position_monitor_thread = Thread(
            target=position_monitor, args=(shared_info_dict,)
        )
        position_monitor_thread.start()
        sleep(3)  # To ensure that the position monitor thread has started

        refresh_book = True

        # Wait for exit time or both stop losses to hit (Main Loop)
        while all(
            [
                currenttime().time() < time(*exit_time),
                not any(shared_info_dict["exit_triggers"].values()),
            ]
        ):
            check_for_stop_loss(
                shared_info_dict, "call", refresh_orderbook=refresh_book
            )
            if shared_info_dict["call_sl"]:
                process_stop_loss(shared_info_dict, "call")
                break
            check_for_stop_loss(shared_info_dict, "put")
            if shared_info_dict["put_sl"]:
                process_stop_loss(shared_info_dict, "put")
                break
            refresh_book = False
            sleep(1)

        # Out of the while loop, so exit time reached or both stop losses hit, or we are hedged

        # If we are hedged then wait till exit time
        # noinspection PyTypeChecker
        if shared_info_dict["exit_triggers"]["convert_to_butterfly"]:
            hedge_strangle = shared_info_dict["ctb_hedge"]
            place_option_order_and_notify(
                hedge_strangle,
                "BUY",
                quantity_in_lots,
                "LIMIT",
                order_tag,
                notification_url,
                return_avg_price=False,
            )
            if place_sl_orders:
                cancel_pending_orders(
                    shared_info_dict["call_stop_loss_order_ids"]
                    + shared_info_dict["put_stop_loss_order_ids"]
                )
            notifier(f"{self.name}: Converted to butterfly", notification_url)
            while currenttime().time() < time(*exit_time):
                sleep(3)

        call_sl = shared_info_dict["call_sl"]
        put_sl = shared_info_dict["put_sl"]

        if not call_sl and not put_sl:  # Both stop losses not hit
            if shared_info_dict["time_left_day_start"] * 365 < 1:  # expiry day
                call_exit_avg_price, put_exit_avg_price = (
                    shared_info_dict["call_ltp"],
                    shared_info_dict["put_ltp"],
                )
            else:
                call_exit_avg_price, put_exit_avg_price = place_option_order_and_notify(
                    strangle,
                    "BUY",
                    quantity_in_lots,
                    "LIMIT",
                    order_tag,
                    notification_url,
                    return_avg_price=True,
                )
            # noinspection PyTypeChecker
            if (
                place_sl_orders
                and not shared_info_dict["exit_triggers"]["convert_to_butterfly"]
            ):
                cancel_pending_orders(
                    shared_info_dict["call_stop_loss_order_ids"]
                    + shared_info_dict["put_stop_loss_order_ids"]
                )
            shared_info_dict["call_exit_price"] = call_exit_avg_price
            shared_info_dict["put_exit_price"] = put_exit_avg_price

        elif (call_sl or put_sl) and not (call_sl and put_sl):  # Only one stop loss hit
            exit_option_type: str = "put" if call_sl else "call"
            if shared_info_dict["time_left_day_start"] * 365 < 1:  # expiry day
                non_sl_exit_price = shared_info_dict[f"{exit_option_type}_ltp"]
            else:
                exit_option = strangle.put_option if call_sl else strangle.call_option
                non_sl_exit_price = place_option_order_and_notify(
                    exit_option,
                    "BUY",
                    quantity_in_lots,
                    "LIMIT",
                    order_tag,
                    notification_url,
                )
            if place_sl_orders:
                cancel_pending_orders(
                    shared_info_dict[f"{exit_option_type}_stop_loss_order_ids"]
                )
            shared_info_dict[f"{exit_option_type}_exit_price"] = non_sl_exit_price

        else:  # Both stop losses hit
            pass

        # Calculate profit
        total_exit_price = (
            shared_info_dict["call_exit_price"] + shared_info_dict["put_exit_price"]
        )
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
            "Call stop loss": shared_info_dict["call_sl"],
            "Put stop loss": shared_info_dict["put_sl"],
        }

        notifier(exit_message, notification_url)
        shared_info_dict["trade_complete"] = True
        position_monitor_thread.join()

        trade_log.update(exit_dict)
        self.strategy_log[order_tag].append(trade_log)

        return shared_info_dict

    @log_errors
    def intraday_trend(
        self,
        quantity_in_lots,
        start_time=(9, 15, 58),
        exit_time=(15, 27),
        sleep_time=5,
        threshold_movement=None,
        seconds_to_avg=45,
        beta=0.8,
        max_entries=3,
        notification_url=None,
    ):
        strategy_tag = "Intraday trend"

        while currenttime().time() < time(*start_time):
            sleep(1)

        open_price = self.fetch_ltp()
        threshold_movement = (
            threshold_movement or (get_current_vix() * (beta or 1)) / 48
        )

        exit_time = time(*exit_time)
        scan_end_time = (
            datetime.combine(currenttime().date(), exit_time) - timedelta(minutes=10)
        ).time()
        price_boundaries = [
            open_price * (1 + ((-1) ** i) * threshold_movement / 100) for i in range(2)
        ]

        # Price deque
        n_prices = max(int(seconds_to_avg / sleep_time), 1)
        price_deque = deque(maxlen=n_prices)

        notifier(
            f"{self.name} trender starting with {threshold_movement:0.2f} threshold movement\n"
            f"Current Price: {open_price}\nUpper limit: {price_boundaries[0]:0.2f}\n"
            f"Lower limit: {price_boundaries[1]:0.2f}.",
            notification_url,
        )

        entries = 0
        last_print_time = currenttime()
        movement = 0
        entries_log = []
        while entries < max_entries and currenttime().time() < exit_time:
            # Scan for entry condition
            notifier(
                f"{self.name} trender {entries+1} scanning for entry condition.",
                notification_url,
            )
            while (abs(movement) < threshold_movement) and (
                currenttime().time() < scan_end_time
            ):
                ltp = self.fetch_ltp()
                price_deque.append(ltp)
                avg_price = (
                    (sum(price_deque) / len(price_deque)) if price_deque else ltp
                )
                movement = (avg_price - open_price) / open_price * 100

                if currenttime() > last_print_time + timedelta(minutes=1):
                    print(f"{self.name} trender: {movement:0.2f} movement.")
                    last_print_time = currenttime()
                sleep(sleep_time)

            if currenttime().time() > scan_end_time:
                notifier(
                    f"{self.name} trender {entries+1} exiting due to time.",
                    notification_url,
                )
                return

            # Entry condition met taking position
            price = self.fetch_ltp()
            atm_strike = findstrike(price, self.base)
            position = "BUY" if movement > 0 else "SELL"
            atm_synthetic_fut = SyntheticFuture(
                atm_strike, self.name, self.current_expiry
            )
            stop_loss_price = price * (0.997 if position == "BUY" else 1.003)
            stop_loss_hit = False
            notifier(
                f"{self.name} {position} trender triggered with {movement:0.2f} movement. {self.name} at {price}. "
                f"Stop loss at {stop_loss_price}.",
                notification_url,
            )
            call_entry_price, put_entry_price = place_option_order_and_notify(
                atm_synthetic_fut,
                position,
                quantity_in_lots,
                "LIMIT",
                strategy_tag,
                notification_url,
                return_avg_price=True,
            )

            if call_entry_price == 0 or put_entry_price == 0:
                call_entry_price, put_entry_price = atm_synthetic_fut.fetch_ltp()

            entry_price = atm_strike + call_entry_price - put_entry_price
            spot_future_basis = entry_price - price

            notifier(
                f"{self.name} trender {entries+1} entry price: {entry_price}, "
                f"spot-future basis: {spot_future_basis}",
                notification_url,
            )

            trade_info = {
                "Entry time": currenttime().strftime("%d-%m-%Y %H:%M:%S"),
                "Position": position,
                "Spot price": price,
                "Entry price": entry_price,
                "Spot-future basis": spot_future_basis,
                "Stop loss": stop_loss_price,
                "Threshold movement": threshold_movement,
                "Movement": movement,
            }

            # Tracking position
            while currenttime().time() < exit_time and not stop_loss_hit:
                ltp = self.fetch_ltp()
                price_deque.append(ltp)
                avg_price = sum(price_deque) / len(price_deque) if price_deque else ltp
                movement = (avg_price - open_price) / open_price * 100

                stop_loss_hit = (
                    (avg_price < stop_loss_price)
                    if position == "BUY"
                    else (avg_price > stop_loss_price)
                )
                sleep(sleep_time)

            # Exit condition met exiting position (stop loss or time)
            price = self.fetch_ltp()
            stop_loss_message = f"Trender stop loss hit. " if stop_loss_hit else ""
            notifier(
                f"{stop_loss_message}{self.name} trender {entries+1} exiting. {self.name} at {price}.",
                notification_url,
            )
            call_exit_price, put_exit_price = place_option_order_and_notify(
                atm_synthetic_fut,
                "BUY" if position == "SELL" else "SELL",
                quantity_in_lots,
                "LIMIT",
                strategy_tag,
                notification_url,
            )

            if call_exit_price == 0 or put_exit_price == 0:
                call_exit_price, put_exit_price = atm_synthetic_fut.fetch_ltp()

            exit_price = atm_strike + call_exit_price - put_exit_price
            pnl = (
                (exit_price - entry_price)
                if position == "BUY"
                else (entry_price - exit_price)
            )
            spot_future_basis = exit_price - price

            notifier(
                f"{self.name} trender {entries+1} exit price: {exit_price}, "
                f"spot-future basis: {spot_future_basis}, pnl: {pnl}",
                notification_url,
            )

            trade_info.update(
                {
                    "Exit time": currenttime().strftime("%d-%m-%Y %H:%M:%S"),
                    "Stop loss hit": stop_loss_hit,
                    "Spot exit price": price,
                    "Exit price": exit_price,
                    "Spot-future basis": spot_future_basis,
                    "PnL": pnl,
                }
            )
            entries_log.append(trade_info)
            entries += 1

        self.strategy_log[strategy_tag].extend(entries_log)


class Stock(Index):
    def __init__(self, name, spot_future_difference=0.06):
        super().__init__(name, spot_future_difference)


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


def append_data_to_json(new_data: defaultdict | dict | list, file_name: str) -> None:
    if new_data is None:
        return

    # Attempt to read the existing data from the JSON file
    try:
        with open(file_name, "r") as file:
            existing_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        # If the file doesn't exist or has invalid JSON content, create an empty list and write it to the file
        existing_data = []
        with open(file_name, "w") as file:
            json.dump(existing_data, file)

    # Convert the defaultdict to a regular dict, make it JSON serializable, and append it to the list

    if isinstance(new_data, (defaultdict, dict)):
        if isinstance(new_data, defaultdict):
            new_data = dict(new_data)
        serialized_data = convert_to_serializable(new_data)
        existing_data.append(serialized_data)
    elif isinstance(new_data, list):
        serialized_data = convert_to_serializable(new_data)
        existing_data.extend(serialized_data)
    else:
        raise TypeError("New data must be a defaultdict, dict, or list.")

    # Write the updated data back to the JSON file with indentation
    with open(file_name, "w") as file:
        json.dump(existing_data, file, indent=4, default=str)


def word_to_num(s):
    word = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
        "thirty": 30,
        "forty": 40,
        "fifty": 50,
        "sixty": 60,
        "seventy": 70,
        "eighty": 80,
        "ninety": 90,
    }
    multiplier = {
        "thousand": 1000,
        "hundred": 100,
        "million": 1000000,
        "billion": 1000000000,
    }

    words = s.lower().split()
    if words[0] == "a":
        words[0] = "one"
    total = 0
    current = 0
    for w in words:
        if w in word:
            current += word[w]
        if w in multiplier:
            current *= multiplier[w]
        if w == "and":
            continue
        if w == "thousand" or w == "million" or w == "billion":
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


def lookup_and_return(
    book, field_to_lookup, value_to_lookup, field_to_return
) -> np.array:
    def filter_and_return(data: list):
        if not isinstance(field_to_lookup, (list, tuple, np.ndarray)):
            field_to_lookup_ = [field_to_lookup]
            value_to_lookup_ = [value_to_lookup]
        else:
            field_to_lookup_ = field_to_lookup
            value_to_lookup_ = value_to_lookup

        if isinstance(
            field_to_return, (list, tuple, np.ndarray)
        ):  # Return a dict as multiple fields are requested
            bucket = {field: [] for field in field_to_return}
            for entry in data:
                if all(
                    (
                        entry[field] == value
                        if not isinstance(value, (list, tuple, np.ndarray))
                        else entry[field] in value
                    )
                    for field, value in zip(field_to_lookup_, value_to_lookup_)
                ) and all(entry[field] != "" for field in field_to_lookup_):
                    for field in field_to_return:
                        bucket[field].append(entry[field])

            if all(len(v) == 0 for v in bucket.values()):
                return {}
            else:
                # Flatten the dictionary if all fields contain only one value
                if all(len(v) == 1 for v in bucket.values()):
                    bucket = {k: v[0] for k, v in bucket.items()}
                return bucket
        else:  # Return a numpy array as only one field is requested
            # Check if 'orderid' is in field_to_lookup_
            if "orderid" in field_to_lookup_:
                sort_by_orderid = True
                orderid_index = field_to_lookup_.index("orderid")
            else:
                sort_by_orderid = False
                orderid_index = None

            bucket = [
                (entry["orderid"], entry[field_to_return])
                if sort_by_orderid
                else entry[field_to_return]
                for entry in data
                if all(
                    (
                        entry[field] == value
                        if not isinstance(value, (list, tuple, np.ndarray))
                        else entry[field] in value
                    )
                    for field, value in zip(field_to_lookup_, value_to_lookup_)
                )
                and all(entry[field] != "" for field in field_to_lookup_)
            ]

            if len(bucket) == 0:
                return np.array([])
            else:
                if sort_by_orderid:
                    # Create a dict mapping order ids to their index in value_to_lookup
                    orderid_to_index = {
                        value: index
                        for index, value in enumerate(value_to_lookup_[orderid_index])
                    }
                    # Sort the bucket based on the order of 'orderid' in value_to_lookup
                    bucket.sort(key=lambda x: orderid_to_index[x[0]])
                    # Return only the field_to_return values
                    return np.array([x[1] for x in bucket])
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


def check_and_notify_order_placement_statuses(
    statuses, target_status="complete", webhook_url=None, **kwargs
) -> str:
    order_prefix = (
        f"{kwargs['order_tag']}: "
        if ("order_tag" in kwargs and kwargs["order_tag"])
        else ""
    )
    order_message = [f"{k}-{v}" for k, v in kwargs.items() if k != "order_tag"]
    order_message = ", ".join(order_message)

    if all(statuses == target_status):
        notifier(
            f"{order_prefix}Order(s) placed successfully for {order_message}",
            webhook_url,
        )
        return "all complete"
    elif all(statuses == "rejected"):
        notifier(f"{order_prefix}All orders rejected for {order_message}", webhook_url)
        raise Exception("Orders rejected")
    elif all(statuses == "open"):
        notifier(f"{order_prefix}All orders pending for {order_message}", webhook_url)
        return "all open"
    elif any(statuses == "open"):
        notifier(
            f"{order_prefix}Some orders pending for {order_message}. You can modify the orders.",
            webhook_url,
        )
        return "some open"
    elif any(statuses == "rejected"):
        notifier(
            f"{order_prefix}Some orders rejected for {order_message}.\nYou can place the rejected orders again.",
            webhook_url,
        )
        return "some rejected"
    else:
        notifier(
            f"{order_prefix}ERROR. Order statuses uncertain for {order_message}",
            webhook_url,
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
    **kwargs,
):
    def return_avg_price_from_orderbook(orderbook, ids):
        avg_prices = lookup_and_return(
            orderbook, ["orderid", "status"], [ids, "complete"], "averageprice"
        )
        if avg_prices.size > 0:
            return avg_prices.astype(float).mean()
        else:
            return None

    notify_dict = {
        "order_tag": order_tag,
        "Underlying": instrument.underlying,
        "Action": action,
        "Expiry": instrument.expiry,
        "Qty": qty_in_lots,
    }

    order_params = {
        "transaction_type": action,
        "quantity_in_lots": qty_in_lots,
        "stop_loss_order": stop_loss_order,
        "order_tag": order_tag,
    }

    if isinstance(instrument, (Strangle, Straddle, SyntheticFuture)):
        notify_dict.update({"Strikes": [instrument.call_strike, instrument.put_strike]})
        order_params.update({"prices": prices})
    elif isinstance(instrument, Option):
        notify_dict.update(
            {"Strike": instrument.strike, "OptionType": instrument.option_type}
        )
        order_params.update({"price": prices})
    else:
        raise ValueError("Invalid instrument type")

    notify_dict.update(kwargs)

    if stop_loss_order:
        assert isinstance(
            prices, (int, float, tuple, list, np.ndarray)
        ), "Stop loss order requires a price"
        target_status = "trigger pending"

    # Placing the order
    order_ids = instrument.place_order(**order_params)

    if isinstance(order_ids, tuple):  # Strangle/Straddle/SyntheticFuture
        call_order_ids, put_order_ids = order_ids[0], order_ids[1]
        order_ids = list(itertools.chain(call_order_ids, put_order_ids))
    else:  # Option
        call_order_ids, put_order_ids = False, False

    order_book = fetch_book("orderbook")
    order_statuses_ = lookup_and_return(order_book, "orderid", order_ids, "status")
    placement_status = check_and_notify_order_placement_statuses(
        statuses=order_statuses_,
        target_status=target_status,
        webhook_url=webhook_url,
        **notify_dict,
    )

    if return_avg_price:
        if call_order_ids and put_order_ids:  # Strangle/Straddle/SyntheticFuture
            call_ltp, put_ltp = instrument.fetch_ltp()
            if placement_status == "all open":
                call_avg_price, put_avg_price = call_ltp, put_ltp
            else:
                call_avg_price = (
                    return_avg_price_from_orderbook(order_book, call_order_ids)
                    or call_ltp
                )
                put_avg_price = (
                    return_avg_price_from_orderbook(order_book, put_order_ids)
                    or put_ltp
                )

            return call_avg_price, put_avg_price
        else:  # Option
            ltp = instrument.fetch_ltp()
            if placement_status == "all open":
                avg_price = ltp
            else:
                avg_price = (
                    return_avg_price_from_orderbook(order_book, order_ids) or ltp
                )
            return avg_price

    return order_ids


def process_stop_loss_order_statuses(
    order_book,
    order_ids,
    context="",
    notify_url=None,
):
    pending_text = "trigger pending"
    context = f"{context} " if context else ""

    statuses = lookup_and_return(order_book, "orderid", order_ids, "status")

    if not isinstance(statuses, np.ndarray) or statuses.size == 0:
        logger.error(f"Statuses is {statuses} for orderid(s) {order_ids}")

    if all(statuses == pending_text):
        return False, False

    elif all(statuses == "rejected") or all(statuses == "cancelled"):
        rejection_reasons = lookup_and_return(order_book, "orderid", order_ids, "text")
        if all(rejection_reasons == "17070 : The Price is out of the LPP range"):
            return True, False
        else:
            notifier(
                f"{context}Order rejected or cancelled. Reasons: {rejection_reasons[0]}",
                notify_url,
            )
            raise Exception(f"Orders rejected or cancelled.")

    elif all(statuses == "pending"):
        sleep(5)
        order_book = fetch_book("orderbook")
        statuses = lookup_and_return(order_book, "orderid", order_ids, "status")

        if all(statuses == "pending"):
            try:
                cancel_pending_orders(order_ids, "NORMAL")
            except Exception as e:
                try:
                    cancel_pending_orders(order_ids, "STOPLOSS")
                except Exception as e:
                    notifier(f"{context}Could not cancel orders: {e}", notify_url)
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


def fetch_orderbook_if_needed(data_class=None, refresh_needed: bool = False):
    if data_class is None or refresh_needed:
        return fetch_book("orderbook")
    if (
        currenttime() - data_class.updated_time < timedelta(seconds=15)
        and data_class.orderbook_data is not None
    ):
        return data_class.orderbook_data
    return fetch_book("orderbook")


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
    price_func = bs.put if flag == "p" else bs.call

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
        logger.info(
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
    upper_bound = np.percentile(strike_array, 90)
    lower_bound = np.percentile(strike_array, 30)
    strike_array = strike_array[
        strike_array.between(lower_bound, upper_bound, inclusive="both")
    ]
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
                (scrips.name == name)
                & (scrips.exch_seg == "NSE")
                & (scrips.instrumenttype != "AMXIDX")
            ]  # Temp fix for AMXIDX

            if len(filtered_scrips) != 1:
                logger.error(
                    f"Could not find symbol, token for {name} from scrips file."
                )
                if len(filtered_scrips) == 0:
                    logger.error(f"No scrips found: {filtered_scrips}")
                else:
                    logger.error(f"Multiple scrips found: {filtered_scrips}")

                symbol, token = (
                    ("NIFTY", "26000") if name == "NIFTY" else ("BANKNIFTY", "26009")
                )  # Temp fix for NIFTY & BANKNIFTY. To be removed later.

            else:
                symbol, token = filtered_scrips[["symbol", "token"]].values[0]

        elif name in ["FINNIFTY", "MIDCPNIFTY"]:  # Finnifty & Midcpnifty temp fix
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
            assert (
                len(filtered_scrips) == 1
            ), "More than one equity scrip found for name."
            symbol, token = filtered_scrips[["symbol", "token"]].values[0]

    elif (
        expiry is not None and strike is not None and option_type is not None
    ):  # Options segment
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
    option_type=None,
):
    """Available intervals:

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
        interval_unit = (
            interval_unit + "s" if interval_unit[-1] != "s" else interval_unit
        )
        interval_digit = word_to_num(interval_digit)
        time_delta = interval_digit * last_n_intervals
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
    data.set_index(pd.Series(data.iloc[:, 0], name="date"), inplace=True)
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


def get_strangle_indices_to_trade(*indices, safe_indices=None):
    if safe_indices is None:
        safe_indices = ["NIFTY", "BANKNIFTY"]

    times_to_expiries = [
        timetoexpiry(index.current_expiry, effective_time=True, in_days=True)
        for index in indices
    ]

    # Check if any index has less than 1 day to expiry
    indices_less_than_1_day = [
        index
        for index, time_to_expiry in zip(indices, times_to_expiries)
        if time_to_expiry < 1
    ]

    if indices_less_than_1_day:
        return indices_less_than_1_day

    # If no index has less than 1 day to expiry
    min_expiry_time = min(times_to_expiries)
    indices_with_closest_expiries = [
        index
        for index, time_to_expiry in zip(indices, times_to_expiries)
        if time_to_expiry == min_expiry_time
    ]
    closest_index_names = [index.name for index in indices_with_closest_expiries]
    weekend_in_range = check_for_weekend(
        indices_with_closest_expiries[0].current_expiry
    )

    if "MIDCPNIFTY" in closest_index_names:
        indices_without_midcp = [
            index for index in indices if index.name != "MIDCPNIFTY"
        ]
        return get_strangle_indices_to_trade(*indices_without_midcp)

    if "FINNIFTY" in closest_index_names and weekend_in_range:
        return [index for index in indices if index.name in safe_indices]

    return indices_with_closest_expiries


def indices_to_trade(nifty, bnf, finnifty, multi_before_weekend=False):  # delete this
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
    return round(time_to_expiry * multiplier, 5)


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
        data_frame["atm_strike"] = data_frame.apply(
            lambda row: findstrike(row.spot, 50)
            if row.symbol == "NIFTY"
            else findstrike(row.spot, 100),
            axis=1,
        )
        data_frame["strike_iv"] = np.where(
            data_frame.strike > data_frame.atm_strike,
            data_frame.call_iv,
            np.where(
                data_frame.strike < data_frame.atm_strike,
                data_frame.put_iv,
                data_frame.avg_iv,
            ),
        )
        data_frame["atm_iv"] = data_frame.apply(
            lambda row: data_frame[
                (data_frame.strike == row.atm_strike)
                & (data_frame.expiry == row.expiry)
            ].strike_iv.values[0],
            axis=1,
        )
        data_frame.sort_values(["symbol", "expiry", "strike"], inplace=True)
        data_frame["distance"] = data_frame["strike"] / data_frame["spot"] - 1
        data_frame["iv_multiple"] = data_frame["strike_iv"] / data_frame["atm_iv"]
        data_frame["distance_squared"] = data_frame["distance"] ** 2

        return data_frame

    symbol_dfs = []
    for symbol in option_chains:
        spot_price = option_chains[symbol].underlying_price
        expiry_dfs = []
        for expiry in option_chains[symbol]:
            df = pd.DataFrame(option_chains[symbol][expiry]).T
            df.index = df.index.set_names("strike")
            df = df.reset_index()
            df["spot"] = spot_price
            df["expiry"] = expiry
            df["symbol"] = symbol
            df["time_to_expiry"] = timetoexpiry(expiry)
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
        params.update(
            {
                "variety": "STOPLOSS",
                "ordertype": "STOPLOSS_LIMIT",
                "triggerprice": round(price, 1),
                "price": round(execution_price, 1),
            }
        )
    else:
        order_type, execution_price = (
            ("MARKET", 0) if price == "MARKET" else ("LIMIT", price)
        )
        if order_type == "LIMIT":
            if execution_price < 10 and qty < 6000:
                execution_price = (
                    np.ceil(price) if action == "BUY" else max(np.floor(price), 0.05)
                )

        params.update(
            {
                "variety": "NORMAL",
                "ordertype": order_type,
                "price": custom_round(execution_price),
            }
        )

    for attempt in range(1, 4):
        try:
            return obj.placeOrder(params)
        except Exception as e:
            if attempt == 3:
                raise e
            print(
                f"Error {attempt} in placing {'stop-loss ' if stop_loss_order else ''}order for {symbol}: {e}"
            )
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


def handle_open_orders(*order_ids, action, modify_percentage=0.01, stage=0):
    """Modifies orders if they are pending by the provided modification percentage"""
    print(
        f"\nEntering handle_open_orders with order_ids: {order_ids}, "
        f"modify_percentage: {modify_percentage}, stage: {stage}"
    )

    if stage >= 10:
        print("Stage >= 10, exiting function without modifying any orders")
        return None

    stage_increment = int(modify_percentage * 100)
    stage_increment = max(stage_increment, 1)
    print(f"Calculated stage_increment as: {stage_increment}")

    order_book = fetch_book("orderbook")
    sleep(1)

    statuses = lookup_and_return(order_book, "orderid", order_ids, "status")
    print(f"Looked up order statuses: {statuses}")

    if all(statuses == "complete"):
        print("All orders are complete, exiting function without modifying any orders")
        return None
    elif any(np.isin(statuses, ["rejected", "cancelled"])):
        print(
            "Some orders are rejected or cancelled, exiting function without modifying any orders"
        )
        return None
    elif any(statuses == "open"):
        print("Some orders are open, proceeding with modifications")

        open_order_ids = [
            order_id
            for order_id, status in zip(order_ids, statuses)
            if status == "open"
        ]
        print(f"Open order ids: {open_order_ids}")

        for order_id in open_order_ids:
            relevant_fields = [
                "orderid",
                "variety",
                "symboltoken",
                "price",
                "ordertype",
                "producttype",
                "exchange",
                "tradingsymbol",
                "quantity",
                "duration",
                "status",
            ]

            current_params = lookup_and_return(
                order_book, "orderid", order_id, relevant_fields
            )

            old_price = current_params["price"]

            new_price = (
                old_price * (1 + modify_percentage)
                if action == "BUY"
                else old_price * (1 - modify_percentage)
            )
            new_price = custom_round(new_price)

            modified_params = current_params.copy()
            modified_params["price"] = new_price
            modified_params.pop("status")

            obj.modifyOrder(modified_params)
            print(
                f"Modified order {order_id} with new price: {new_price} from old price: {old_price}"
            )

        order_book = fetch_book("orderbook")
        sleep(1)

        statuses = lookup_and_return(order_book, "orderid", open_order_ids, "status")
        print(f"Looked up order statuses after modifications: {statuses}")

        if any(statuses == "open"):
            print("Some orders are still open, recalling function with increased stage")
            return handle_open_orders(
                *open_order_ids,
                action=action,
                modify_percentage=modify_percentage,
                stage=stage + stage_increment,
            )

        print("All orders are now complete or closed, exiting function")


def handle_open_orders_lite(
    *order_ids, action, modify_percentage=0.02, orderbook="orderbook", sleep_interval=1
):
    """Modifies orders if they are pending by the provided modification percentage"""

    iterations = 0.1 / modify_percentage
    iterations = int(iterations)
    iterations = max(iterations, 1)

    relevant_fields = [
        "orderid",
        "variety",
        "symboltoken",
        "price",
        "ordertype",
        "producttype",
        "exchange",
        "tradingsymbol",
        "quantity",
        "duration",
        "status",
    ]

    order_params = {
        order_id: lookup_and_return(orderbook, "orderid", order_id, relevant_fields)
        for order_id in order_ids
    }

    for i in range(iterations):
        for order_id in order_ids:
            old_price = order_params[order_id]["price"]

            increment = old_price * modify_percentage
            increment = max(increment, 0.1)
            new_price = (
                old_price + increment if action == "BUY" else old_price - increment
            )

            new_price = custom_round(new_price)

            modified_params = order_params[order_id].copy()
            modified_params["price"] = new_price
            order_params[order_id]["price"] = new_price
            modified_params.pop("status")

            try:
                obj.modifyOrder(modified_params)
            except Exception as e:
                logger.error(f"Error in modifying order: {e}")
            sleep(sleep_interval)


def cancel_pending_orders(order_ids, variety="STOPLOSS"):
    if isinstance(order_ids, (list, np.ndarray)):
        for order_id in order_ids:
            obj.cancelOrder(order_id, variety)
    else:
        obj.cancelOrder(order_ids, variety)
